from dataclasses import dataclass
from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from models.abstract import Base


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm matching CUDA implementation.

    No learnable scale/bias - only normalizes with fixed gain=1.
    Matches the CUDA implementation where only eps is configurable.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Compute RMS: sqrt(mean(x²) + eps)
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        rms = mx.sqrt(ms + self.eps)
        # Normalize (no learnable scale)
        return x / rms


@dataclass
class ModelConfig:
    in_channels: int
    depth: int
    dim: int
    heads: int
    patch_size: tuple
    n_outputs: int
    pool: str = "cls"
    n: int = 6  # latent steps
    T: int = 3  # deep steps
    halt_max_steps: int = 8  # maximum supervision steps
    halt_exploration_prob: float = 0.2  # exploratory q probability
    halt_follow_q: bool = True  # follow q (True) or max steps (False)


class SwiGLU(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(dim, mlp_dim * 2, bias=False)
        self.w2 = nn.Linear(mlp_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(self.dropout(nn.silu(gate) * up))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.scale = self.d**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(self.d)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, _ = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = map(
            lambda x: x.reshape(b, n, self.h, -1).transpose(0, 2, 1, 3), (q, k, v)
        )
        q, k = self.rope(q), self.rope(k)
        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out(x.transpose(0, 2, 1, 3).reshape(b, n, -1))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = RMSNorm()  # Parameter-free, matching CUDA
        self.attn = Attention(dim, heads)
        self.n2 = RMSNorm()  # Parameter-free, matching CUDA
        self.ff = SwiGLU(dim, 3 * dim)  # 3× expansion, matching CUDA (was 8/3× from LLaMA)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.n1(x + self.attn(x))
        x = self.n2(x + self.ff(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: tuple, dim: int, pool: str):
        super().__init__()
        p1, p2 = patch_size
        self.emb = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=(p1, p2),
            stride=(p1, p2),
            bias=True,
        )
        self.q_token = 0.02 * mx.random.normal((1, 1, dim))
        self.cls_token = 0.02 * mx.random.normal((1, 1, dim)) if pool == "cls" else None

    def __call__(self, x: mx.array) -> mx.array:
        b = x.shape[0]
        x = self.emb(x)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        if self.cls_token is not None:
            x = mx.concat([mx.repeat(self.cls_token, b, axis=0), x], axis=1)
        x = mx.concat([mx.repeat(self.q_token, b, axis=0), x], axis=1)
        return x  # b n+(1 or 2) d


class OutputHead(nn.Module):
    def __init__(self, dim: int, n_outputs: int, pool: str):
        super().__init__()
        self.pool = pool
        self.out = nn.Linear(dim, n_outputs)

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.mean(x, axis=1) if self.pool == "mean" else x[:, 0]
        return self.out(x)


class QHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.out = nn.Linear(dim, 1)
        self.out.weight[:] = 0
        self.out.bias[:] = -5.0

    def __call__(self, x: mx.array) -> mx.array:
        return self.out(x)


class Model(Base):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        assert config.pool in {"cls", "mean"}, "pool must be either 'cls' or 'mean'"
        self.config = config

        self.embed = PatchEmbedding(
            config.in_channels, config.patch_size, config.dim, config.pool
        )
        self.blocks = nn.Sequential(
            *[Block(config.dim, config.heads) for _ in range(config.depth)]
        )
        self.out_head = OutputHead(config.dim, config.n_outputs, config.pool)
        self.q_head = QHead(config.dim)

        self._y_init = mx.random.truncated_normal(-2, 2, (self.config.dim,))
        self._z_init = mx.random.truncated_normal(-2, 2, (self.config.dim,))

    def initial_carry(self, batch: Dict[str, mx.array]):
        b = batch["image"].shape[0]
        return dict(
            inner_carry=dict(
                y=mx.zeros((b, 1, self.config.dim)),
                z=mx.zeros((b, 1, self.config.dim)),
            ),
            steps=mx.zeros((b,), dtype=mx.int32),
            halted=mx.ones((b,), dtype=mx.bool_),
            current_data={k: mx.zeros_like(v) for k, v in batch.items()},
        )

    def reset_inner_carry(self, halted: mx.array, carry: dict):
        mask = halted.reshape(-1, 1, 1)
        return dict(
            y=mx.where(mask, self._y_init, carry["y"]),
            z=mx.where(mask, self._z_init, carry["z"]),
        )

    def latent_recursion(self, carry: dict, x: mx.array) -> dict:
        # NOTE: (b 1 d) only at first step
        y, z = carry["y"], carry["z"]
        for _ in range(self.config.n):
            z = self.blocks(z + y + x)
        y = self.blocks(y + z)
        carry["y"], carry["z"] = y, z
        return carry

    def deep_recursion(self, carry: dict, batch: Dict[str, mx.array]):
        x = self.embed(batch["image"])  # b n d
        for _ in range(self.config.T):
            carry = self.latent_recursion(carry, x)
        carry["y"] = mx.stop_gradient(carry["y"])
        carry["z"] = mx.stop_gradient(carry["z"])

        carry = self.latent_recursion(carry, x)

        outputs = {
            "logits": self.out_head(carry["y"][:, 1:]),
            "q_halt_logits": self.q_head(carry["y"][:, 0]).reshape(-1),
        }
        carry["y"] = mx.stop_gradient(carry["y"])
        carry["z"] = mx.stop_gradient(carry["z"])

        return carry, outputs

    def __call__(self, carry: dict, batch: Dict[str, mx.array]):
        new_inner_carry = self.reset_inner_carry(carry["halted"], carry["inner_carry"])
        new_steps = mx.where(carry["halted"], 0, carry["steps"])
        new_current_data = {
            k: mx.where(
                carry["halted"].reshape((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                mx.array(v),
            )
            for k, v in carry["current_data"].items()
        }

        new_inner_carry, outputs = self.deep_recursion(
            new_inner_carry, new_current_data
        )

        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        halted = is_last_step

        # Adaptive Computational Time (ACT)
        if (self.config.halt_max_steps > 1) and (
            self.training or self.config.halt_follow_q
        ):
            halted = halted | (outputs["q_halt_logits"] > 0)

        if (
            self.training
            and (self.config.halt_max_steps > 1)
            and (self.config.halt_exploration_prob > 0)
        ):  # Exploration
            min_halt_steps = (
                mx.random.uniform(shape=outputs["q_halt_logits"].shape)
                < self.config.halt_exploration_prob
            ) * mx.random.randint(
                low=2, high=self.config.halt_max_steps + 1, shape=new_steps.shape
            )
            halted = halted & (new_steps >= min_halt_steps)

        return (
            dict(
                inner_carry=new_inner_carry,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )


if __name__ == "__main__":
    mx.random.seed(0)

    x = mx.random.normal(shape=(10, 32, 32, 3))
    t = mx.random.randint(0, 10, shape=(10,))
    model = Model(
        config=ModelConfig(
            in_channels=x.shape[-1],
            depth=2,
            dim=32,
            heads=4,
            patch_size=(8, 8),
            n_outputs=10,
        )
    )
    model.summary()

    batch = {"image": x, "label": t}
    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)

    print("logits shape:", outputs["logits"].shape)
    print("q logits shape:", outputs["q_halt_logits"].shape)
