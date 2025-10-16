"""Construct a PyTorch analogue of the ARC TRM model and export it to CoreML."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from safetensors.numpy import load_file as safe_load
    SAFETENSORS_AVAILABLE = True
except Exception:  # pragma: no cover
    safe_load = None
    SAFETENSORS_AVAILABLE = False

try:
    import ml_dtypes
    BF16_DTYPE = getattr(ml_dtypes, "bfloat16")
except Exception:  # pragma: no cover
    ml_dtypes = None
    BF16_DTYPE = None
except Exception:  # pragma: no cover
    SAFETENSORS_AVAILABLE = False

# Optional MLX import for safetensors checkpoint conversion
try:
    import mlx.core as mx  # type: ignore
    from models.trm_arc import ARCModel as MLXARCModel
    from models.trm_arc import ARCModelConfig as MLXARCConfig
    MX_AVAILABLE = True
    mx.set_default_device(mx.cpu)
except Exception:  # pragma: no cover - fallback when MLX unavailable
    MX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Building blocks mirroring models/trm_arc.py
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    b, h, n, d = x.shape
    half = d // 2
    if half == 0:
        return x

    device = x.device
    dtype = x.dtype
    freq_seq = torch.arange(0, half, dtype=dtype, device=device)
    freq = 1.0 / (10000 ** (freq_seq / half))
    positions = torch.arange(0, n, dtype=dtype, device=device)
    theta = torch.outer(positions, freq)
    cos = torch.cos(theta)[None, None, :, :]
    sin = torch.sin(theta)[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class RoPEAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, d)
        return self.out(out)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        hidden = int(8 / 3 * dim)
        self.w1 = nn.Linear(dim, hidden * 2, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(self.dropout(F.silu(gate) * up))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = RoPEAttention(dim, heads, dropout)
        self.ff = SwiGLU(dim, dropout)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


@dataclass
class TorchARCConfig:
    vocab_size: int = 12
    seq_len: int = 900
    dim: int = 256
    depth: int = 2
    heads: int = 8
    n_cycles: int = 6
    t_cycles: int = 3
    dropout: float = 0.0
    puzzle_emb_dim: int = 16
    num_puzzles: int = 1000
    use_puzzle_ids: bool = True


class TokenEmbedding(nn.Module):
    def __init__(self, config: TorchARCConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        if config.use_puzzle_ids:
            self.puzzle_emb = nn.Embedding(config.num_puzzles, config.puzzle_emb_dim)
            self.puzzle_proj = nn.Linear(config.puzzle_emb_dim, config.dim, bias=False)
        else:
            self.puzzle_emb = None

    def forward(self, token_ids: torch.Tensor, puzzle_ids: torch.Tensor | None) -> torch.Tensor:
        x = self.token_emb(token_ids)
        if self.puzzle_emb is not None and puzzle_ids is not None:
            p = self.puzzle_emb(puzzle_ids)
            p = self.puzzle_proj(p).unsqueeze(1)
            x = x + p
        return x


class TorchARCModel(nn.Module):
    def __init__(self, config: TorchARCConfig):
        super().__init__()
        self.config = config
        self.embedding = TokenEmbedding(config)
        self.blocks = nn.Sequential(*[TransformerBlock(config.dim, config.heads, config.dropout)
                                      for _ in range(config.depth)])
        self.out_head = nn.Linear(config.dim, config.vocab_size)
        self.q_head = nn.Linear(config.dim, 1)

        self.q_token = nn.Parameter(torch.randn(1, 1, config.dim) * 0.02)
        self.register_buffer("y_init", torch.zeros(1, config.seq_len + 1, config.dim))
        self.register_buffer("z_init", torch.zeros(1, config.seq_len + 1, config.dim))

    def apply_blocks(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def latent_recursion(self, y: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(self.config.n_cycles):
            z = self.apply_blocks(z + y + x)
        y = self.apply_blocks(y + z)
        return y, z

    def forward(self, token_ids: torch.Tensor, puzzle_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq = token_ids.shape
        x = self.embedding(token_ids, puzzle_ids)
        q = self.q_token.expand(b, -1, -1)
        x = torch.cat([q, x], dim=1)

        y = self.y_init[:, :seq + 1, :].expand(b, -1, -1).clone()
        z = self.z_init[:, :seq + 1, :].expand(b, -1, -1).clone()

        for _ in range(self.config.t_cycles):
            y, z = self.latent_recursion(y, z, x)

        y, z = self.latent_recursion(y, z, x)

        logits = self.out_head(y[:, 1:, :])
        q_halt_logits = self.q_head(y[:, 0, :]).squeeze(-1)
        return logits, q_halt_logits


# ---------------------------------------------------------------------------
# Checkpoint loading helpers
# ---------------------------------------------------------------------------


def load_weights(torch_model: nn.Module, path: str | None, config: TorchARCConfig) -> None:
    if not path:
        return
    p = Path(path)
    if p.suffix in {".pt", ".pth"}:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        torch_model.load_state_dict(state, strict=True)
        print(f"Loaded PyTorch weights from {path}")
        return

    if p.suffix == ".safetensors":
        if not SAFETENSORS_AVAILABLE:
            raise RuntimeError("Install safetensors to load .safetensors checkpoints")

        arrays = safe_load(str(p))

        def map_ckpt_name(name: str) -> str | None:
            if name == "embed.q_token":
                return "q_token"
            if name.startswith("embed."):
                return "embedding." + name[len("embed."):]
            if name.startswith("out_head.out."):
                return "out_head." + name[len("out_head.out."):]
            if name.startswith("q_head.out."):
                return "q_head." + name[len("q_head.out."):]
            if name.startswith("blocks.layers."):
                parts = name.split(".")
                idx = parts[2]
                rest_parts = parts[3:]
                if rest_parts[0] == "n1":
                    rest_parts[0] = "norm1"
                elif rest_parts[0] == "n2":
                    rest_parts[0] = "norm2"
                rest = ".".join(rest_parts)
                return f"blocks.{idx}.{rest}"
            if name in {"y_init", "z_init"}:
                return None
            return name

        state_dict = torch_model.state_dict()
        new_state = state_dict.copy()
        loaded_keys = set()
        for ckpt_name, value in arrays.items():
            target = map_ckpt_name(ckpt_name)
            if not target or target not in state_dict:
                continue
            arr = np.array(value)
            if BF16_DTYPE is not None and arr.dtype == BF16_DTYPE:
                arr = arr.astype(np.float32)
            elif str(arr.dtype) == "bfloat16":
                arr = arr.astype(np.float32)
            else:
                arr = arr.astype(np.float32, copy=False)
            new_state[target] = torch.from_numpy(arr)
            loaded_keys.add(target)

        missing = [k for k in state_dict.keys() if k not in loaded_keys and k not in {"y_init", "z_init"}]
        missing = [k for k in missing if not k.startswith("embedding.puzzle_emb")]
        if missing:
            raise KeyError(f"Missing parameters in checkpoint {path}: {missing}")

        torch_model.load_state_dict(new_state, strict=True)
        print(f"Loaded safetensors weights from {path}")
        return

    raise ValueError(f"Unsupported checkpoint format: {path}")


def maybe_cast_fp16(model: nn.Module, precision: str) -> nn.Module:
    if precision == "float16":
        return model.half()
    return model


# ---------------------------------------------------------------------------
# CoreML export
# ---------------------------------------------------------------------------


def export_coreml(
    output_path: str,
    config: TorchARCConfig,
    precision: str = "float16",
    seed: int | None = None,
    weights_path: str | None = None,
    batch_size: int = 1,
):
    if seed is not None:
        torch.manual_seed(seed)

    model = TorchARCModel(config)
    load_weights(model, weights_path, config)
    model = maybe_cast_fp16(model, precision)
    model.eval()

    example_tokens = torch.randint(0, config.vocab_size, (batch_size, config.seq_len), dtype=torch.int32)
    if config.use_puzzle_ids:
        example_puzzles = torch.randint(0, config.num_puzzles, (batch_size,), dtype=torch.int32)
        example_inputs = (example_tokens, example_puzzles)
        input_types = [
            ct.TensorType(name="token_ids", shape=example_tokens.shape, dtype=int),
            ct.TensorType(name="puzzle_ids", shape=example_puzzles.shape, dtype=int),
        ]
    else:
        example_inputs = (example_tokens,)
        input_types = [ct.TensorType(name="token_ids", shape=example_tokens.shape, dtype=int)]

    with torch.no_grad():
        traced = torch.jit.trace(model, example_inputs)

    convert_kwargs = {"inputs": input_types}
    if precision == "float16":
        convert_kwargs["minimum_deployment_target"] = ct.target.iOS16
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16
    elif precision == "float32":
        convert_kwargs["minimum_deployment_target"] = ct.target.iOS13
        convert_kwargs["compute_precision"] = ct.precision.FLOAT32
    else:
        raise ValueError("precision must be 'float16' or 'float32'")

    mlmodel = ct.convert(traced, **convert_kwargs)
    mlmodel.save(output_path)
    print(f"CoreML model saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ARC PyTorch model to CoreML")
    parser.add_argument("--output", type=str, default="arc_model.mlpackage", help="Output CoreML path")
    parser.add_argument("--vocab-size", type=int, default=12)
    parser.add_argument("--seq-len", type=int, default=900)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--n-cycles", type=int, default=6)
    parser.add_argument("--t-cycles", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--puzzle-emb-dim", type=int, default=16)
    parser.add_argument("--num-puzzles", type=int, default=1000)
    parser.add_argument("--no-puzzle-ids", action="store_true", help="Disable puzzle ID embeddings")
    parser.add_argument("--precision", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--weights", type=str, default=None, help="Path to checkpoint (.pt/.pth/.safetensors)")
    parser.add_argument("--test-forward", action="store_true", help="Run a PyTorch forward pass instead of exporting")
    parser.add_argument("--batch", type=int, default=1, help="Fixed batch size for export/tracing (CoreML input shape)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TorchARCConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        n_cycles=args.n_cycles,
        t_cycles=args.t_cycles,
        dropout=args.dropout,
        puzzle_emb_dim=args.puzzle_emb_dim,
        num_puzzles=args.num_puzzles,
        use_puzzle_ids=not args.no_puzzle_ids,
    )

    if args.test_forward:
        torch.manual_seed(args.seed or 0)
        model = TorchARCModel(config)
        load_weights(model, args.weights, config)
        model = maybe_cast_fp16(model, args.precision)
        model.eval()
        token_ids = torch.randint(0, config.vocab_size, (1, config.seq_len), dtype=torch.long)
        puzzle_ids = torch.randint(0, config.num_puzzles, (1,), dtype=torch.long) if config.use_puzzle_ids else None
        with torch.no_grad():
            logits, q_halt = model(token_ids, puzzle_ids)
        print(f"Logits shape: {logits.shape}, Q-halt shape: {q_halt.shape}")
        print("Sample logits (first position):", logits[0, 0, :5])
        return

    export_coreml(
        output_path=args.output,
        config=config,
        precision=args.precision,
        seed=args.seed,
        weights_path=args.weights,
        batch_size=args.batch,
    )


if __name__ == "__main__":
    main()
