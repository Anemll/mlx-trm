"""Token-based Tiny Recursive Model for ARC-AGI.

This implementation follows the original TinyRecursiveModels paper for ARC tasks.
Uses token embeddings for sequence-to-sequence modeling, not patch embeddings.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import math

import mlx.core as mx
import mlx.nn as nn

from models.abstract import Base
from models.sparse_embedding import SparseEmbedding


class RMSNorm(nn.Module):
    """Parameter-free RMSNorm matching CUDA implementation.

    No learnable scale/bias - only normalizes with fixed gain=1.
    Matches the CUDA implementation where only eps is configurable.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Compute RMS: sqrt(mean(x²) + eps)
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        rms = mx.sqrt(ms + self.eps)
        # Normalize (no learnable scale)
        return x / rms


@dataclass
class ARCModelConfig:
    """Configuration for ARC token-based model.

    Default values match the original TinyRecursiveModels paper for ARC-AGI-1:
    - n (L_cycles): 6 latent recursion steps
    - T (H_cycles): 3 deep recursion steps
    - depth (L_layers): 2 transformer blocks
    - dim (hidden_size): 512
    - heads: 8
    - halt_max_steps: 16
    - halt_exploration_prob: 0.1
    - Uses RoPE (Rotary Position Embedding) for attention
    - puzzle_emb_ndim: 16 (dimension for puzzle embeddings, from GitHub reference)
    """
    vocab_size: int  # 12 for ARC (0-9 colors + padding (10) + EOS (11))
    max_seq_len: int  # Maximum sequence length (e.g., 900 for 30x30)
    depth: int  # Number of transformer blocks (L_layers in paper)
    dim: int  # Model dimension (hidden_size in paper: 512)
    heads: int  # Number of attention heads (paper: 8)
    n: int = 6  # latent recursion steps (L_cycles in paper)
    T: int = 3  # deep recursion steps (H_cycles in paper)
    ff_mult: int = 3  # feedforward expansion multiplier
    halt_max_steps: int = 16  # maximum adaptive computation steps (paper: 16)
    halt_exploration_prob: float = 0.1  # exploration probability (paper: 0.1)
    halt_follow_q: bool = True  # follow Q-learning halting
    # Puzzle identifier embeddings (matching TinyRecursiveModels)
    num_puzzle_identifiers: int = 1000  # Max number of unique puzzle IDs
    puzzle_emb_ndim: int = 16  # Dimension for puzzle embeddings (from GitHub)
    use_sparse_embeddings: bool = True  # Use sparse embeddings for efficiency
    use_fast_rope: bool = True
    use_mlx_ff: bool = True
    # Training semantics: match original TRM more closely
    # Only keep gradients for the LAST latent update within the final deep step
    grad_last_latent_only: bool = True


class SwiGLU(nn.Module):
    """SwiGLU activation function."""
    def __init__(self, dim, mlp_dim, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(dim, mlp_dim * 2, bias=False)
        self.w2 = nn.Linear(mlp_dim, dim, bias=False)

    def __call__(self, x) -> mx.array:
        gate, up = mx.split(self.w1(x), 2, axis=-1)
        return self.w2(self.dropout(nn.silu(gate) * up))


class Attention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, dim: int, heads: int, use_fast_rope: bool = False, rope_base: float = 10000.0):
        super().__init__()
        self.h = heads
        self.d = dim // heads
        self.scale = self.d**-0.5
        self.use_fast_rope = use_fast_rope
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

        if self.use_fast_rope:
            half = self.d // 2
            if half == 0:
                self._rope_freqs = None
            else:
                # Precompute negative frequencies (odd-even trick per mlx-lm nanochat discussion)
                log_base = math.log(self.rope_base)
                freqs = -mx.exp(
                    mx.arange(0.0, float(half), dtype=mx.float32)
                    * (log_base / float(half))
                )
                self._rope_freqs = freqs
        else:
            self.rope = nn.RoPE(self.d)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, _ = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q, k, v = map(
            lambda x: x.reshape(b, n, self.h, -1).transpose(0, 2, 1, 3), (q, k, v)
        )

        if self.use_fast_rope and getattr(self, "_rope_freqs", None) is not None:
            q = mx.fast.rope(
                q,
                dims=self.d,
                traditional=False,
                base=None,
                freqs=self._rope_freqs,
                scale=1.0,
                offset=0,
            )
            k = mx.fast.rope(
                k,
                dims=self.d,
                traditional=False,
                base=None,
                freqs=self._rope_freqs,
                scale=1.0,
                offset=0,
            )
        else:
            q, k = self.rope(q), self.rope(k)

        x = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out(x.transpose(0, 2, 1, 3).reshape(b, n, -1))


class Block(nn.Module):
    """Transformer block with attention and feedforward."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 3, use_fast_rope: bool = False, use_mlx_ff: bool = False):
        super().__init__()
        '''
        # orignal CUDA CODE
        self.n1 = RMSNorm()  # Parameter-free, matching CUDA
        self.attn = Attention(dim, heads)
        self.n2 = RMSNorm()  # Parameter-free, matching CUDA
        self.ff = SwiGLU(dim, ff_mult * dim)  # configurable expansion
        '''
        # Use MLX's RMSNorm (eps matches reference implementation)
        self.n1 = nn.RMSNorm(dim, eps=1e-5)
        self.attn = Attention(dim, heads, use_fast_rope=use_fast_rope)
        self.n2 = nn.RMSNorm(dim, eps=1e-5)
        self.use_mlx_ff = use_mlx_ff
        self.ff_mult = ff_mult
        self.ff = SwiGLU(dim, ff_mult * dim)
        self.ff_linear1 = nn.Linear(dim, ff_mult * dim, bias=False)
        self.ff_linear2 = nn.Linear(ff_mult * dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        attn_out = self.attn(x)
        x = self.n1(x + attn_out)

        if self.use_mlx_ff:
            y = self.ff_linear1(x)
            y = nn.silu(y)
            y = self.ff_linear2(y)
        else:
            y = self.ff(x)

        x = self.n2(x + y)
        return x


class TokenEmbedding(nn.Module):
    """Token embedding layer for sequences with puzzle identifiers.

    Matches TinyRecursiveModels implementation:
    - Token embeddings for vocabulary
    - Puzzle embeddings for task identifiers (SPARSE)
    - Q-token for halting decision
    """
    def __init__(self, vocab_size: int, dim: int, max_seq_len: int,
                 num_puzzle_identifiers: int = 1000, puzzle_emb_ndim: int = 16,
                 use_sparse_embeddings: bool = True):
        super().__init__()
        self.dim = dim
        self.use_sparse_embeddings = use_sparse_embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)

        # Puzzle identifier embeddings (matching TinyRecursiveModels)
        # GitHub ref uses CastedSparseEmbedding with init_std=0
        if use_sparse_embeddings:
            # Use sparse embedding for efficiency (only updates used embeddings)
            self.puzzle_emb = SparseEmbedding(num_puzzle_identifiers, puzzle_emb_ndim, init_std=0.0)
        else:
            # Standard embedding (updates all embeddings)
            self.puzzle_emb = nn.Embedding(num_puzzle_identifiers, puzzle_emb_ndim)
            # Initialize puzzle embeddings with small values (matching init_std=0 behavior)
            self.puzzle_emb.weight = mx.zeros((num_puzzle_identifiers, puzzle_emb_ndim))

        # Projection to combine puzzle embedding with token embedding
        self.puzzle_proj = nn.Linear(puzzle_emb_ndim, dim, bias=False)

        # Scale embeddings like in the original implementation
        self.scale = dim ** 0.5

        # Q-token for halting decision (prepended to sequence)
        self.q_token = 0.02 * mx.random.normal((1, 1, dim))

    def __call__(self, x: mx.array, puzzle_ids: mx.array = None) -> mx.array:
        """
        Args:
            x: Token indices of shape (batch, seq_len)
            puzzle_ids: Puzzle identifiers of shape (batch,)
        Returns:
            Embeddings of shape (batch, seq_len+1, dim) with q_token prepended
        """
        b = x.shape[0]
        seq_len = x.shape[1]

        # Embed tokens and scale
        token_embeddings = self.token_emb(x) * self.scale

        # Add puzzle embeddings if provided (matching TinyRecursiveModels)
        if puzzle_ids is not None:
            # Get puzzle embeddings for batch
            puzzle_emb = self.puzzle_emb(puzzle_ids)  # (batch, puzzle_emb_ndim)
            # Project to model dimension
            puzzle_emb = self.puzzle_proj(puzzle_emb)  # (batch, dim)
            # Broadcast and add to all positions
            puzzle_emb = puzzle_emb.reshape(b, 1, self.dim)
            puzzle_emb = mx.broadcast_to(puzzle_emb, (b, seq_len, self.dim))
            token_embeddings = token_embeddings + puzzle_emb

        # Prepend Q-token for halting
        x = mx.concat([mx.repeat(self.q_token, b, axis=0), token_embeddings], axis=1)
        return x


class OutputHead(nn.Module):
    """Output head for sequence prediction."""
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.out = nn.Linear(dim, vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Hidden states of shape (batch, seq_len, dim)
        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        return self.out(x)


class QHead(nn.Module):
    """Q-learning head for adaptive halting (2-class, matching CUDA)."""
    def __init__(self, dim: int):
        super().__init__()
        self.out = nn.Linear(dim, 2)  # 2 classes: continue (0) and halt (1)
        # Initialize with bias towards not halting
        self.out.weight[:] = 0
        self.out.bias[:] = mx.array([5.0, -5.0])  # [continue_bias, halt_bias]

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: Q-token hidden state of shape (batch, 1, dim)
        Returns:
            Halt logits of shape (batch, 2) - [continue_logit, halt_logit]
        """
        return self.out(x).reshape(-1, 2)  # (batch, 2)


class ARCModel(Base):
    """Token-based Tiny Recursive Model for ARC tasks.

    This model treats ARC as a sequence-to-sequence problem:
    - Input: Flattened grid tokens (e.g., 30x30 = 900 tokens)
    - Output: Predicted output grid tokens

    Uses recursive reasoning with adaptive computation time (ACT).
    """

    def __init__(self, config: ARCModelConfig):
        super().__init__()
        self.config = config

        self.embed = TokenEmbedding(
            config.vocab_size, config.dim, config.max_seq_len,
            config.num_puzzle_identifiers, config.puzzle_emb_ndim,
            config.use_sparse_embeddings
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    config.dim,
                    config.heads,
                    config.ff_mult,
                    use_fast_rope=config.use_fast_rope,
                    use_mlx_ff=config.use_mlx_ff,
                )
                for _ in range(config.depth)
            ]
        )
        if config.use_mlx_ff:
            for block in self.blocks.layers:
                block.use_mlx_ff = True
        self.out_head = OutputHead(config.dim, config.vocab_size)
        self.q_head = QHead(config.dim)

        # Initialize latent carry states
        self._y_init = mx.random.truncated_normal(-2, 2, (self.config.dim,))
        self._z_init = mx.random.truncated_normal(-2, 2, (self.config.dim,))

    def initial_carry(self, batch: Dict[str, mx.array]):
        """Initialize the recursive carry state."""
        b = batch["input_tokens"].shape[0]
        seq_len = batch["input_tokens"].shape[1]

        return dict(
            inner_carry=dict(
                y=mx.zeros((b, seq_len + 1, self.config.dim)),  # +1 for q_token
                z=mx.zeros((b, seq_len + 1, self.config.dim)),
            ),
            steps=mx.zeros((b,), dtype=mx.int32),
            halted=mx.ones((b,), dtype=mx.bool_),
            current_data={k: mx.zeros_like(v) for k, v in batch.items()},
        )

    def reset_inner_carry(self, halted: mx.array, carry: dict):
        """Reset carry state for halted sequences."""
        mask = halted.reshape(-1, 1, 1)
        return dict(
            y=mx.where(mask, self._y_init, carry["y"]),
            z=mx.where(mask, self._z_init, carry["z"]),
        )

    def latent_recursion(self, carry: dict, x: mx.array) -> dict:
        """Latent recursion: n steps of processing."""
        y, z = carry["y"], carry["z"]
        if self.config.grad_last_latent_only and self.training:
            # Perform n-1 latent updates with gradients detached between steps
            # so they don't contribute to backprop graph
            for _ in range(max(0, self.config.n - 1)):
                z = self.blocks(z + y + x)
                # Detach state before next latent step to avoid storing graph
                y = mx.stop_gradient(y)
                z = mx.stop_gradient(z)

            # Final latent update with gradients kept
            z = self.blocks(z + y + x)
            y = self.blocks(y + z)
        else:
            for _ in range(self.config.n):
                z = self.blocks(z + y + x)
            y = self.blocks(y + z)
        carry["y"], carry["z"] = y, z
        return carry

    def deep_recursion(self, carry: dict, batch: Dict[str, mx.array]):
        """Deep recursion: T steps with gradient stopping."""
        # Get puzzle IDs if available (matching TinyRecursiveModels)
        puzzle_ids = batch.get("puzzle_ids", None)
        x = self.embed(batch["input_tokens"], puzzle_ids)  # (batch, seq_len+1, dim)

        # T-1 steps without gradients (as intended by TRM)
        # Only the final deep step should retain gradients.
        for _ in range(max(0, self.config.T - 1)):
            carry = self.latent_recursion(carry, x)
            # Block gradients between deep steps to avoid storing activations
            carry["y"] = mx.stop_gradient(carry["y"])
            carry["z"] = mx.stop_gradient(carry["z"])

        # Final step with gradients
        carry = self.latent_recursion(carry, x)

        # Output logits from all positions except q_token
        outputs = {
            "logits": self.out_head(carry["y"][:, 1:]),  # (batch, seq_len, vocab)
            "q_halt_logits": self.q_head(carry["y"][:, 0:1]),  # (batch,)
        }

        carry["y"] = mx.stop_gradient(carry["y"])
        carry["z"] = mx.stop_gradient(carry["z"])

        return carry, outputs

    def __call__(self, carry: dict, batch: Dict[str, mx.array]):
        """Forward pass with adaptive computation time."""
        # Reset carry for halted sequences
        new_inner_carry = self.reset_inner_carry(carry["halted"], carry["inner_carry"])
        new_steps = mx.where(carry["halted"], 0, carry["steps"])

        # Update current data for halted sequences
        new_current_data = {
            k: mx.where(
                carry["halted"].reshape((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                mx.array(v),
            )
            for k, v in carry["current_data"].items()
        }

        # Deep recursion
        new_inner_carry, outputs = self.deep_recursion(
            new_inner_carry, new_current_data
        )

        # Increment steps
        new_steps = new_steps + 1
        is_last_step = new_steps >= self.config.halt_max_steps
        halted = is_last_step

        # Adaptive Computational Time (ACT) with Q-learning
        if (self.config.halt_max_steps > 1) and (
            self.training or self.config.halt_follow_q
        ):
            # outputs["q_halt_logits"] is (batch, 2): [continue_logit, halt_logit]
            # Halt if halt_logit > continue_logit
            halted = halted | (outputs["q_halt_logits"][:, 1] > outputs["q_halt_logits"][:, 0])

        # Exploration during training
        if (
            self.training
            and (self.config.halt_max_steps > 1)
            and (self.config.halt_exploration_prob > 0)
        ):
            # Shape for random uniform should match batch size
            batch_size = outputs["q_halt_logits"].shape[0]
            min_halt_steps = (
                mx.random.uniform(shape=(batch_size,))
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
    # Test the model
    mx.random.seed(0)

    batch_size = 4
    seq_len = 900  # 30x30 grid flattened
    vocab_size = 12  # 0-9 colors + padding + EOS

    config = ARCModelConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        depth=2,
        dim=512,
        heads=8,
    )

    model = ARCModel(config)
    model.summary()

    # Create dummy batch
    batch = {
        "input_tokens": mx.random.randint(0, vocab_size, (batch_size, seq_len)),
        "output_tokens": mx.random.randint(0, vocab_size, (batch_size, seq_len)),
    }

    carry = model.initial_carry(batch)
    carry, outputs = model(carry, batch)

    print(f"\nOutput shapes:")
    print(f"  logits: {outputs['logits'].shape}")  # (batch, seq_len, vocab)
    print(f"  q_halt_logits: {outputs['q_halt_logits'].shape}")  # (batch,)
