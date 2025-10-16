"""Compare PyTorch export model vs MLX model on a single batch."""

import argparse
import numpy as np
import torch
import mlx.core as mx
from models.trm_arc import ARCModel as MLXARCModel
from models.trm_arc import ARCModelConfig as MLXARCConfig

from export_arc_coreml import TorchARCModel, TorchARCConfig, load_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    seq_len = 900
    batch = 1

    token_ids = torch.randint(0, 12, (batch, seq_len), dtype=torch.int32)
    puzzle_ids = torch.randint(0, 1000, (batch,), dtype=torch.int32)

    torch_model = TorchARCModel(TorchARCConfig(dim=args.dim, seq_len=seq_len, num_puzzles=1000))
    load_weights(torch_model, args.checkpoint, TorchARCConfig(dim=args.dim, seq_len=seq_len, num_puzzles=1000))
    torch_model.eval()

    with torch.no_grad():
        torch_logits, torch_q = torch_model(token_ids, puzzle_ids)

    mx.set_default_device(mx.cpu)
    mlx_model = MLXARCModel(MLXARCConfig(vocab_size=12, max_seq_len=seq_len, depth=2, dim=args.dim, heads=8,
                                         n=6, T=3, halt_max_steps=16, halt_exploration_prob=0.0,
                                         num_puzzle_identifiers=1000, puzzle_emb_ndim=16, use_sparse_embeddings=True))
    mlx_model.load_weights(args.checkpoint)
    mlx_model.eval()
    mlx_model._y_init = mx.zeros_like(mlx_model._y_init)
    mlx_model._z_init = mx.zeros_like(mlx_model._z_init)

    mx_tokens = mx.array(token_ids.numpy())
    mx_puzzles = mx.array(puzzle_ids.numpy())

    carry = mlx_model.initial_carry({"input_tokens": mx_tokens, "output_tokens": mx.zeros_like(mx_tokens)})
    carry, outputs = mlx_model(carry, {"input_tokens": mx_tokens, "output_tokens": mx.zeros_like(mx_tokens), "puzzle_ids": mx_puzzles})

    mlx_logits = np.array(outputs["logits"])

    torch_np = torch_logits.numpy()
    diff = np.abs(torch_np - mlx_logits)
    print('torch logits first token:', torch_np[0, 0, :5])
    print('mlx logits first token:', mlx_logits[0, 0, :5])
    print('max abs diff:', diff.max())


if __name__ == '__main__':
    main()
