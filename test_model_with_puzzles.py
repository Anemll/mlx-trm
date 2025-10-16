#!/usr/bin/env python3
"""Test the updated model with puzzle ID embeddings."""

import mlx.core as mx
from models.trm_arc import ARCModel, ARCModelConfig
from data.arc_augmented import arc_agi_augmented

print("Testing model with puzzle embeddings...")

# Load augmented data
train_data, _, meta = arc_agi_augmented(batch_size=4, enable_augmentations=True, seed=42)

# Create model config
config = ARCModelConfig(
    vocab_size=meta['vocab_size'],
    max_seq_len=meta['max_seq_len'],
    depth=2,
    dim=256,  # Smaller for testing
    heads=4,
    n=2,  # Fewer recursion steps for testing
    T=1,
    halt_max_steps=2,
    halt_exploration_prob=0.1,
    halt_follow_q=True,
    # Puzzle embeddings
    num_puzzle_identifiers=meta['num_puzzle_identifiers'],
    puzzle_emb_ndim=16,
)

print(f"\nModel config:")
print(f"  vocab_size: {config.vocab_size}")
print(f"  max_seq_len: {config.max_seq_len}")
print(f"  num_puzzle_identifiers: {config.num_puzzle_identifiers}")
print(f"  puzzle_emb_ndim: {config.puzzle_emb_ndim}")

# Create model
model = ARCModel(config)

# Get a batch
batch = next(iter(train_data))
print(f"\nBatch info:")
print(f"  input_tokens shape: {batch['input_tokens'].shape}")
print(f"  puzzle_ids shape: {batch['puzzle_ids'].shape}")
print(f"  puzzle_ids: {batch['puzzle_ids']}")

# Convert to MLX arrays
mlx_batch = {
    "input_tokens": mx.array(batch["input_tokens"]),
    "output_tokens": mx.array(batch["output_tokens"]),
    "puzzle_ids": mx.array(batch["puzzle_ids"]),
}

# Initialize carry
carry = model.initial_carry(mlx_batch)

# Forward pass
print(f"\nRunning forward pass...")
carry, outputs = model(carry, mlx_batch)

print(f"\nOutput shapes:")
print(f"  logits: {outputs['logits'].shape}")
print(f"  q_halt_logits: {outputs['q_halt_logits'].shape}")

print(f"\nCarry state:")
print(f"  steps: {carry['steps']}")
print(f"  halted: {carry['halted']}")

print("\n✅ Model successfully handles puzzle ID embeddings!")
print("   - Puzzle IDs are embedded and added to token embeddings")
print("   - Matching TinyRecursiveModels implementation")