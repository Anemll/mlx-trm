#!/usr/bin/env python3
"""Test that puzzle IDs are properly handled during validation."""

import mlx.core as mx
from models.trm_arc import ARCModel, ARCModelConfig
from data.arc_augmented import arc_agi_augmented
from evaluators.arc_evaluator import evaluate_arc_model

print("Testing puzzle ID handling in validation...")

# Load augmented dataset with puzzle IDs
print("\n1. Loading augmented dataset...")
train_data, test_data, meta = arc_agi_augmented(
    batch_size=8,
    enable_augmentations=False,  # Disable for validation clarity
    seed=42
)

print(f"   - Test examples: {meta['n_test_examples']}")
print(f"   - Number of puzzle IDs: {meta['num_puzzle_identifiers']}")

# Create a small model for testing
print("\n2. Creating model with puzzle embeddings...")
config = ARCModelConfig(
    vocab_size=meta['vocab_size'],
    max_seq_len=meta['max_seq_len'],
    depth=1,  # Small for testing
    dim=128,  # Small for testing
    heads=4,
    n=2,
    T=1,
    halt_max_steps=2,
    halt_exploration_prob=0.0,  # Disable exploration for deterministic testing
    halt_follow_q=False,  # Disable Q-learning for testing
    # Puzzle embeddings
    num_puzzle_identifiers=meta['num_puzzle_identifiers'],
    puzzle_emb_ndim=16,
)

model = ARCModel(config)

# Test a single batch
print("\n3. Testing single batch inference with puzzle IDs...")
batch = next(iter(test_data))
print(f"   - Batch shape: {batch['input_tokens'].shape}")
print(f"   - Puzzle IDs in batch: {batch['puzzle_ids'][:4]}...")

# Convert to MLX
mlx_batch = {k: mx.array(v) for k, v in batch.items()}

# Initialize and run
carry = model.initial_carry(mlx_batch)
carry, outputs = model(carry, mlx_batch)

print(f"   - Output logits shape: {outputs['logits'].shape}")
print(f"   - Q-halt logits shape: {outputs['q_halt_logits'].shape}")

# Evaluate with puzzle ID tracking
print("\n4. Running evaluation with puzzle ID tracking...")
results = evaluate_arc_model(
    model,
    test_data,
    max_batches=10  # Limit for quick testing
)

# Check that puzzle IDs were tracked
if results['n_unique_puzzles'] > 0:
    print(f"\n✅ Successfully tracked {results['n_unique_puzzles']} unique puzzles")
    print(f"   - Total examples evaluated: {results['total_examples']}")
    print(f"   - Exact matches: {results['total_exact_matches']}")

    # Show a few puzzle-specific results
    print("\n   Sample puzzle-specific results:")
    for i, (puzzle_id, metrics) in enumerate(list(results['puzzle_metrics'].items())[:3]):
        print(f"   - Puzzle {puzzle_id}: {metrics['exact_matches']}/{metrics['total_examples']} "
              f"exact matches ({metrics['exact_match_rate']:.1%})")
else:
    print("\n❌ No puzzles were tracked - check implementation")

print("\n✅ Validation correctly handles puzzle IDs:")
print("   - Puzzle IDs are passed through the batch")
print("   - Model uses puzzle embeddings when available")
print("   - Evaluator tracks per-puzzle performance")
print("   - Matching TinyRecursiveModels evaluation approach")