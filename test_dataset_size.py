"""Quick test to verify dataset sizes match training script.

This script shows the difference between including/excluding eval demos.
"""

from data.arc_augmented import arc_agi_augmented

print("=" * 80)
print("DATASET SIZE COMPARISON")
print("=" * 80)

print("\n1. WITHOUT eval demos (include_eval_demos=False):")
print("-" * 80)
train, test, meta = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=True,
    include_eval_demos=False,  # Only use training split demos
    max_augmentations_per_puzzle=300,
    seed=0
)
print(f"Training examples: {meta['n_train_examples']}")
print(f"Training tasks: {meta['n_train_tasks']}")
print(f"Eval examples: {meta['n_test_examples']}")
print(f"Eval tasks: {meta['n_test_tasks']}")

print("\n2. WITH eval demos (include_eval_demos=True) ← TRAINING DEFAULT:")
print("-" * 80)
train2, test2, meta2 = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=True,
    include_eval_demos=True,  # Also use evaluation split demos for training
    max_augmentations_per_puzzle=300,
    seed=0
)
print(f"Training examples: {meta2['n_train_examples']}")
print(f"Training tasks: {meta2['n_train_tasks']}")
print(f"Eval examples: {meta2['n_test_examples']}")
print(f"Eval tasks: {meta2['n_test_tasks']}")

print("\n" + "=" * 80)
print("EXPLANATION")
print("=" * 80)
print()
print("The training script uses include_eval_demos=True by default.")
print("This means it includes BOTH:")
print("  1. Training split demo pairs (400 tasks)")
print("  2. Evaluation split demo pairs (400 tasks)")
print()
print("This gives:")
print(f"  - Base: {meta2['n_train_tasks']} tasks (400 train + 400 eval)")
print(f"  - With augmentation (×~95): ~{meta2['n_train_examples']} examples")
print()
print("The evaluation still uses only TEST pairs (not demos), so there's")
print("no data leakage - we never train on the actual test outputs!")
print()
print("=" * 80)
print("MATCH WITH YOUR TRAINING OUTPUT")
print("=" * 80)
print()
print("Your training output showed:")
print("  Training examples: 23520 (from 800 tasks)")
print()
print("Our output shows:")
print(f"  Training examples: {meta2['n_train_examples']} (from {meta2['n_train_tasks']} tasks)")
print()

if abs(meta2['n_train_examples'] - 23520) < 1000:
    print("✅ MATCHES! (within expected variance)")
else:
    print(f"⚠️  Difference: {abs(meta2['n_train_examples'] - 23520)} examples")
    print("   This could be due to:")
    print("   - Different random seed")
    print("   - Different max_augmentations_per_puzzle")
    print("   - Different data directory")
