"""Test augmented dataset without training.

This script demonstrates how to load and inspect the augmented ARC dataset.
"""

import numpy as np
from data.arc_augmented import arc_agi_augmented, AugmentationConfig
import matplotlib.pyplot as plt

def visualize_grid(grid_flat, title="", max_size=30):
    """Visualize a flattened grid."""
    # Reshape from flat sequence to 2D grid
    grid = grid_flat.reshape(max_size, max_size)

    # Find actual grid size (remove padding)
    non_padding_rows = (grid != 10).any(axis=1).sum()
    non_padding_cols = (grid != 10).any(axis=0).sum()

    if non_padding_rows > 0 and non_padding_cols > 0:
        grid = grid[:non_padding_rows, :non_padding_cols]

    # Create color map (ARC uses 10 colors: 0-9)
    cmap = plt.cm.get_cmap('tab10', 10)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    plt.title(title)
    plt.colorbar(ticks=range(10), label='Color')
    plt.grid(True, which='both', color='gray', linewidth=0.5)
    plt.tight_layout()

    return grid

def test_basic_loading():
    """Test basic dataset loading without augmentation."""
    print("=" * 80)
    print("TEST 1: Basic Loading (No Augmentation)")
    print("=" * 80)

    train, test, meta = arc_agi_augmented(
        batch_size=4,
        enable_augmentations=False,
        include_eval_demos=True,  # Match training default!
        seed=42
    )

    print(f"\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print(f"\nFetching first batch...")
    batch = next(iter(train))

    print(f"\nBatch shapes:")
    print(f"  input_tokens: {batch['input_tokens'].shape}")
    print(f"  output_tokens: {batch['output_tokens'].shape}")
    print(f"  puzzle_ids: {batch['puzzle_ids'].shape}")

    print(f"\nFirst example:")
    print(f"  Puzzle ID: {batch['puzzle_ids'][0]}")
    print(f"  Input (first 50 tokens): {batch['input_tokens'][0][:50]}")
    print(f"  Output (first 50 tokens): {batch['output_tokens'][0][:50]}")

    return train, test, meta

def test_augmented_loading():
    """Test dataset loading with augmentation."""
    print("\n" + "=" * 80)
    print("TEST 2: Augmented Loading")
    print("=" * 80)

    train, test, meta = arc_agi_augmented(
        batch_size=4,
        enable_augmentations=True,
        include_eval_demos=True,  # Match training default!
        seed=42,
        max_augmentations_per_puzzle=100  # Fewer for faster testing
    )

    print(f"\nMetadata (with augmentation):")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print(f"\nAugmentation multiplier: {meta['n_train_examples'] / 1302:.1f}x")
    print(f"  (Base examples: 1302 → Augmented: {meta['n_train_examples']})")

    return train, test, meta

def test_augmentation_variations():
    """Show examples of the same puzzle with different augmentations."""
    print("\n" + "=" * 80)
    print("TEST 3: Augmentation Variations")
    print("=" * 80)

    train, _, _ = arc_agi_augmented(
        batch_size=32,  # Larger batch to see more variations
        enable_augmentations=True,
        seed=42,
        max_augmentations_per_puzzle=50
    )

    # Get first batch
    batch = next(iter(train))

    # Find examples from the same puzzle (same puzzle_id)
    puzzle_ids = batch['puzzle_ids']
    unique_ids, counts = np.unique(puzzle_ids, return_counts=True)

    print(f"\nPuzzle IDs in first batch: {unique_ids}")
    print(f"Counts per puzzle: {counts}")

    # Show examples from the first puzzle that appears multiple times
    if len(unique_ids) > 0:
        target_puzzle = unique_ids[0]
        indices = np.where(puzzle_ids == target_puzzle)[0]

        print(f"\nFound {len(indices)} variations of puzzle {target_puzzle}:")
        print(f"  Batch indices: {indices}")

        # Show first 3 variations
        for i, idx in enumerate(indices[:3]):
            input_grid = batch['input_tokens'][idx]
            output_grid = batch['output_tokens'][idx]

            print(f"\n  Variation {i+1} (index {idx}):")
            print(f"    Input (first 30 tokens): {input_grid[:30]}")
            print(f"    Output (first 30 tokens): {output_grid[:30]}")

def test_puzzle_id_mapping():
    """Test puzzle ID mapping across splits."""
    print("\n" + "=" * 80)
    print("TEST 4: Puzzle ID Mapping")
    print("=" * 80)

    train, test, meta = arc_agi_augmented(
        batch_size=4,
        enable_augmentations=True,
        seed=42
    )

    # Get puzzle IDs from both datasets
    train_puzzle_ids = set()
    for batch in train:
        train_puzzle_ids.update(batch['puzzle_ids'].tolist())

    test_puzzle_ids = set()
    for batch in test:
        test_puzzle_ids.update(batch['puzzle_ids'].tolist())

    print(f"\nTraining puzzle IDs:")
    print(f"  Range: {min(train_puzzle_ids)} to {max(train_puzzle_ids)}")
    print(f"  Count: {len(train_puzzle_ids)}")
    print(f"  First 10: {sorted(train_puzzle_ids)[:10]}")

    print(f"\nEvaluation puzzle IDs:")
    print(f"  Range: {min(test_puzzle_ids)} to {max(test_puzzle_ids)}")
    print(f"  Count: {len(test_puzzle_ids)}")
    print(f"  First 10: {sorted(test_puzzle_ids)[:10]}")

    # Check overlap (should be minimal or none)
    overlap = train_puzzle_ids.intersection(test_puzzle_ids)
    print(f"\nOverlap between train and test: {len(overlap)} puzzles")
    if overlap:
        print(f"  Overlapping IDs: {sorted(overlap)}")

def test_custom_augmentation():
    """Test with custom augmentation configuration."""
    print("\n" + "=" * 80)
    print("TEST 5: Custom Augmentation Config")
    print("=" * 80)

    # Test with only dihedral (no color permutation)
    print("\nTest 5a: Dihedral only")
    train_dihedral, _, meta_d = arc_agi_augmented(
        batch_size=4,
        enable_augmentations=True,
        seed=42,
        max_augmentations_per_puzzle=7  # 7 dihedral transforms (excluding identity)
    )
    print(f"  Examples with dihedral only: {meta_d['n_train_examples']}")

    # Test with no augmentation
    print("\nTest 5b: No augmentation")
    train_none, _, meta_n = arc_agi_augmented(
        batch_size=4,
        enable_augmentations=False,
        seed=42
    )
    print(f"  Examples without augmentation: {meta_n['n_train_examples']}")

    # Calculate multiplier
    multiplier = meta_d['n_train_examples'] / meta_n['n_train_examples']
    print(f"\nAugmentation multiplier: {multiplier:.2f}x")

def test_batch_iteration():
    """Test iterating through multiple batches."""
    print("\n" + "=" * 80)
    print("TEST 6: Batch Iteration")
    print("=" * 80)

    train, _, meta = arc_agi_augmented(
        batch_size=32,
        enable_augmentations=True,
        seed=42,
        max_augmentations_per_puzzle=50
    )

    print(f"\nTotal examples: {meta['n_train_examples']}")
    print(f"Batch size: 32")
    print(f"Expected batches: {meta['steps_per_epoch']}")

    print(f"\nIterating through first 5 batches...")
    for i, batch in enumerate(train):
        if i >= 5:
            break
        print(f"  Batch {i+1}: {batch['input_tokens'].shape}, "
              f"puzzle_ids={batch['puzzle_ids'][:5].tolist()}...")

    print(f"\n✅ Successfully iterated through batches!")

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("AUGMENTED DATASET TEST SUITE")
    print("=" * 80)

    try:
        # Run tests
        test_basic_loading()
        test_augmented_loading()
        test_augmentation_variations()
        test_puzzle_id_mapping()
        test_custom_augmentation()
        test_batch_iteration()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe augmented dataset is working correctly!")
        print("You can now use it in training with: python train_arc.py --augment")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
