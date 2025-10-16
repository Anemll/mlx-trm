"""Visualize augmented ARC examples.

This script shows visual examples of augmentations applied to ARC puzzles.
"""

import numpy as np
import matplotlib.pyplot as plt
from data.arc_augmented import (
    arc_agi_augmented,
    apply_dihedral_transform,
    apply_color_permutation
)

def reshape_grid(grid_flat, max_size=30):
    """Reshape flattened grid and remove padding."""
    grid = grid_flat.reshape(max_size, max_size)

    # Find actual grid size (remove padding=10)
    non_padding_rows = np.where((grid != 10).any(axis=1))[0]
    non_padding_cols = np.where((grid != 10).any(axis=0))[0]

    if len(non_padding_rows) > 0 and len(non_padding_cols) > 0:
        grid = grid[non_padding_rows[0]:non_padding_rows[-1]+1,
                    non_padding_cols[0]:non_padding_cols[-1]+1]

    return grid

def visualize_example(ax, grid, title="", show_colorbar=False):
    """Visualize a single grid."""
    # ARC color palette
    colors = [
        '#000000',  # 0: Black
        '#0074D9',  # 1: Blue
        '#FF4136',  # 2: Red
        '#2ECC40',  # 3: Green
        '#FFDC00',  # 4: Yellow
        '#AAAAAA',  # 5: Grey
        '#F012BE',  # 6: Magenta
        '#FF851B',  # 7: Orange
        '#7FDBFF',  # 8: Sky
        '#870C25',  # 9: Maroon
    ]

    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, which='both', color='white', linewidth=0.5)

    if show_colorbar:
        cbar = plt.colorbar(ax.images[0], ax=ax, ticks=range(10))
        cbar.ax.set_yticklabels([str(i) for i in range(10)])

def show_dihedral_transforms():
    """Show all 8 dihedral transformations of a puzzle."""
    print("Loading dataset...")
    train, _, _ = arc_agi_augmented(
        batch_size=1,
        enable_augmentations=False,
        seed=42
    )

    # Get one example
    batch = next(iter(train))
    input_grid = reshape_grid(batch['input_tokens'][0])
    output_grid = reshape_grid(batch['output_tokens'][0])

    # Apply all 8 dihedral transforms
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Dihedral Transformations (D4 Group)', fontsize=14, fontweight='bold')

    transforms = [
        "Identity (0°)",
        "Rotate 90°",
        "Rotate 180°",
        "Rotate 270°",
        "Flip Horizontal",
        "Flip Vertical",
        "Transpose",
        "Anti-diagonal"
    ]

    for i in range(8):
        # Apply transform to input
        transformed_input = apply_dihedral_transform(input_grid, i)
        # Apply transform to output
        transformed_output = apply_dihedral_transform(output_grid, i)

        # Plot input
        visualize_example(axes[i//2, (i%2)*2], transformed_input, f"Input: {transforms[i]}")
        # Plot output
        visualize_example(axes[i//2, (i%2)*2+1], transformed_output, f"Output: {transforms[i]}")

    plt.tight_layout()
    plt.savefig('dihedral_transforms.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: dihedral_transforms.png")
    plt.show()

def show_color_permutations():
    """Show color permutation examples."""
    print("\nLoading dataset for color permutations...")
    train, _, _ = arc_agi_augmented(
        batch_size=1,
        enable_augmentations=False,
        seed=42
    )

    # Get one example
    batch = next(iter(train))
    input_grid = reshape_grid(batch['input_tokens'][0])
    output_grid = reshape_grid(batch['output_tokens'][0])

    # Generate 6 random color permutations
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Color Permutations (keeping black=0)', fontsize=14, fontweight='bold')

    np.random.seed(42)

    # Original
    visualize_example(axes[0, 0], input_grid, "Original Input")
    visualize_example(axes[0, 1], output_grid, "Original Output")

    # 3 random permutations
    for i in range(3):
        # Generate random permutation
        colors = np.arange(1, 10)
        np.random.shuffle(colors)
        perm = np.zeros(10, dtype=int)
        perm[0] = 0  # Black stays black
        perm[1:] = colors

        permuted_input = apply_color_permutation(input_grid, perm)
        permuted_output = apply_color_permutation(output_grid, perm)

        visualize_example(axes[i+1, 0], permuted_input, f"Permutation {i+1} Input")
        visualize_example(axes[i+1, 1], permuted_output, f"Permutation {i+1} Output")

    # Hide unused axes
    for i in range(4):
        for j in range(2, 4):
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('color_permutations.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: color_permutations.png")
    plt.show()

def show_augmentation_stats():
    """Show statistics about augmented dataset."""
    print("\n" + "=" * 80)
    print("AUGMENTATION STATISTICS")
    print("=" * 80)

    # Without augmentation
    print("\n1. Without augmentation:")
    train_none, _, meta_none = arc_agi_augmented(
        batch_size=32,
        enable_augmentations=False,
        seed=42
    )
    print(f"   Total examples: {meta_none['n_train_examples']}")
    print(f"   Unique puzzles: {meta_none['num_puzzle_identifiers']}")

    # With augmentation (small)
    print("\n2. With augmentation (max_aug=30):")
    train_small, _, meta_small = arc_agi_augmented(
        batch_size=32,
        enable_augmentations=True,
        seed=42,
        max_augmentations_per_puzzle=30
    )
    print(f"   Total examples: {meta_small['n_train_examples']}")
    print(f"   Multiplier: {meta_small['n_train_examples'] / meta_none['n_train_examples']:.1f}x")

    # With augmentation (medium)
    print("\n3. With augmentation (max_aug=300, default):")
    train_med, _, meta_med = arc_agi_augmented(
        batch_size=32,
        enable_augmentations=True,
        seed=42,
        max_augmentations_per_puzzle=300
    )
    print(f"   Total examples: {meta_med['n_train_examples']}")
    print(f"   Multiplier: {meta_med['n_train_examples'] / meta_none['n_train_examples']:.1f}x")

    # With augmentation (large, like paper)
    print("\n4. With augmentation (max_aug=1000, paper):")
    train_large, _, meta_large = arc_agi_augmented(
        batch_size=32,
        enable_augmentations=True,
        seed=42,
        max_augmentations_per_puzzle=1000
    )
    print(f"   Total examples: {meta_large['n_train_examples']}")
    print(f"   Multiplier: {meta_large['n_train_examples'] / meta_none['n_train_examples']:.1f}x")

    print("\n" + "=" * 80)
    print("RECOMMENDATION:")
    print("  - Quick prototyping: max_aug=30 (~30x data)")
    print("  - Standard training: max_aug=300 (~95x data, good balance)")
    print("  - Paper matching: max_aug=1000 (~260x data, slower)")
    print("=" * 80)

def compare_augmented_examples():
    """Show a comparison of original vs augmented examples."""
    print("\n" + "=" * 80)
    print("COMPARING ORIGINAL vs AUGMENTED")
    print("=" * 80)

    train, _, _ = arc_agi_augmented(
        batch_size=16,
        enable_augmentations=True,
        seed=42,
        max_augmentations_per_puzzle=20
    )

    # Get first batch
    batch = next(iter(train))

    # Find the first puzzle that appears multiple times
    puzzle_ids = batch['puzzle_ids']
    unique_ids, counts = np.unique(puzzle_ids, return_counts=True)

    for puzzle_id, count in zip(unique_ids, counts):
        if count >= 4:  # Need at least 4 examples
            print(f"\nFound puzzle {puzzle_id} with {count} variations")

            # Get indices for this puzzle
            indices = np.where(puzzle_ids == puzzle_id)[0][:4]

            # Plot
            fig, axes = plt.subplots(4, 2, figsize=(8, 12))
            fig.suptitle(f'Puzzle {puzzle_id}: Original + Augmentations', fontsize=14, fontweight='bold')

            for i, idx in enumerate(indices):
                input_grid = reshape_grid(batch['input_tokens'][idx])
                output_grid = reshape_grid(batch['output_tokens'][idx])

                label = "Original" if i == 0 else f"Augmented {i}"
                visualize_example(axes[i, 0], input_grid, f"{label} Input")
                visualize_example(axes[i, 1], output_grid, f"{label} Output")

            plt.tight_layout()
            plt.savefig(f'puzzle_{puzzle_id}_variations.png', dpi=150, bbox_inches='tight')
            print(f"✅ Saved: puzzle_{puzzle_id}_variations.png")
            plt.show()
            break

def main():
    """Run all visualizations."""
    print("=" * 80)
    print("ARC AUGMENTATION VISUALIZER")
    print("=" * 80)

    try:
        # Show statistics
        show_augmentation_stats()

        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS...")
        print("=" * 80)

        # Generate visualizations
        print("\n1. Dihedral transformations...")
        show_dihedral_transforms()

        print("\n2. Color permutations...")
        show_color_permutations()

        print("\n3. Comparing variations...")
        compare_augmented_examples()

        print("\n" + "=" * 80)
        print("✅ ALL VISUALIZATIONS GENERATED!")
        print("=" * 80)
        print("\nFiles created:")
        print("  - dihedral_transforms.png")
        print("  - color_permutations.png")
        print("  - puzzle_*_variations.png")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
