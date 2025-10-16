"""Create sample ARC data for testing.

This creates minimal synthetic ARC-like data for testing the pipeline.
For real training, you should use the official ARC-AGI-1 dataset from:
https://github.com/fchollet/ARC-AGI
"""

import json
from pathlib import Path

import numpy as np


def create_sample_arc_data(output_dir: str = "data/arc-sample", n_train: int = 100, n_test: int = 20):
    """Create sample ARC data files.

    Args:
        output_dir: Directory to save the sample data
        n_train: Number of training puzzles
        n_test: Number of test puzzles
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ARC uses 30x30 grids with values 0-9
    seq_len = 900  # 30 * 30
    vocab_size = 12  # 0-9 colors + padding (10) + EOS (11)

    print(f"Creating sample ARC data in {output_path}")
    print(f"  Train puzzles: {n_train}")
    print(f"  Test puzzles: {n_test}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")

    # Create training data
    # For simplicity, we create random sequences with mostly padding
    # Real ARC data would have structured grid patterns
    train_inputs = np.random.randint(0, 10, size=(n_train, seq_len), dtype=np.uint8)
    train_labels = np.random.randint(0, 10, size=(n_train, seq_len), dtype=np.uint8)

    # Add some structure: first few tokens are the "pattern", rest is padding
    for i in range(n_train):
        # Pattern length varies
        pattern_len = np.random.randint(100, 300)
        # Rest is padding
        train_inputs[i, pattern_len:] = 10  # padding token
        train_labels[i, pattern_len:] = 10  # padding token

    # Create test data
    test_inputs = np.random.randint(0, 10, size=(n_test, seq_len), dtype=np.uint8)
    test_labels = np.random.randint(0, 10, size=(n_test, seq_len), dtype=np.uint8)

    for i in range(n_test):
        pattern_len = np.random.randint(100, 300)
        test_inputs[i, pattern_len:] = 10
        test_labels[i, pattern_len:] = 10

    # Save to files
    np.save(output_path / "train_inputs.npy", train_inputs)
    np.save(output_path / "train_labels.npy", train_labels)
    np.save(output_path / "test_inputs.npy", test_inputs)
    np.save(output_path / "test_labels.npy", test_labels)

    # Save metadata
    metadata = {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "n_train_puzzles": n_train,
        "n_test_puzzles": n_test,
        "note": "This is sample/synthetic data for testing. Use real ARC-AGI-1 data for actual training.",
    }

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSample data created successfully!")
    print(f"  train_inputs.npy: {train_inputs.shape}")
    print(f"  train_labels.npy: {train_labels.shape}")
    print(f"  test_inputs.npy: {test_inputs.shape}")
    print(f"  test_labels.npy: {test_labels.shape}")
    print(f"  metadata.json: {metadata}")
    print(f"\nTo use this data, run:")
    print(f"  python train.py --dataset arc-sample -e 5")


if __name__ == "__main__":
    create_sample_arc_data()
