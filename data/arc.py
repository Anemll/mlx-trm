"""ARC-AGI dataset loader for MLX.

The ARC (Abstraction and Reasoning Corpus) dataset consists of puzzles
represented as 30x30 grids with values 0-9 (colors/cell values).

Data format:
- Grids are flattened to sequences of length 900 (30x30)
- Vocabulary: 12 tokens (0-9 colors + padding + EOS)
- Each puzzle has input/output pairs
"""

import json
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np


class ARCDataset:
    """Iterator for ARC puzzle dataset.

    The dataset is expected to be in a directory with:
    - train_inputs.npy: Training input sequences
    - train_labels.npy: Training label sequences
    - test_inputs.npy: Test input sequences
    - test_labels.npy: Test label sequences
    - metadata.json: Dataset metadata
    """

    def __init__(self, data_path: str, split: str = "train", batch_size: int = 32):
        """Initialize ARC dataset.

        Args:
            data_path: Path to directory containing ARC data files
            split: Either "train" or "test"
            batch_size: Number of puzzles per batch
        """
        self.data_path = Path(data_path)
        self.split = split
        self.batch_size = batch_size

        # Load data
        inputs_file = self.data_path / f"{split}_inputs.npy"
        labels_file = self.data_path / f"{split}_labels.npy"

        if not inputs_file.exists() or not labels_file.exists():
            raise FileNotFoundError(
                f"ARC data files not found in {self.data_path}. "
                f"Expected {inputs_file} and {labels_file}"
            )

        self.inputs = np.load(inputs_file)  # shape: (n_puzzles, seq_len)
        self.labels = np.load(labels_file)  # shape: (n_puzzles, seq_len)

        # Load metadata if available
        metadata_file = self.data_path / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
        else:
            # Infer metadata from data
            self.metadata = {
                "seq_len": self.inputs.shape[1],
                "vocab_size": 12,  # 0-9 colors + padding + EOS
                "n_puzzles": len(self.inputs),
            }

        self.n_puzzles = len(self.inputs)
        self.indices = np.arange(self.n_puzzles)
        self._current_idx = 0

        if split == "train":
            # Shuffle training data
            np.random.shuffle(self.indices)

    def reset(self):
        """Reset the iterator."""
        if self.split == "train":
            np.random.shuffle(self.indices)

    def __iter__(self):
        """Return self as iterator."""
        self._current_idx = 0
        return self

    def __next__(self) -> dict:
        """Get next batch."""
        if self._current_idx >= self.n_puzzles:
            raise StopIteration

        i = self._current_idx
        self._current_idx += self.batch_size
        batch_indices = self.indices[i : i + self.batch_size]

        # Get batch data
        batch_inputs = self.inputs[batch_indices]
        batch_labels = self.labels[batch_indices]

        # Convert to format expected by the model
        # The model expects 2D images, so we reshape the sequences
        # back to grid format (batch, height, width, channels)
        # For ARC, we'll treat it as 1-channel image with the token as value
        seq_len = batch_inputs.shape[1]
        grid_size = int(np.sqrt(seq_len))  # Should be 30

        # Reshape to (batch, height, width) then add channel dimension
        batch_inputs_grid = batch_inputs.reshape(-1, grid_size, grid_size)
        batch_inputs_grid = np.expand_dims(batch_inputs_grid, axis=-1)  # Add channel dim

        return {
            "image": batch_inputs_grid.astype(np.float32),
            "label": batch_labels[:, 0].astype(np.int32),  # Use first token as label for now
        }

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (self.n_puzzles + self.batch_size - 1) // self.batch_size


def arc_agi_1(batch_size: int = 32, data_path: str = "data/arc1concept-aug-1000"):
    """Load ARC-AGI-1 dataset.

    Args:
        batch_size: Batch size for training
        data_path: Path to ARC-AGI-1 data directory

    Returns:
        (train_dataset, test_dataset, metadata)
    """
    train = ARCDataset(data_path, split="train", batch_size=batch_size)
    test = ARCDataset(data_path, split="test", batch_size=batch_size)

    # Calculate steps per epoch
    steps_per_epoch = len(train)

    metadata = {
        "seq_len": train.metadata["seq_len"],
        "vocab_size": train.metadata["vocab_size"],
        "n_train_puzzles": train.n_puzzles,
        "n_test_puzzles": test.n_puzzles,
        "steps_per_epoch": steps_per_epoch,
    }

    return train, test, metadata
