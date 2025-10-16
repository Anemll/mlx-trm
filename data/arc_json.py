"""ARC-AGI JSON dataset loader for token-based models.

Loads ARC tasks from JSON files and converts them to token sequences.
"""

import json
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np


def grid_to_sequence(grid: List[List[int]], max_size: int = 30) -> np.ndarray:
    """Convert a 2D grid to a flattened sequence with padding.

    Args:
        grid: 2D list of integers (0-9)
        max_size: Maximum grid dimension (30 for ARC)

    Returns:
        Flattened sequence of length max_size*max_size with padding (token 10)
    """
    grid = np.array(grid, dtype=np.int32)
    h, w = grid.shape

    # Pad to max_size x max_size
    padded = np.full((max_size, max_size), fill_value=10, dtype=np.int32)  # 10 = padding
    padded[:h, :w] = grid

    # Flatten to sequence
    return padded.flatten()


def sequence_to_grid(sequence: np.ndarray, original_shape: Tuple[int, int] = None) -> np.ndarray:
    """Convert a flattened sequence back to a 2D grid.

    Args:
        sequence: Flattened sequence of tokens
        original_shape: (height, width) of the original grid. If None, inferred from non-padding tokens

    Returns:
        2D grid
    """
    max_size = int(np.sqrt(len(sequence)))
    grid = sequence.reshape(max_size, max_size)

    if original_shape:
        h, w = original_shape
        return grid[:h, :w]

    # Find the actual size by locating padding
    # Find last non-padding row and column
    non_pad_mask = grid != 10
    if not non_pad_mask.any():
        return np.array([[]])

    rows_with_data = non_pad_mask.any(axis=1)
    cols_with_data = non_pad_mask.any(axis=0)

    h = np.where(rows_with_data)[0][-1] + 1 if rows_with_data.any() else 1
    w = np.where(cols_with_data)[0][-1] + 1 if cols_with_data.any() else 1

    return grid[:h, :w]


class ARCJSONDataset:
    """Iterator for ARC tasks from JSON files.

    Each ARC task contains:
    - Multiple training examples (input/output pairs)
    - One test example (input/output pair)

    We flatten each example into input_sequence -> output_sequence format.
    """

    def __init__(self, data_dir: str, split: str = "training", batch_size: int = 32, max_size: int = 30):
        """Initialize ARC JSON dataset.

        Args:
            data_dir: Path to ARC-AGI data directory (e.g., "data/ARC-AGI/data")
            split: Either "training" or "evaluation"
            batch_size: Number of examples per batch
            max_size: Maximum grid dimension (30 for ARC)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.max_size = max_size
        self.seq_len = max_size * max_size

        # Load all task files
        task_dir = self.data_dir / split
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        self.task_files = sorted(list(task_dir.glob("*.json")))
        if not self.task_files:
            raise ValueError(f"No JSON files found in {task_dir}")

        # Preload all examples
        self.examples = []
        for task_file in self.task_files:
            with open(task_file, "r") as f:
                task = json.load(f)

            # Add training examples from this task
            for train_example in task.get("train", []):
                input_grid = train_example["input"]
                output_grid = train_example["output"]

                input_seq = grid_to_sequence(input_grid, self.max_size)
                output_seq = grid_to_sequence(output_grid, self.max_size)

                self.examples.append({
                    "input_tokens": input_seq,
                    "output_tokens": output_seq,
                    "task_id": task_file.stem,
                })

        self.n_examples = len(self.examples)
        self.indices = np.arange(self.n_examples)
        self._current_idx = 0

        if split == "training":
            # Shuffle training data
            np.random.shuffle(self.indices)

        print(f"Loaded {self.n_examples} examples from {len(self.task_files)} tasks ({split})")

    def reset(self):
        """Reset the iterator."""
        self._current_idx = 0
        if self.split == "training":
            np.random.shuffle(self.indices)

    def __iter__(self):
        """Return self as iterator."""
        self._current_idx = 0
        return self

    def __next__(self) -> dict:
        """Get next batch."""
        if self._current_idx >= self.n_examples:
            raise StopIteration

        i = self._current_idx
        self._current_idx += self.batch_size
        batch_indices = self.indices[i : i + self.batch_size]

        # Gather batch
        batch_input_tokens = []
        batch_output_tokens = []

        for idx in batch_indices:
            example = self.examples[idx]
            batch_input_tokens.append(example["input_tokens"])
            batch_output_tokens.append(example["output_tokens"])

        return {
            "input_tokens": np.array(batch_input_tokens, dtype=np.int32),
            "output_tokens": np.array(batch_output_tokens, dtype=np.int32),
        }

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (self.n_examples + self.batch_size - 1) // self.batch_size


def arc_agi_json(batch_size: int = 32, data_dir: str = "data/ARC-AGI/data"):
    """Load ARC-AGI dataset from JSON files.

    Args:
        batch_size: Batch size for training
        data_dir: Path to ARC-AGI data directory

    Returns:
        (train_dataset, test_dataset, metadata)
    """
    train = ARCJSONDataset(data_dir, split="training", batch_size=batch_size)
    test = ARCJSONDataset(data_dir, split="evaluation", batch_size=batch_size)

    # Calculate steps per epoch
    steps_per_epoch = len(train)

    metadata = {
        "vocab_size": 12,  # 0-9 colors + padding (10) + EOS (11)
        "max_seq_len": 900,  # 30x30
        "n_train_examples": train.n_examples,
        "n_test_examples": test.n_examples,
        "n_train_tasks": len(train.task_files),
        "n_test_tasks": len(test.task_files),
        "steps_per_epoch": steps_per_epoch,
    }

    return train, test, metadata


if __name__ == "__main__":
    # Test the data loader
    print("Testing ARC JSON data loader...")

    train, test, meta = arc_agi_json(batch_size=4)

    print(f"\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print(f"\nFetching first batch...")
    batch = next(iter(train))

    print(f"Batch shapes:")
    print(f"  input_tokens: {batch['input_tokens'].shape}")
    print(f"  output_tokens: {batch['output_tokens'].shape}")

    print(f"\nFirst example input (first 100 tokens):")
    print(batch['input_tokens'][0][:100])

    # Test grid conversion
    print(f"\nTesting grid conversion...")
    test_grid = [[1, 2, 3], [4, 5, 6]]
    seq = grid_to_sequence(test_grid, max_size=5)
    print(f"Original grid:\n{np.array(test_grid)}")
    print(f"Sequence (first 30 tokens): {seq[:30]}")
    recovered = sequence_to_grid(seq, original_shape=(2, 3))
    print(f"Recovered grid:\n{recovered}")
