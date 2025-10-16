"""ARC-AGI test dataset loader for validation.

This loader specifically uses task["test"] pairs for validation,
NOT the task["train"] demonstration pairs. This matches the proper
ARC evaluation methodology.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
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


class ARCTestDataset:
    """ARC test dataset using actual test pairs from tasks.

    IMPORTANT: This uses task["test"] pairs, NOT task["train"] pairs.
    This is the proper way to evaluate ARC models - on the test examples
    that require generalization from the demonstration pairs.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "evaluation",
        batch_size: int = 32,
        max_size: int = 30,
        use_puzzle_ids: bool = True,
        puzzle_id_map: Optional[Dict[str, int]] = None,
    ):
        """Initialize ARC test dataset.

        Args:
            data_dir: Path to ARC-AGI data directory
            split: Either "training" or "evaluation"
            batch_size: Number of examples per batch
            max_size: Maximum grid dimension (30 for ARC)
            use_puzzle_ids: Whether to include puzzle IDs in batches
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.max_size = max_size
        self.seq_len = max_size * max_size
        self.use_puzzle_ids = use_puzzle_ids
        self._external_puzzle_id_map = puzzle_id_map

        # Load all task files
        task_dir = self.data_dir / split
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        self.task_files = sorted(list(task_dir.glob("*.json")))
        if not self.task_files:
            raise ValueError(f"No JSON files found in {task_dir}")

        # Load TEST examples (not train!)
        self.examples = []
        if puzzle_id_map is not None:
            # Copy to avoid side effects
            self.puzzle_id_map = dict(puzzle_id_map)
            next_puzzle_id = max(self.puzzle_id_map.values(), default=0) + 1
        else:
            self.puzzle_id_map = {}
            next_puzzle_id = 1

        for task_file in self.task_files:
            with open(task_file, "r") as f:
                task = json.load(f)

            task_id = task_file.stem

            # Assign puzzle ID
            if task_id not in self.puzzle_id_map:
                self.puzzle_id_map[task_id] = next_puzzle_id
                next_puzzle_id += 1

            puzzle_id = self.puzzle_id_map[task_id]

            # IMPORTANT: Use test examples, not train examples!
            test_examples = task.get("test", [])

            if not test_examples:
                print(f"Warning: No test examples in {task_id}, skipping")
                continue

            # Add each test example
            for test_idx, test_example in enumerate(test_examples):
                input_grid = test_example["input"]
                output_grid = test_example["output"]

                input_seq = grid_to_sequence(input_grid, self.max_size)
                output_seq = grid_to_sequence(output_grid, self.max_size)

                example_data = {
                    "input_tokens": input_seq,
                    "output_tokens": output_seq,
                    "task_id": task_id,
                    "test_idx": test_idx,
                }

                if self.use_puzzle_ids:
                    example_data["puzzle_id"] = puzzle_id

                self.examples.append(example_data)

        self.n_examples = len(self.examples)
        self.n_tasks = len(self.puzzle_id_map)
        self.indices = np.arange(self.n_examples)
        self._current_idx = 0

        print(f"Loaded {self.n_examples} TEST examples from {self.n_tasks} tasks ({split})")
        print(f"  Using task['test'] pairs for proper ARC evaluation")
        if self.use_puzzle_ids:
            print(f"  Unique puzzle IDs: {len(self.puzzle_id_map)}")

    def reset(self):
        """Reset the iterator."""
        self._current_idx = 0
        # Don't shuffle validation data - keep it deterministic

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
        batch_puzzle_ids = [] if self.use_puzzle_ids else None

        for idx in batch_indices:
            example = self.examples[idx]
            batch_input_tokens.append(example["input_tokens"])
            batch_output_tokens.append(example["output_tokens"])
            if self.use_puzzle_ids:
                batch_puzzle_ids.append(example["puzzle_id"])

        batch = {
            "input_tokens": np.array(batch_input_tokens, dtype=np.int32),
            "output_tokens": np.array(batch_output_tokens, dtype=np.int32),
        }

        if self.use_puzzle_ids:
            batch["puzzle_ids"] = np.array(batch_puzzle_ids, dtype=np.int32)

        return batch

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (self.n_examples + self.batch_size - 1) // self.batch_size

    def get_num_puzzle_identifiers(self) -> int:
        """Get the maximum number of unique puzzle identifiers."""
        return len(self.puzzle_id_map) + 1


def arc_test_dataset(
    batch_size: int = 32,
    data_dir: str = "data/ARC-AGI/data",
    split: str = "evaluation",
    use_puzzle_ids: bool = True,
    puzzle_id_map: Optional[Dict[str, int]] = None,
):
    """Load ARC test dataset for validation.

    This specifically loads task["test"] pairs, not task["train"] pairs.

    Args:
        batch_size: Batch size
        data_dir: Path to ARC-AGI data directory
        split: Dataset split ("training" or "evaluation")
        use_puzzle_ids: Whether to include puzzle IDs

    Returns:
        Test dataset with proper test examples
    """
    return ARCTestDataset(
        data_dir=data_dir,
        split=split,
        batch_size=batch_size,
        use_puzzle_ids=use_puzzle_ids,
        puzzle_id_map=puzzle_id_map,
    )


if __name__ == "__main__":
    # Test the loader
    print("Testing ARC test dataset loader...")

    test_data = arc_test_dataset(batch_size=4, split="evaluation")

    print(f"\nTotal test examples: {test_data.n_examples}")
    print(f"Total tasks: {test_data.n_tasks}")
    print(f"Batches: {len(test_data)}")

    # Get a batch
    batch = next(iter(test_data))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")

    print("\nNOTE: These are TEST pairs from task['test'], not training demonstrations!")
