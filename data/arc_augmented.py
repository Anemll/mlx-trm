"""ARC-AGI augmented dataset loader with puzzle IDs matching TinyRecursiveModels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple

import numpy as np


class AugmentationConfig(NamedTuple):
    """Configuration for ARC data augmentations."""

    enable_dihedral: bool = True
    enable_color_permute: bool = True
    enable_translation: bool = False
    max_augmentations_per_puzzle: int = 1000
    puzzle_id_offset: int = 1


def apply_dihedral_transform(grid: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply one of 8 dihedral transformations (D4 group).

    0: Identity
    1: 90° rotation
    2: 180° rotation
    3: 270° rotation
    4: Horizontal flip
    5: Vertical flip
    6: Diagonal flip (transpose)
    7: Anti-diagonal flip
    """
    if transform_id == 0:
        return grid
    elif transform_id == 1:
        return np.rot90(grid, 1)
    elif transform_id == 2:
        return np.rot90(grid, 2)
    elif transform_id == 3:
        return np.rot90(grid, 3)
    elif transform_id == 4:
        return np.fliplr(grid)
    elif transform_id == 5:
        return np.flipud(grid)
    elif transform_id == 6:
        return grid.T
    elif transform_id == 7:
        return np.rot90(np.fliplr(grid), 1)
    else:
        raise ValueError(f"Invalid transform_id: {transform_id}")


def apply_color_permutation(grid: np.ndarray, perm: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply color permutation, keeping black (0) unchanged.

    Args:
        grid: Input grid with values 0-9
        perm: Optional permutation array. If None, generates random permutation

    Returns:
        Grid with permuted colors
    """
    if perm is None:
        # Generate random permutation for colors 1-9 (keep 0 as black)
        colors = np.arange(1, 10)
        np.random.shuffle(colors)
        perm = np.zeros(10, dtype=int)
        perm[0] = 0  # Black stays black
        perm[1:] = colors

    # Apply permutation
    result = np.zeros_like(grid)
    for old_color in range(10):
        mask = grid == old_color
        result[mask] = perm[old_color]

    return result


def grid_to_sequence(grid: np.ndarray, max_size: int = 30) -> np.ndarray:
    """Convert a 2D grid to a flattened sequence with padding.

    Args:
        grid: 2D array of integers (0-9)
        max_size: Maximum grid dimension (30 for ARC)

    Returns:
        Flattened sequence of length max_size*max_size with padding (token 10)
    """
    h, w = grid.shape

    # Pad to max_size x max_size
    padded = np.full((max_size, max_size), fill_value=10, dtype=np.int32)  # 10 = padding
    padded[:h, :w] = grid

    # Flatten to sequence
    return padded.flatten()


def build_puzzle_id_map(
    data_dir: str,
    splits: Iterable[str] = ("training", "evaluation"),
    start: int = 1,
) -> Dict[str, int]:
    """Create a consistent mapping from task_id to puzzle identifier.

    Scans the provided splits so that training, validation, and evaluation
    share the same puzzle ID indexing. This mirrors the reference
    TinyRecursiveModels pipeline which assigns a single embedding per task
    regardless of split.
    """

    base = Path(data_dir)
    all_task_ids: set[str] = set()

    for split in splits:
        task_dir = base / split
        if not task_dir.exists():
            continue
        all_task_ids.update(task_file.stem for task_file in task_dir.glob("*.json"))

    puzzle_id_map: Dict[str, int] = {}
    next_id = start
    for task_id in sorted(all_task_ids):
        puzzle_id_map[task_id] = next_id
        next_id += 1

    return puzzle_id_map


class ARCAugmentedDataset:
    """ARC dataset with augmentations and puzzle identifiers.

    Implements the augmentation strategy from TinyRecursiveModels:
    - Each puzzle gets a unique identifier
    - Augmentations are encoded in the identifier string using "|||" separator
    - Supports dihedral transformations and color permutations
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        batch_size: int = 32,
        max_size: int = 30,
        augmentation_config: Optional[AugmentationConfig] = None,
        seed: int = 0,
         puzzle_id_map: Optional[Dict[str, int]] = None,
        additional_splits: Optional[Iterable[str]] = None,
    ):
        """Initialize augmented ARC dataset.

        Args:
            data_dir: Path to ARC-AGI data directory
            split: Either "training" or "evaluation"
            batch_size: Number of examples per batch
            max_size: Maximum grid dimension (30 for ARC)
            augmentation_config: Configuration for augmentations
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.batch_size = batch_size
        self.max_size = max_size
        self.seq_len = max_size * max_size
        self.config = augmentation_config or AugmentationConfig()
        self.additional_splits = list(additional_splits or [])

        np.random.seed(seed)

        # Load all task files
        task_dir = self.data_dir / split
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {task_dir}")

        self.primary_task_files = sorted(list(task_dir.glob("*.json")))
        if not self.primary_task_files:
            raise ValueError(f"No JSON files found in {task_dir}")
        # Backwards compatibility attribute
        self.task_files = self.primary_task_files

        # Track task IDs for reporting
        self.primary_task_ids = {path.stem for path in self.primary_task_files}

        # Initialize puzzle ID mapping
        if puzzle_id_map is None:
            self.puzzle_id_map: Dict[str, int] = {}
            self.next_puzzle_id = self.config.puzzle_id_offset
        else:
            self.puzzle_id_map = puzzle_id_map
            self.next_puzzle_id = max(puzzle_id_map.values(), default=self.config.puzzle_id_offset - 1) + 1

        self._loaded_task_ids: set[str] = set()

        # Load and augment examples
        self.examples = []
        self._load_examples()

        self.n_examples = len(self.examples)
        self.indices = np.arange(self.n_examples)
        self._current_idx = 0

        if split == "training":
            # Shuffle training data
            np.random.shuffle(self.indices)

        # Count unique puzzle IDs
        unique_puzzle_ids = set(ex["puzzle_id"] for ex in self.examples)

        total_tasks = len(self._loaded_task_ids)
        extra_tasks = total_tasks - len(self.primary_task_ids)
        extra_info = f" + {extra_tasks} extra" if extra_tasks > 0 else ""
        print(f"Loaded {self.n_examples} examples from {total_tasks} tasks ({split}{extra_info})")
        print(f"  Unique puzzle IDs: {len(unique_puzzle_ids)}")
        if split == "training" and (self.config.enable_dihedral or self.config.enable_color_permute):
            print(f"  Augmentations enabled: dihedral={self.config.enable_dihedral}, "
                  f"color={self.config.enable_color_permute}")

    def _load_examples(self):
        """Load and optionally augment all examples."""
        split_order = [self.split] + [s for s in self.additional_splits if s != self.split]
        for current_split in split_order:
            task_files = (
                self.primary_task_files
                if current_split == self.split
                else sorted((self.data_dir / current_split).glob("*.json"))
            )
            if not task_files:
                continue

            allow_aug = (
                current_split == "training"
                and self.split == "training"
                and (self.config.enable_dihedral or self.config.enable_color_permute)
            )

            for task_file in task_files:
                with open(task_file, "r") as f:
                    task = json.load(f)

                task_id = task_file.stem
                self._loaded_task_ids.add(task_id)

                # Assign base puzzle ID for this task
                if task_id not in self.puzzle_id_map:
                    self.puzzle_id_map[task_id] = self.next_puzzle_id
                    self.next_puzzle_id += 1

                base_puzzle_id = self.puzzle_id_map[task_id]

                # Process training (demonstration) examples only
                train_examples = task.get("train", [])

                if allow_aug:
                    self._add_augmented_examples(train_examples, task_id, base_puzzle_id)
                else:
                    for example in train_examples:
                        self._add_example(example, task_id, base_puzzle_id)

    def _add_example(
        self,
        example: dict,
        task_id: str,
        puzzle_id: int,
        transform_id: int = 0,
        color_perm: Optional[np.ndarray] = None,
    ):
        """Add a single example (possibly augmented) to the dataset."""
        input_grid = np.array(example["input"], dtype=np.int32)
        output_grid = np.array(example["output"], dtype=np.int32)

        # Apply transformations if specified
        if transform_id > 0:
            input_grid = apply_dihedral_transform(input_grid, transform_id)
            output_grid = apply_dihedral_transform(output_grid, transform_id)

        if color_perm is not None:
            input_grid = apply_color_permutation(input_grid, color_perm)
            output_grid = apply_color_permutation(output_grid, color_perm)

        # Convert to sequences
        input_seq = grid_to_sequence(input_grid, self.max_size)
        output_seq = grid_to_sequence(output_grid, self.max_size)

        # Build augmentation string for puzzle identifier
        aug_parts = []
        if transform_id > 0:
            aug_parts.append(f"d{transform_id}")
        if color_perm is not None:
            # Encode color permutation as string
            perm_str = "".join(str(color_perm[i]) for i in range(10))
            aug_parts.append(f"c{perm_str}")

        # Create puzzle identifier with augmentation info
        if aug_parts:
            puzzle_id_str = f"{puzzle_id}|||{'_'.join(aug_parts)}"
        else:
            puzzle_id_str = str(puzzle_id)

        self.examples.append({
            "input_tokens": input_seq,
            "output_tokens": output_seq,
            "task_id": task_id,
            "puzzle_id": puzzle_id,  # Numeric ID
            "puzzle_id_str": puzzle_id_str,  # String with augmentation info
            "transform_id": transform_id,
        })

    def _add_augmented_examples(
        self,
        train_examples: List[dict],
        task_id: str,
        base_puzzle_id: int,
    ):
        """Add augmented versions of training examples."""
        # Always add original examples
        for example in train_examples:
            self._add_example(example, task_id, base_puzzle_id)

        # Generate augmentations
        n_augmentations = 0
        max_aug = self.config.max_augmentations_per_puzzle // len(train_examples)

        for example in train_examples:
            if n_augmentations >= self.config.max_augmentations_per_puzzle:
                break

            # Dihedral transformations
            if self.config.enable_dihedral:
                for transform_id in range(1, 8):  # Skip 0 (identity)
                    if n_augmentations >= max_aug:
                        break
                    self._add_example(example, task_id, base_puzzle_id, transform_id=transform_id)
                    n_augmentations += 1

            # Color permutations
            if self.config.enable_color_permute:
                # Generate a few random color permutations
                n_color_perms = min(10, max_aug - n_augmentations)
                for _ in range(n_color_perms):
                    if n_augmentations >= max_aug:
                        break
                    # Generate random color permutation
                    colors = np.arange(1, 10)
                    np.random.shuffle(colors)
                    perm = np.zeros(10, dtype=int)
                    perm[0] = 0  # Black stays black
                    perm[1:] = colors

                    self._add_example(example, task_id, base_puzzle_id, color_perm=perm)
                    n_augmentations += 1

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
        batch_puzzle_ids = []

        for idx in batch_indices:
            example = self.examples[idx]
            batch_input_tokens.append(example["input_tokens"])
            batch_output_tokens.append(example["output_tokens"])
            batch_puzzle_ids.append(example["puzzle_id"])

        return {
            "input_tokens": np.array(batch_input_tokens, dtype=np.int32),
            "output_tokens": np.array(batch_output_tokens, dtype=np.int32),
            "puzzle_ids": np.array(batch_puzzle_ids, dtype=np.int32),
        }

    def __len__(self) -> int:
        """Number of batches in the dataset."""
        return (self.n_examples + self.batch_size - 1) // self.batch_size

    def get_num_puzzle_identifiers(self) -> int:
        """Get the maximum number of unique puzzle identifiers."""
        return self.next_puzzle_id


def arc_agi_augmented(
    batch_size: int = 32,
    data_dir: str = "data/ARC-AGI/data",
    enable_augmentations: bool = True,
    seed: int = 0,
    puzzle_id_map: Optional[Dict[str, int]] = None,
    include_eval_demos: bool = True,
    max_augmentations_per_puzzle: int = 300,
):
    """Load augmented ARC-AGI dataset.

    Args:
        batch_size: Batch size for training
        data_dir: Path to ARC-AGI data directory
        enable_augmentations: Whether to enable data augmentations
        seed: Random seed for reproducibility
        puzzle_id_map: Optional pre-built puzzle ID mapping
        include_eval_demos: Include evaluation demo pairs in training
        max_augmentations_per_puzzle: Max augmented versions per task (default: 300)
            - Paper uses 1000
            - ARC Prize analysis shows 300 achieves near-max performance
            - 30 augmentations gets within 4% of max performance

    Returns:
        (train_dataset, test_dataset, metadata)
    """
    # Configure augmentations
    train_config = AugmentationConfig(
        enable_dihedral=enable_augmentations,
        enable_color_permute=enable_augmentations,
        enable_translation=False,  # Not implemented yet
        max_augmentations_per_puzzle=max_augmentations_per_puzzle,
    )

    eval_config = AugmentationConfig(
        enable_dihedral=False,
        enable_color_permute=False,
        enable_translation=False,
    )

    shared_map = puzzle_id_map or build_puzzle_id_map(data_dir)

    additional_splits = ["evaluation"] if include_eval_demos else []

    train = ARCAugmentedDataset(
        data_dir,
        split="training",
        batch_size=batch_size,
        augmentation_config=train_config,
        seed=seed,
        puzzle_id_map=shared_map,
        additional_splits=additional_splits,
    )

    test = ARCAugmentedDataset(
        data_dir,
        split="evaluation",
        batch_size=batch_size,
        augmentation_config=eval_config,
        seed=seed,
        puzzle_id_map=shared_map,
    )

    # Calculate steps per epoch
    steps_per_epoch = len(train)

    metadata = {
        "vocab_size": 12,  # 0-9 colors + padding (10) + EOS (11)
        "max_seq_len": 900,  # 30x30
        "n_train_examples": train.n_examples,
        "n_test_examples": test.n_examples,
        "n_train_tasks": len(train._loaded_task_ids),
        "n_test_tasks": len(test._loaded_task_ids),
        "steps_per_epoch": steps_per_epoch,
        "num_puzzle_identifiers": max(
            train.get_num_puzzle_identifiers(),
            test.get_num_puzzle_identifiers()
        ),
    }

    return train, test, metadata


if __name__ == "__main__":
    # Test the augmented data loader
    print("Testing augmented ARC data loader...")

    train, test, meta = arc_agi_augmented(batch_size=4, enable_augmentations=True)

    print(f"\nMetadata:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print(f"\nFetching first batch...")
    batch = next(iter(train))

    print(f"Batch shapes:")
    print(f"  input_tokens: {batch['input_tokens'].shape}")
    print(f"  output_tokens: {batch['output_tokens'].shape}")
    print(f"  puzzle_ids: {batch['puzzle_ids'].shape}")

    print(f"\nFirst example:")
    print(f"  Puzzle ID: {batch['puzzle_ids'][0]}")
    print(f"  Input tokens (first 50): {batch['input_tokens'][0][:50]}")
