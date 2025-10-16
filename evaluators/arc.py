"""ARC evaluation metrics matching TinyRecursiveModels implementation.

This evaluator implements exact-match evaluation for ARC tasks using grid hashing.
Based on: https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/evaluators/arc.py
"""

from typing import Dict, List, Optional, Tuple
import json
import hashlib
import numpy as np
from pathlib import Path


def grid_hash(grid: np.ndarray) -> str:
    """Create a unique hash for a grid.

    Args:
        grid: 2D numpy array representing an ARC grid

    Returns:
        SHA256 hash string of the grid
    """
    # Ensure consistent dtype for hashing
    grid = np.asarray(grid, dtype=np.int32)
    # Create hash from bytes representation
    return hashlib.sha256(grid.tobytes()).hexdigest()


def arc_grid_to_np(grid: List[List[int]]) -> np.ndarray:
    """Convert ARC JSON grid to numpy array.

    Args:
        grid: List of lists representing an ARC grid

    Returns:
        Numpy array of the grid
    """
    return np.array(grid, dtype=np.int32)


def crop_padding(grid: np.ndarray, padding_value: int = 10) -> np.ndarray:
    """Remove padding from a flattened grid.

    Args:
        grid: Flattened grid (e.g., 900 tokens for 30x30)
        padding_value: The padding token value

    Returns:
        Cropped 2D grid without padding
    """
    # Reshape to 2D
    size = int(np.sqrt(len(grid)))
    grid = grid.reshape(size, size)

    # Find the actual size by locating padding
    non_pad_mask = grid != padding_value
    if not non_pad_mask.any():
        return np.array([[]], dtype=np.int32)

    # Find last non-padding row and column
    rows_with_data = non_pad_mask.any(axis=1)
    cols_with_data = non_pad_mask.any(axis=0)

    last_row = np.where(rows_with_data)[0][-1] + 1 if rows_with_data.any() else 1
    last_col = np.where(cols_with_data)[0][-1] + 1 if cols_with_data.any() else 1

    return grid[:last_row, :last_col]


class ARCEvaluator:
    """Evaluator for ARC tasks with exact-match scoring.

    This evaluator implements the same logic as TinyRecursiveModels:
    - Exact grid matching using hashes
    - Pass@K metrics
    - Support for multiple prediction attempts
    """

    def __init__(
        self,
        data_dir: str = "data/ARC-AGI/data",
        pass_ks: Tuple[int, ...] = (1,),
        verbose: bool = True
    ):
        """Initialize ARC evaluator.

        Args:
            data_dir: Path to ARC-AGI data directory
            pass_ks: K values for pass@K metrics
            verbose: Whether to print detailed results
        """
        self.data_dir = Path(data_dir)
        self.pass_ks = pass_ks
        self.verbose = verbose

        # Load evaluation tasks
        self.eval_tasks = self._load_evaluation_tasks()

        # Storage for predictions
        self.predictions = {}

    def _load_evaluation_tasks(self) -> Dict[str, Dict]:
        """Load evaluation tasks from JSON files.

        Returns:
            Dictionary mapping task IDs to task data
        """
        eval_dir = self.data_dir / "evaluation"
        if not eval_dir.exists():
            raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

        tasks = {}
        for task_file in sorted(eval_dir.glob("*.json")):
            with open(task_file, "r") as f:
                task_data = json.load(f)
            task_id = task_file.stem
            tasks[task_id] = task_data

        if self.verbose:
            print(f"Loaded {len(tasks)} evaluation tasks")

        return tasks

    def reset(self):
        """Reset predictions for new evaluation run."""
        self.predictions = {}

    def add_prediction(
        self,
        task_id: str,
        test_idx: int,
        pred_grid: np.ndarray,
        confidence: float = 1.0
    ):
        """Add a prediction for evaluation.

        Args:
            task_id: Task identifier
            test_idx: Index of test example within task
            pred_grid: Predicted output grid
            confidence: Confidence score for this prediction
        """
        # Remove padding if present
        if len(pred_grid.shape) == 1:
            pred_grid = crop_padding(pred_grid)

        # Create hash for prediction
        pred_hash = grid_hash(pred_grid)

        # Store prediction
        if task_id not in self.predictions:
            self.predictions[task_id] = {}
        if test_idx not in self.predictions[task_id]:
            self.predictions[task_id][test_idx] = []

        self.predictions[task_id][test_idx].append({
            'grid': pred_grid,
            'hash': pred_hash,
            'confidence': confidence
        })

    def evaluate(self) -> Dict[str, float]:
        """Evaluate all predictions against ground truth.

        Returns:
            Dictionary of metrics (pass@K scores)
        """
        results = {f"pass@{k}": 0.0 for k in self.pass_ks}
        total_tests = 0
        correct_by_k = {k: 0 for k in self.pass_ks}

        for task_id, task_data in self.eval_tasks.items():
            # Skip if no predictions for this task
            if task_id not in self.predictions:
                if self.verbose:
                    print(f"Warning: No predictions for task {task_id}")
                continue

            # Evaluate each test example
            for test_idx, test_example in enumerate(task_data.get("test", [])):
                total_tests += 1

                # Get ground truth
                gt_grid = arc_grid_to_np(test_example["output"])
                gt_hash = grid_hash(gt_grid)

                # Get predictions for this test
                if test_idx not in self.predictions[task_id]:
                    if self.verbose:
                        print(f"Warning: No prediction for task {task_id} test {test_idx}")
                    continue

                preds = self.predictions[task_id][test_idx]

                # Sort by confidence
                preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)

                # Check pass@K
                for k in self.pass_ks:
                    # Check if any of top-k predictions match
                    for pred in preds[:k]:
                        if pred['hash'] == gt_hash:
                            correct_by_k[k] += 1
                            break

        # Calculate final metrics
        if total_tests > 0:
            for k in self.pass_ks:
                results[f"pass@{k}"] = correct_by_k[k] / total_tests

        if self.verbose:
            print(f"\nEvaluation Results ({total_tests} test examples):")
            for k in self.pass_ks:
                print(f"  Pass@{k}: {results[f'pass@{k}']:.2%}")

        return results

    def evaluate_batch(
        self,
        task_ids: List[str],
        test_indices: List[int],
        predictions: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.

        Args:
            task_ids: List of task IDs
            test_indices: List of test indices within tasks
            predictions: Array of predicted grids (batch_size, seq_len)
            confidences: Optional confidence scores

        Returns:
            Dictionary of metrics
        """
        if confidences is None:
            confidences = np.ones(len(predictions))

        # Add all predictions
        for task_id, test_idx, pred, conf in zip(
            task_ids, test_indices, predictions, confidences
        ):
            self.add_prediction(task_id, test_idx, pred, conf)

        # Evaluate
        return self.evaluate()


class ARCTestDataset:
    """Dataset for ARC test examples (not training examples).

    This loads the actual test cases from task["test"] for proper evaluation.
    """

    def __init__(
        self,
        data_dir: str = "data/ARC-AGI/data",
        split: str = "evaluation",
        include_output: bool = True
    ):
        """Initialize ARC test dataset.

        Args:
            data_dir: Path to ARC-AGI data directory
            split: Which split to load ("evaluation" or "training")
            include_output: Whether to include ground truth outputs
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.include_output = include_output

        # Load test examples
        self.examples = []
        self.task_dir = self.data_dir / split

        if not self.task_dir.exists():
            raise FileNotFoundError(f"Task directory not found: {self.task_dir}")

        for task_file in sorted(self.task_dir.glob("*.json")):
            with open(task_file, "r") as f:
                task_data = json.load(f)

            task_id = task_file.stem

            # Load TEST examples (not train!)
            for test_idx, test_example in enumerate(task_data.get("test", [])):
                example = {
                    'task_id': task_id,
                    'test_idx': test_idx,
                    'input': np.array(test_example["input"], dtype=np.int32),
                }

                if self.include_output and "output" in test_example:
                    example['output'] = np.array(test_example["output"], dtype=np.int32)

                # Also store demonstration pairs for reference
                example['demonstrations'] = [
                    {
                        'input': np.array(demo["input"], dtype=np.int32),
                        'output': np.array(demo["output"], dtype=np.int32)
                    }
                    for demo in task_data.get("train", [])
                ]

                self.examples.append(example)

        print(f"Loaded {len(self.examples)} test examples from {split}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


if __name__ == "__main__":
    # Test the evaluator
    print("Testing ARC Evaluator...")

    # Create evaluator
    evaluator = ARCEvaluator(pass_ks=(1, 2, 5))

    # Test hash function
    grid1 = np.array([[1, 2], [3, 4]])
    grid2 = np.array([[1, 2], [3, 4]])
    grid3 = np.array([[1, 2], [3, 5]])

    print(f"\nHash test:")
    print(f"  Grid1 hash: {grid_hash(grid1)}")
    print(f"  Grid2 hash: {grid_hash(grid2)}")
    print(f"  Grid3 hash: {grid_hash(grid3)}")
    print(f"  Grid1 == Grid2: {grid_hash(grid1) == grid_hash(grid2)}")
    print(f"  Grid1 == Grid3: {grid_hash(grid1) == grid_hash(grid3)}")

    # Test padding removal
    padded = np.array([1, 2, 10, 3, 4, 10, 10, 10, 10])  # 3x3 with padding
    cropped = crop_padding(padded, padding_value=10)
    print(f"\nPadding removal test:")
    print(f"  Original shape: {padded.shape}")
    print(f"  Cropped shape: {cropped.shape}")
    print(f"  Cropped grid:\n{cropped}")

    print("\nEvaluator ready for use!")