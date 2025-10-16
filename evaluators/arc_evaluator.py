"""ARC-specific evaluation with puzzle identifier tracking.

This implementation follows the evaluation strategy from:
https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/evaluators/arc.py

Key features:
- Tracks puzzle identifiers during evaluation
- Groups predictions by puzzle ID for analysis
- Handles both seen (training) and unseen (test) puzzles
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import mlx.core as mx


class ARCEvaluator:
    """Evaluator for ARC tasks with puzzle ID tracking."""

    def __init__(self, model, data_dir: str = "data/ARC-AGI/data"):
        """Initialize ARC evaluator.

        Args:
            model: The trained ARC model
            data_dir: Path to ARC-AGI data directory
        """
        self.model = model
        self.data_dir = Path(data_dir)

        # Load task metadata for mapping puzzle IDs to names
        self.puzzle_id_to_name = {}
        self.name_to_puzzle_id = {}
        self._load_puzzle_mappings()

    def _load_puzzle_mappings(self):
        """Load mappings between puzzle IDs and task names."""
        # This would be populated from the dataset's puzzle_id_map
        # For now, we'll build it dynamically during evaluation
        pass

    def evaluate_dataset(self, dataset, max_batches: Optional[int] = None) -> Dict:
        """Evaluate model on a dataset, tracking puzzle-specific performance.

        Args:
            dataset: The dataset to evaluate (should include puzzle_ids)
            max_batches: Maximum number of batches to evaluate (for debugging)

        Returns:
            Dictionary with evaluation metrics including per-puzzle results
        """
        self.model.eval()
        dataset.reset()

        # Metrics accumulators
        total_loss = 0.0
        total_token_accuracy = 0.0
        n_batches = 0

        # Puzzle-specific tracking
        puzzle_predictions = defaultdict(list)  # puzzle_id -> list of (pred, target, is_correct)
        puzzle_exact_matches = defaultdict(int)  # puzzle_id -> count of exact matches
        puzzle_total_examples = defaultdict(int)  # puzzle_id -> total examples

        for batch_idx, batch in enumerate(dataset):
            if max_batches and batch_idx >= max_batches:
                break

            # Convert to MLX arrays
            mlx_batch = {k: mx.array(v) for k, v in batch.items()}

            # Get puzzle IDs for this batch
            puzzle_ids = batch.get("puzzle_ids", None)
            if puzzle_ids is None:
                # If no puzzle IDs, create dummy ones
                puzzle_ids = np.zeros(batch["input_tokens"].shape[0], dtype=np.int32)

            # Initialize carry state
            carry = self.model.initial_carry(mlx_batch)

            # Run inference (with adaptive computation)
            for _ in range(self.model.config.halt_max_steps):
                carry, outputs = self.model(carry, mlx_batch)
                if carry["halted"].all():
                    break

            # Get predictions and targets
            logits = outputs["logits"]
            target = mlx_batch["output_tokens"]
            pred_tokens = mx.argmax(logits, axis=-1)

            # Calculate metrics
            is_not_padding = (target != 10)  # 10 is padding token
            correct_tokens = (pred_tokens == target) & is_not_padding

            # Token-level accuracy
            batch_token_acc = correct_tokens.sum() / is_not_padding.sum()
            total_token_accuracy += float(batch_token_acc)

            # Sequence-level exact matches (per example in batch)
            for i in range(len(puzzle_ids)):
                puzzle_id = int(puzzle_ids[i])

                # Check if entire sequence matches
                example_correct = correct_tokens[i]
                example_not_padding = is_not_padding[i]
                is_exact_match = (example_correct.sum() == example_not_padding.sum())

                # Store prediction info
                puzzle_predictions[puzzle_id].append({
                    "pred_tokens": pred_tokens[i],
                    "target_tokens": target[i],
                    "is_exact_match": bool(is_exact_match),
                    "token_accuracy": float(example_correct.sum() / example_not_padding.sum())
                })

                puzzle_total_examples[puzzle_id] += 1
                if is_exact_match:
                    puzzle_exact_matches[puzzle_id] += 1

            n_batches += 1

        # Calculate overall metrics
        avg_token_accuracy = total_token_accuracy / n_batches if n_batches > 0 else 0.0

        # Calculate per-puzzle metrics
        puzzle_metrics = {}
        total_exact_matches = 0
        total_examples = 0

        for puzzle_id in puzzle_predictions:
            n_exact = puzzle_exact_matches[puzzle_id]
            n_total = puzzle_total_examples[puzzle_id]

            puzzle_metrics[puzzle_id] = {
                "exact_matches": n_exact,
                "total_examples": n_total,
                "exact_match_rate": n_exact / n_total if n_total > 0 else 0.0,
                "predictions": puzzle_predictions[puzzle_id][:5]  # Store first 5 for inspection
            }

            total_exact_matches += n_exact
            total_examples += n_total

        # Overall exact match rate
        overall_exact_match_rate = total_exact_matches / total_examples if total_examples > 0 else 0.0

        return {
            "token_accuracy": avg_token_accuracy,
            "exact_match_rate": overall_exact_match_rate,
            "total_exact_matches": total_exact_matches,
            "total_examples": total_examples,
            "n_unique_puzzles": len(puzzle_predictions),
            "puzzle_metrics": puzzle_metrics,
        }

    def evaluate_test_puzzles(self, test_dir: Optional[str] = None) -> Dict:
        """Evaluate on test puzzles specifically.

        This method loads test puzzles directly from JSON files and evaluates
        them individually, similar to the reference implementation.

        Args:
            test_dir: Directory containing test JSON files

        Returns:
            Dictionary with test evaluation results
        """
        if test_dir is None:
            test_dir = self.data_dir / "evaluation"

        test_dir = Path(test_dir)
        test_files = sorted(list(test_dir.glob("*.json")))

        if not test_files:
            raise ValueError(f"No test files found in {test_dir}")

        results = {}
        total_correct = 0
        total_tasks = 0

        for task_file in test_files:
            task_name = task_file.stem

            with open(task_file, "r") as f:
                task_data = json.load(f)

            # Evaluate test examples
            test_examples = task_data.get("test", [])
            if not test_examples:
                print(f"Warning: No test examples in {task_name}")
                continue

            task_correct = 0
            for test_example in test_examples:
                # Convert test example to model input format
                # This would need the actual grid-to-sequence conversion
                # For now, this is a placeholder
                is_correct = self._evaluate_single_example(test_example, task_name)
                if is_correct:
                    task_correct += 1

            results[task_name] = {
                "correct": task_correct,
                "total": len(test_examples),
                "accuracy": task_correct / len(test_examples) if test_examples else 0
            }

            if task_correct == len(test_examples):
                total_correct += 1
            total_tasks += 1

        return {
            "tasks_solved": total_correct,
            "total_tasks": total_tasks,
            "solve_rate": total_correct / total_tasks if total_tasks > 0 else 0,
            "task_results": results
        }

    def _evaluate_single_example(self, example: dict, task_name: str) -> bool:
        """Evaluate a single test example.

        Args:
            example: Dictionary with 'input' and 'output' grids
            task_name: Name of the task (for puzzle ID lookup)

        Returns:
            True if prediction matches target exactly
        """
        # This is a simplified placeholder
        # In practice, this would:
        # 1. Convert grids to sequences
        # 2. Get puzzle ID from mapping
        # 3. Run model inference
        # 4. Compare predictions
        return False

    def print_summary(self, results: Dict):
        """Print a formatted summary of evaluation results.

        Args:
            results: Dictionary from evaluate_dataset or evaluate_test_puzzles
        """
        print("\n" + "="*60)
        print("ARC EVALUATION SUMMARY")
        print("="*60)

        if "token_accuracy" in results:
            print(f"Token Accuracy: {results['token_accuracy']:.2%}")

        if "exact_match_rate" in results:
            print(f"Exact Match Rate: {results['exact_match_rate']:.2%}")
            print(f"Exact Matches: {results['total_exact_matches']} / {results['total_examples']}")

        if "n_unique_puzzles" in results:
            print(f"Unique Puzzles: {results['n_unique_puzzles']}")

        if "tasks_solved" in results:
            print(f"Tasks Solved: {results['tasks_solved']} / {results['total_tasks']}")
            print(f"Solve Rate: {results['solve_rate']:.2%}")

        # Print worst-performing puzzles
        if "puzzle_metrics" in results and results["puzzle_metrics"]:
            print("\nWorst-Performing Puzzles:")
            puzzle_scores = [
                (pid, metrics["exact_match_rate"])
                for pid, metrics in results["puzzle_metrics"].items()
            ]
            puzzle_scores.sort(key=lambda x: x[1])

            for i, (puzzle_id, score) in enumerate(puzzle_scores[:5]):
                metrics = results["puzzle_metrics"][puzzle_id]
                print(f"  {i+1}. Puzzle {puzzle_id}: {score:.1%} "
                      f"({metrics['exact_matches']}/{metrics['total_examples']})")

        print("="*60 + "\n")


def evaluate_arc_model(
    model,
    dataset,
    data_dir: str = "data/ARC-AGI/data",
    max_batches: Optional[int] = None
) -> Dict:
    """Convenience function to evaluate an ARC model.

    Args:
        model: Trained ARC model
        dataset: Dataset to evaluate
        data_dir: Path to ARC data directory
        max_batches: Maximum batches to evaluate

    Returns:
        Evaluation results dictionary
    """
    evaluator = ARCEvaluator(model, data_dir)
    results = evaluator.evaluate_dataset(dataset, max_batches)
    evaluator.print_summary(results)
    return results


if __name__ == "__main__":
    # Test the evaluator
    print("ARC Evaluator module loaded successfully")
    print("Use evaluate_arc_model() to evaluate a trained model")