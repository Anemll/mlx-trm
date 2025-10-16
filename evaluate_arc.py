#!/usr/bin/env python3
"""Evaluate trained model on ARC test examples with proper exact-match scoring.

This script implements the same evaluation protocol as TinyRecursiveModels:
- Loads actual test examples from task["test"]
- Uses exact grid matching via hashing
- Reports pass@K metrics
"""

import argparse
from pathlib import Path
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tqdm import tqdm

from models.trm_arc import ARCModel, ARCModelConfig
from evaluators.arc import ARCEvaluator, crop_padding
from data.arc_augmented import build_puzzle_id_map
from data.arc_json import ARCJSONDataset
from data.arc_test import ARCTestDataset


def evaluate_model(
    model_path: str,
    data_dir: str = "data/ARC-AGI/data",
    pass_ks: tuple = (1, 2, 5),
    batch_size: int = 32,
    use_bf16: bool = False,
    verbose: bool = True,
    compute_token_accuracy: bool = False,
    use_training_examples: bool = False,
    dim: int = 512,
    depth: int = 2,
    heads: int = 8,
    n_cycles: int = 6,
    t_cycles: int = 3,
    halt_max_steps: int = 16,
    puzzle_emb_dim: int = 16,
):
    """Evaluate a trained model on ARC test examples.

    Args:
        model_path: Path to saved model checkpoint
        data_dir: Path to ARC data directory
        pass_ks: K values for pass@K metrics
        batch_size: Batch size for inference
        use_bf16: Whether to use bfloat16 precision
        verbose: Whether to print detailed results
    """
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint_path = Path(model_path)

    if not checkpoint_path.exists():
        if (checkpoint_path / "best_model.safetensors").exists():
            checkpoint_path = checkpoint_path / "best_model.safetensors"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

    # Create model configuration
    config = ARCModelConfig(
        vocab_size=12,
        max_seq_len=900,
        depth=depth,
        dim=dim,
        heads=heads,
        n=n_cycles,
        T=t_cycles,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=0.1,
        halt_follow_q=True,
        puzzle_emb_ndim=puzzle_emb_dim,
    )

    model = ARCModel(config)

    if use_bf16:
        print("Converting to bfloat16...")
        model.set_dtype(mx.bfloat16)

    # Load weights
    model.load_weights(str(checkpoint_path))
    model.eval()

    # Count parameters (flatten nested dicts)
    def count_params(params):
        total = 0
        for _, value in tree_flatten(params):
            if hasattr(value, "size"):
                total += int(value.size)
            else:
                value_array = mx.array(value)
                total += int(value_array.size)
        return total

    total_params = count_params(model.parameters())
    print(f"Model loaded. Parameters: {total_params:,}")

    puzzle_id_map = build_puzzle_id_map(data_dir)

    if use_training_examples:
        dataset_label = "training demonstrations"
        dataset = ARCJSONDataset(data_dir=data_dir, split="training", batch_size=batch_size)
        dataset_examples = dataset.examples
        total_examples = len(dataset_examples)
        evaluator = None
    else:
        dataset_label = "evaluation test puzzles"
        evaluator = ARCEvaluator(data_dir=data_dir, pass_ks=pass_ks, verbose=verbose)
        dataset = ARCTestDataset(data_dir=data_dir, split="evaluation", puzzle_id_map=puzzle_id_map)
        dataset_examples = dataset.examples
        total_examples = len(dataset_examples)

    print(f"\nEvaluating on {total_examples} {dataset_label} with batch size {batch_size}...")

    token_correct = 0
    token_total = 0
    exact_matches = 0
    confidence_sum = 0.0
    confidence_count = 0

    progress_bar = tqdm(total=total_examples, desc="Evaluating", disable=not verbose, unit="example")
    hash_exact = 0
    token_perfect_matches = 0

    # Process test examples in batches for better throughput
    for start_idx in range(0, total_examples, batch_size):
        end_idx = min(start_idx + batch_size, total_examples)
        batch_examples = [dataset_examples[idx] for idx in range(start_idx, end_idx)]

        if use_training_examples:
            input_sequences = np.stack(
                [example['input_tokens'] for example in batch_examples],
                axis=0,
            )
            target_sequences = [example['output_tokens'] for example in batch_examples]
        else:
            input_sequences = np.stack(
                [example['input_tokens'] for example in batch_examples],
                axis=0,
            )
            target_sequences = [example['output_tokens'] for example in batch_examples]

        try:
            puzzle_ids = [puzzle_id_map[example['task_id']] for example in batch_examples]
        except KeyError as exc:
            missing_id = exc.args[0]
            raise KeyError(f"Puzzle ID for task '{missing_id}' not found in mapping") from exc

        batch = {
            "input_tokens": mx.array(input_sequences, dtype=mx.int32),
            "output_tokens": mx.zeros((len(batch_examples), 900), dtype=mx.int32),
            "puzzle_ids": mx.array(puzzle_ids, dtype=mx.int32),
        }

        carry = model.initial_carry(batch)

        max_iterations = model.config.halt_max_steps
        for _ in range(max_iterations):
            carry, outputs = model(carry, batch)
            if mx.all(carry["halted"]).item():
                break

        logits = outputs["logits"]
        pred_tokens = mx.argmax(logits, axis=-1)
        confidences = mx.sigmoid(outputs["q_halt_logits"])

        preds_np = np.array(pred_tokens)
        confidences_np = np.array(confidences)

        for idx, example in enumerate(batch_examples):
            pred_seq = preds_np[idx]

            target_seq = target_sequences[idx] if idx < len(target_sequences) else None
            target_mask = None

            if not use_training_examples:
                pred_grid = crop_padding(pred_seq, padding_value=10)
                pred_grid = pred_grid.astype(np.int32, copy=False)
                evaluator.add_prediction(
                    task_id=example['task_id'],
                    test_idx=example.get('test_idx', idx),
                    pred_grid=pred_grid,
                    confidence=float(confidences_np[idx])
                )

            if target_seq is not None:
                target_mask = target_seq != 10
                pred_mask = pred_seq != 10

                if compute_token_accuracy and target_mask.any():
                    correct_tokens = np.sum((pred_seq == target_seq) & target_mask)
                    token_correct += int(correct_tokens)
                    token_total += int(np.sum(target_mask))

                token_perfect = target_mask.any() and np.all((pred_seq == target_seq)[target_mask])
                if token_perfect:
                    token_perfect_matches += 1

                if token_perfect and np.array_equal(pred_mask, target_mask):
                    if not use_training_examples:
                        target_grid = crop_padding(target_seq, padding_value=10).astype(np.int32, copy=False)
                        if np.array_equal(pred_grid, target_grid):
                            hash_exact += 1
                    exact_matches += 1

            confidence_sum += float(confidences_np[idx])
            confidence_count += 1

        progress_bar.update(len(batch_examples))
        if verbose:
            postfix = {}
            if confidence_count:
                postfix["avg_conf"] = f"{confidence_sum / confidence_count:.2f}"
            if compute_token_accuracy and token_total > 0:
                postfix["token_acc"] = f"{token_correct / token_total:.2%}"
            if progress_bar.n:
                postfix["exact"] = f"{exact_matches}/{progress_bar.n}"
                postfix["token_perfect"] = f"{token_perfect_matches}/{progress_bar.n}"
            if postfix:
                progress_bar.set_postfix(postfix)

    progress_bar.close()

    # Evaluate
    print("\n" + "="*50)
    if use_training_examples:
        exact_rate = exact_matches / total_examples if total_examples else 0.0
        results = {f"pass@{k}": exact_rate for k in pass_ks}
    else:
        results = evaluator.evaluate()
        if verbose:
            print(f"Token-perfect matches (ignoring extras): {token_perfect_matches}/{total_examples}")
            print(f"Strict (mask) exact matches: {exact_matches}/{total_examples}")
            print(f"Hash-based exact matches: {hash_exact}/{total_examples}")

    if compute_token_accuracy and token_total > 0:
        results["token_accuracy"] = token_correct / token_total

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC model with exact matching")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint or directory")
    parser.add_argument("--data-dir", type=str, default="data/ARC-AGI/data",
                        help="Path to ARC data directory")
    parser.add_argument("--pass-k", type=int, nargs="+", default=[1, 2, 5],
                        help="K values for pass@K metrics")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bfloat16 precision")
    parser.add_argument("--token-accuracy", action="store_true",
                        help="Compute and report token-level accuracy")
    parser.add_argument("--use-training", action="store_true",
                        help="Evaluate on training demonstrations instead of test puzzles")
    parser.add_argument("--dim", type=int, default=512,
                        help="Model hidden dimension (default: 512)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Number of transformer blocks (default: 2)")
    parser.add_argument("--heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--n-cycles", type=int, default=6,
                        help="Latent recursion steps n (default: 6)")
    parser.add_argument("--t-cycles", type=int, default=3,
                        help="Deep recursion steps T (default: 3)")
    parser.add_argument("--halt-max-steps", type=int, default=16,
                        help="Maximum ACT steps (default: 16)")
    parser.add_argument("--puzzle-emb-dim", type=int, default=16,
                        help="Puzzle embedding dimension (default: 16)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        pass_ks=tuple(args.pass_k),
        batch_size=args.batch_size,
        use_bf16=args.bf16,
        verbose=not args.quiet,
        compute_token_accuracy=args.token_accuracy,
        use_training_examples=args.use_training,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        n_cycles=args.n_cycles,
        t_cycles=args.t_cycles,
        halt_max_steps=args.halt_max_steps,
        puzzle_emb_dim=args.puzzle_emb_dim,
    )

    # Print summary
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print("="*50)
    split_label = "training demonstrations" if args.use_training else "evaluation test puzzles"
    print(f"Split: {split_label}")
    for metric, value in results.items():
        print(f"{metric}: {value:.2%}")


if __name__ == "__main__":
    main()
