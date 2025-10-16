"""Training script for ARC-AGI with multi-optimizer setup matching TinyRecursiveModels.

This version uses MLX's MultiOptimizer to have:
- SGD for puzzle embeddings (lr=1e-2)
- AdamW for main model (lr=1e-4)
Matching the reference implementation more closely.
"""

import argparse
import time
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from data.arc_json import arc_agi_json
from data.arc_augmented import arc_agi_augmented
from data.arc_test import arc_test_dataset
from models.trm_arc import ARCModel, ARCModelConfig
from training.trainer_arc import ARCTrainer

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=15, help="number of epochs")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate for model parameters")
parser.add_argument("--puzzle-emb-lr", type=float, default=1e-2, help="learning rate for puzzle embeddings (default: 1e-2)")
parser.add_argument("--weight-decay", type=float, default=0.1, help="weight decay (default: 0.1)")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")
parser.add_argument("--val-freq", type=int, default=None, help="validation frequency")
parser.add_argument("--save", type=str, default=None, help="checkpoint folder name")
parser.add_argument("--resume", type=str, default=None, help="checkpoint folder to resume from")
parser.add_argument("--resume-from-epoch", type=int, default=None, help="specific epoch number to resume from (default: latest)")
parser.add_argument("--data-dir", type=str, default="data/ARC-AGI/data", help="path to ARC data")
parser.add_argument("--dim", type=int, default=512, help="model hidden dimension")
parser.add_argument("--depth", type=int, default=2, help="number of transformer blocks (default: 2)")
parser.add_argument("--n", type=int, default=6, help="latent recursion steps (default: 6)")
parser.add_argument("--T", type=int, default=3, help="deep recursion steps (default: 3)")
parser.add_argument("--ff-mult", type=int, default=3, help="feedforward expansion multiplier (default: 3)")
parser.add_argument("--no-grad-last-latent-only", action="store_true", help="disable 'grad only on last latent step' (default is enabled)")
parser.add_argument("--halt-max-steps", type=int, default=16, help="maximum adaptive computation steps")
parser.add_argument("--halt-exploration", type=float, default=0.1, help="Q-learning exploration probability")
parser.add_argument("--bf16", action="store_true", help="use bfloat16 precision")
parser.add_argument("--fp16", action="store_true", help="use float16 precision (useful for ANE/CoreML parity)")
parser.add_argument("--puzzle-emb-dim", type=int, default=16, help="dimension for puzzle embeddings")
parser.add_argument("--no-sparse-embeddings", action="store_true", help="disable sparse embeddings (default: sparse embeddings enabled)")
parser.add_argument("--augment", action="store_true", help="enable data augmentations (matching TinyRecursiveModels)")
parser.add_argument("--num-aug", type=int, default=300, help="max augmentations per puzzle (default: 300, paper uses 1000, arcprize analysis shows 300 is near-optimal)")
parser.add_argument("--throughput-samples", type=int, default=None, help="process N training samples then print samples/s and exit (disables validation/checkpointing)")
parser.add_argument("--fixed-act", action="store_true", help="run exactly halt_max_steps iterations without intermediate halt checks")
parser.add_argument("--no-fast-rope", action="store_true", help="disable mx.fast.rope (default: enabled)")
parser.add_argument("--no-mlx-ff", action="store_true", help="disable MLX linear-silu-linear FFN (default: enabled)")

class CachedMultiOptimizer(optim.MultiOptimizer):
    """MultiOptimizer variant that caches parameter-to-optimizer assignments."""

    def __init__(self, optimizers, filters, parameter_tree):
        super().__init__(optimizers, filters)
        self._parameter_tree = parameter_tree
        self._assignment = None

    def _ensure_assignment(self):
        if self._assignment is not None:
            return

        mapping = {}
        for name, param in tree_flatten(self._parameter_tree):
            for idx, predicate in enumerate(self.filters):
                if predicate(name, param):
                    mapping[name] = idx
                    break
        self._assignment = mapping

    def _split_dictionary(self, gradients: dict):
        if len(self.optimizers) == 1:
            return [gradients]

        self._ensure_assignment()

        parts = [[] for _ in range(len(self.optimizers))]
        for name, grad in tree_flatten(gradients):
            idx = self._assignment.get(name)
            if idx is None:
                for candidate_idx, predicate in enumerate(self.filters):
                    if predicate(name, grad):
                        idx = candidate_idx
                        self._assignment[name] = idx
                        break
                else:
                    idx = len(self.optimizers) - 1
                    self._assignment[name] = idx
            parts[idx].append((name, grad))

        return [tree_unflatten(part) if part else {} for part in parts]


def main(args):
    start_time = time.perf_counter()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)

    # Load ARC data
    print(f"Loading ARC data from {args.data_dir}...")
    if args.augment:
        print("Using augmented dataset with puzzle identifiers (matching TinyRecursiveModels)")
        train_data, _, meta = arc_agi_augmented(
            args.batch_size, args.data_dir,
            enable_augmentations=True,
            seed=args.seed,
            max_augmentations_per_puzzle=args.num_aug
        )
        shared_map = train_data.puzzle_id_map
        test_data = arc_test_dataset(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            split="evaluation",
            use_puzzle_ids=True,
            puzzle_id_map=shared_map
        )
        print(f"  Number of puzzle identifiers: {meta['num_puzzle_identifiers']}")
    else:
        print("Using standard dataset (no augmentations)")
        train_data, _, meta = arc_agi_json(args.batch_size, args.data_dir)
        test_data = arc_test_dataset(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            split="evaluation",
            use_puzzle_ids=False
        )
        meta['num_puzzle_identifiers'] = 1000

    meta['n_test_examples'] = test_data.n_examples
    meta['n_test_tasks'] = test_data.n_tasks

    print(f"Dataset info:")
    print(f"  Training examples: {meta['n_train_examples']} (from {meta['n_train_tasks']} tasks)")
    print(f"  Evaluation examples: {meta['n_test_examples']} TEST pairs (from {meta['n_test_tasks']} tasks)")
    print(f"  Steps per epoch: {meta['steps_per_epoch']}")
    print(f"  Vocabulary size: {meta['vocab_size']}")
    print(f"  Max sequence length: {meta['max_seq_len']}")

    # Create model
    config = ARCModelConfig(
        vocab_size=meta['vocab_size'],
        max_seq_len=meta['max_seq_len'],
        depth=args.depth,
        dim=args.dim,
        heads=8,
        n=args.n,  # L_cycles
        T=args.T,  # H_cycles
        ff_mult=args.ff_mult,
        grad_last_latent_only=not args.no_grad_last_latent_only,
        use_fast_rope=not args.no_fast_rope,
        use_mlx_ff=not args.no_mlx_ff,
        halt_max_steps=args.halt_max_steps,
        halt_exploration_prob=args.halt_exploration,
        halt_follow_q=True,
        num_puzzle_identifiers=meta.get('num_puzzle_identifiers', 1000),
        puzzle_emb_ndim=args.puzzle_emb_dim,
        use_sparse_embeddings=not args.no_sparse_embeddings,  # Default: True (enabled)
    )

    print(f"\nModel configuration:")
    print(f"  dim: {args.dim}")
    print(f"  halt_max_steps: {args.halt_max_steps}")
    print(f"  puzzle_emb_ndim: {args.puzzle_emb_dim}")
    print(f"  sparse_embeddings: {not args.no_sparse_embeddings}")
    print(f"  Using MultiOptimizer: SGD for embeddings, AdamW for model")

    model = ARCModel(config)

    if args.fp16:
        print(f"Converting model to float16...")
        model.set_dtype(mx.float16)
    elif args.bf16:
        print(f"Converting model to bfloat16...")
        model.set_dtype(mx.bfloat16)

    model.summary()

    # Learning rate schedules
    n_steps = args.epochs * meta["steps_per_epoch"]
    n_warmup = min(2000, max(10, int(n_steps * 0.1)))

    print(f"\nLearning rate schedule:")
    print(f"  Total steps: {n_steps}")
    print(f"  Warmup steps: {n_warmup}")

    # Schedule for model parameters (AdamW)
    model_linear = optim.linear_schedule(0, args.lr, steps=n_warmup)
    model_cosine = optim.cosine_decay(args.lr, n_steps - n_warmup, 0)
    model_lr_schedule = optim.join_schedules([model_linear, model_cosine], [n_warmup])

    # Schedule for puzzle embeddings (SGD with higher LR)
    puzzle_linear = optim.linear_schedule(0, args.puzzle_emb_lr, steps=n_warmup)
    puzzle_cosine = optim.cosine_decay(args.puzzle_emb_lr, n_steps - n_warmup, 0)
    puzzle_lr_schedule = optim.join_schedules([puzzle_linear, puzzle_cosine], [n_warmup])

    # Create MultiOptimizer matching reference:
    # - SGD for puzzle embeddings (lr=1e-2)
    # - AdamW for everything else (lr=1e-4)

    # Filter function for puzzle embeddings
    trainable_params = model.trainable_parameters()

    puzzle_param_names = {
        name
        for name, _ in tree_flatten(trainable_params)
        if "puzzle_emb" in name or "puzzle_proj" in name
    }

    def is_puzzle_embedding(name, _):
        return name in puzzle_param_names

    # Create optimizers
    puzzle_optimizer = optim.SGD(
        learning_rate=puzzle_lr_schedule,
        momentum=0.0,  # Pure SGD like reference
        weight_decay=args.weight_decay
    )

    model_optimizer = optim.AdamW(
        learning_rate=model_lr_schedule,
        betas=[0.9, 0.95],
        weight_decay=args.weight_decay
    )

    # Combine with MultiOptimizer
    multi_optimizer = CachedMultiOptimizer(
        optimizers=[puzzle_optimizer, model_optimizer],
        filters=[is_puzzle_embedding],
        parameter_tree=trainable_params,
    )

    print(f"\nOptimizer setup:")
    print(f"  Puzzle embeddings: SGD (lr={args.puzzle_emb_lr}, weight_decay={args.weight_decay})")
    print(f"  Model parameters: AdamW (lr={args.lr}, betas=[0.9, 0.95], weight_decay={args.weight_decay})")

    # Trainer
    save_directory = args.save if args.save else args.resume
    manager = ARCTrainer(model, multi_optimizer, save_dir=save_directory, resume_dir=args.resume,
                        resume_from_epoch=args.resume_from_epoch)

    # Validation frequency
    val_freq = args.val_freq if args.val_freq is not None else max(1, args.epochs // 10)
    print(f"Validation frequency: every {val_freq} epoch(s) on {meta['n_test_examples']} test examples")

    # Throughput-only mode: process a fixed number of samples and exit
    if args.throughput_samples is not None and args.throughput_samples > 0:
        print(f"\nThroughput mode: processing {args.throughput_samples} samples (no validation/checkpointing)...")
        manager.train(
            train_data,
            val=None,
            epochs=1,
            val_freq=0,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_train_samples=args.throughput_samples,
            skip_halt_check=args.fixed_act,
        )
        return

    print(f"\nStarting training for {args.epochs} epoch(s)...")
    manager.train(
        train_data,
        val=test_data,
        epochs=args.epochs,
        val_freq=val_freq,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        skip_halt_check=args.fixed_act,
    )

    total_time = time.perf_counter() - start_time
    print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")

    # Plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    lw = 2

    train_epochs = list(range(1, len(manager.train_acc_trace) + 1))
    ax.plot(train_epochs, mx.array(manager.train_acc_trace) * 100, label="train", color="r", lw=lw)

    if manager.val_acc_trace:
        if manager.val_epochs and len(manager.val_epochs) == len(manager.val_acc_trace):
            val_epochs = manager.val_epochs
        else:
            val_epochs = list(range(1, len(manager.val_acc_trace) + 1))
        ax.plot(val_epochs, mx.array(manager.val_acc_trace) * 100, label="val", color="b", lw=lw)

    if manager.val_exact_match_trace:
        if manager.val_epochs and len(manager.val_epochs) == len(manager.val_exact_match_trace):
            val_epochs = manager.val_epochs
        else:
            val_epochs = list(range(1, len(manager.val_exact_match_trace) + 1))
        ax.plot(val_epochs, mx.array(manager.val_exact_match_trace) * 100, label="exact match", color="g", lw=lw, linestyle="--")

    # Add stats
    final_train_acc = manager.train_acc_trace[-1] * 100 if manager.train_acc_trace else 0
    final_val_acc = manager.val_acc_trace[-1] * 100 if manager.val_acc_trace else 0
    total_epochs = len(manager.train_acc_trace)

    # Use actual counts instead of fractions
    if manager.val_exact_match_count_trace:
        exact_match_count = manager.val_exact_match_count_trace[-1]
        val_set_size = manager.val_total_samples
    else:
        # Fallback to old calculation
        final_exact_match = manager.val_exact_match_trace[-1] if manager.val_exact_match_trace else 0
        val_set_size = test_data.n_examples
        exact_match_count = int(final_exact_match * val_set_size)

    info_text = f"Epochs: {total_epochs}\nFinal Train: {final_train_acc:.2f}%\nFinal Val: {final_val_acc:.2f}%\nExact Match: {exact_match_count}/{val_set_size} pairs"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ARC-AGI Training (Multi-Optimizer)")
    fig.tight_layout()

    if save_directory:
        from pathlib import Path
        plot_path = Path(save_directory) / "training_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {plot_path}")

    plt.show()


if __name__ == "__main__":
    main(parser.parse_args())
