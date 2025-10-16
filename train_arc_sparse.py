"""Training script for ARC-AGI with sparse embeddings optimization.

This version uses sparse embeddings matching TinyRecursiveModels:
- Only updates puzzle embeddings that are actually used in each batch
- Should be much faster than updating all 1000 embeddings every step
"""

import argparse
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from data.arc_augmented import arc_agi_augmented
from data.arc_test import arc_test_dataset
from models.trm_arc import ARCModel, ARCModelConfig
from models.sparse_embedding import create_sparse_optimizer
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
parser.add_argument("--data-dir", type=str, default="data/ARC-AGI/data", help="path to ARC data")
parser.add_argument("--dim", type=int, default=512, help="model hidden dimension")
parser.add_argument("--halt-max-steps", type=int, default=16, help="maximum adaptive computation steps")
parser.add_argument("--halt-exploration", type=float, default=0.1, help="Q-learning exploration probability")
parser.add_argument("--bf16", action="store_true", help="use bfloat16 precision")
parser.add_argument("--puzzle-emb-dim", type=int, default=16, help="dimension for puzzle embeddings")


def custom_trainer_with_sparse_update(model, learning_rate, puzzle_emb_lr, weight_decay):
    """Create a custom trainer that uses sparse updates for embeddings."""
    from mlx.optimizers import AdamW
    from functools import partial
    import mlx.nn as nn
    from mlx.optimizers import clip_grad_norm

    # Create main optimizer for non-embedding parameters
    main_optimizer = AdamW(
        learning_rate=learning_rate,
        betas=[0.9, 0.95],
        weight_decay=weight_decay
    )

    # Track state
    state = [model.state, main_optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def custom_step(model, carry, batch):
        """Custom training step with sparse embedding updates."""
        # Forward and backward pass
        def eval_fn(carry, batch):
            # This would be the eval_fn from trainer
            carry, outputs = model(carry, batch)

            # Compute loss (simplified - should match trainer)
            target = batch["output_tokens"]
            logits = outputs["logits"]

            is_not_padding_flat = (target != 10).reshape(-1).astype(mx.float32)
            lm_loss = nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                weights=is_not_padding_flat,
                reduction="mean"
            )

            # Q-halt loss
            pred_tokens = mx.argmax(logits, axis=-1)
            is_not_padding = (target != 10)
            correct_tokens = (pred_tokens == target) & is_not_padding
            seq_is_correct = (correct_tokens.sum(axis=1) == is_not_padding.sum(axis=1)).astype(mx.float32)

            q_halt_loss = nn.losses.binary_cross_entropy(
                outputs["q_halt_logits"],
                seq_is_correct,
                with_logits=True,
                reduction="mean"
            )

            total_loss = (lm_loss + q_halt_loss) / batch["output_tokens"].shape[0]

            token_accuracy = correct_tokens.sum() / is_not_padding.sum()

            stats = {
                "lm_loss": lm_loss,
                "q_halt_loss": q_halt_loss,
                "q_halt_prob_mean": mx.sigmoid(outputs["q_halt_logits"]).mean(),
                "frac_halted": carry["halted"].mean(),
                "avg_steps": carry["steps"].mean(),
                "logits": logits,
            }

            return total_loss, carry, token_accuracy, stats

        # Get gradients
        train_step_fn = nn.value_and_grad(model, eval_fn)
        (loss, carry, accuracy, stats), grads = train_step_fn(carry, batch)

        # Clip gradients
        grads, _ = clip_grad_norm(grads, max_norm=1.0)

        # Separate gradients for sparse and dense parameters
        sparse_grads = {}
        dense_grads = {}

        for name, grad in grads.items():
            if 'puzzle_emb' in name:
                # For sparse embeddings, only update used indices
                if hasattr(model.embed.puzzle_emb, 'current_batch_indices'):
                    indices = model.embed.puzzle_emb.current_batch_indices
                    if indices is not None:
                        # Create sparse gradient
                        sparse_grad = mx.zeros_like(grad)
                        unique_indices = mx.unique(indices.flatten())
                        sparse_grad[unique_indices] = grad[unique_indices]
                        sparse_grads[name] = sparse_grad
                    else:
                        sparse_grads[name] = grad
                else:
                    sparse_grads[name] = grad
            elif 'puzzle_proj' in name:
                # Puzzle projection uses SGD with higher LR
                sparse_grads[name] = grad
            else:
                dense_grads[name] = grad

        # Manual SGD update for sparse parameters (puzzle embeddings)
        for name, grad in sparse_grads.items():
            param = model.parameters()[name]
            # SGD with weight decay
            if weight_decay > 0:
                param = param * (1 - puzzle_emb_lr * weight_decay)
            param = param - puzzle_emb_lr * grad
            model.parameters()[name] = param

        # AdamW update for dense parameters
        if dense_grads:
            main_optimizer.update(model, dense_grads)

        return loss, carry, accuracy, stats

    return custom_step, state


def main(args):
    start_time = time.perf_counter()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)

    # Load augmented ARC data with puzzle IDs
    print(f"Loading augmented ARC data from {args.data_dir}...")
    print("Using SPARSE embeddings for puzzle identifiers (matching TinyRecursiveModels)")
    train_data, _, meta = arc_agi_augmented(
        args.batch_size, args.data_dir,
        enable_augmentations=True,
        seed=args.seed
    )
    shared_map = train_data.puzzle_id_map
    test_data = arc_test_dataset(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        split="evaluation",
        use_puzzle_ids=True,
        puzzle_id_map=shared_map
    )

    meta['n_test_examples'] = test_data.n_examples
    meta['n_test_tasks'] = test_data.n_tasks

    print(f"Dataset info:")
    print(f"  Training examples: {meta['n_train_examples']} (augmented)")
    print(f"  Evaluation examples: {meta['n_test_examples']} TEST pairs")
    print(f"  Puzzle identifiers: {meta['num_puzzle_identifiers']}")

    # Create model with sparse embeddings enabled
    config = ARCModelConfig(
        vocab_size=meta['vocab_size'],
        max_seq_len=meta['max_seq_len'],
        depth=2,
        dim=args.dim,
        heads=8,
        n=6,  # L_cycles
        T=3,  # H_cycles
        halt_max_steps=args.halt_max_steps,
        halt_exploration_prob=args.halt_exploration,
        halt_follow_q=True,
        num_puzzle_identifiers=meta.get('num_puzzle_identifiers', 1000),
        puzzle_emb_ndim=args.puzzle_emb_dim,
        use_sparse_embeddings=True,  # Enable sparse embeddings
    )

    print(f"\nModel configuration:")
    print(f"  dim: {args.dim}")
    print(f"  halt_max_steps: {args.halt_max_steps}")
    print(f"  puzzle_emb_ndim: {args.puzzle_emb_dim}")
    print(f"  Using SPARSE embeddings (only updates ~3-10 embeddings per batch)")

    model = ARCModel(config)

    if args.bf16:
        print(f"Converting model to bfloat16...")
        model.set_dtype(mx.bfloat16)

    model.summary()

    # Learning rate schedules
    n_steps = args.epochs * meta["steps_per_epoch"]
    n_warmup = min(2000, max(10, int(n_steps * 0.1)))

    print(f"\nLearning rate schedule:")
    print(f"  Total steps: {n_steps}")
    print(f"  Warmup steps: {n_warmup}")

    # Create learning rate schedules
    model_linear = optim.linear_schedule(0, args.lr, steps=n_warmup)
    model_cosine = optim.cosine_decay(args.lr, n_steps - n_warmup, 0)
    model_lr_schedule = optim.join_schedules([model_linear, model_cosine], [n_warmup])

    # Create custom sparse optimizer matching reference
    optimizer = create_sparse_optimizer(
        model,
        learning_rate=model_lr_schedule,
        puzzle_emb_lr=args.puzzle_emb_lr,
        weight_decay=args.weight_decay
    )

    print(f"\nOptimizer setup:")
    print(f"  Puzzle embeddings: Sparse SGD (lr={args.puzzle_emb_lr}, weight_decay={args.weight_decay})")
    print(f"  Model parameters: AdamW (lr={args.lr}, betas=[0.9, 0.95], weight_decay={args.weight_decay})")
    print(f"  SPARSE UPDATES: Only updating used embeddings (3-10 per batch)")

    # Trainer
    save_directory = args.save if args.save else args.resume

    # Create a wrapper optimizer that MLX Trainer can use
    class SparseOptimizerWrapper:
        def __init__(self, update_fn):
            self.update_fn = update_fn
            self.state = {}

        def update(self, model, gradients):
            self.update_fn(model, gradients)

    wrapped_optimizer = SparseOptimizerWrapper(optimizer)
    manager = ARCTrainer(model, wrapped_optimizer, save_dir=save_directory, resume_dir=args.resume)

    # Validation frequency
    val_freq = args.val_freq if args.val_freq is not None else max(1, args.epochs // 10)
    print(f"Validation frequency: every {val_freq} epoch(s) on {meta['n_test_examples']} test examples")

    print(f"\nStarting training for {args.epochs} epoch(s)...")
    print("Expected speedup: ~10x faster than dense embeddings")
    manager.train(
        train_data,
        val=test_data,
        epochs=args.epochs,
        val_freq=val_freq,
        batch_size=args.batch_size,
        learning_rate=args.lr,
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

    final_exact_match = manager.val_exact_match_trace[-1] if manager.val_exact_match_trace else 0
    val_set_size = test_data.n_examples
    exact_match_count = int(final_exact_match * val_set_size)

    info_text = f"Epochs: {total_epochs}\nFinal Train: {final_train_acc:.2f}%\nFinal Val: {final_val_acc:.2f}%\nExact Match: {exact_match_count}/{val_set_size}\nSparse Embeddings"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("ARC-AGI Training (Sparse Embeddings)")
    fig.tight_layout()

    if save_directory:
        from pathlib import Path
        plot_path = Path(save_directory) / "training_plot_sparse.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {plot_path}")

    plt.show()


if __name__ == "__main__":
    main(parser.parse_args())
