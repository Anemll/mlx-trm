import argparse

import mlx.core as mx
import mlx.optimizers as optim

from data.vision import cifar10, mnist
from models import trm
from training.trainer import Trainer

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--dataset",
    type=str,
    default=None,
    choices=["mnist", "cifar10"],
    help="dataset to use (if resuming, defaults to saved dataset)",
)
parser.add_argument("-b", "--batch_size", type=int, default=None, help="batch size (if resuming, defaults to saved batch_size)")
parser.add_argument("-e", "--epochs", type=int, default=15, help="number of epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")
parser.add_argument("--val-freq", type=int, default=None, help="validation frequency (every N epochs, default: max(1, epochs//10))")
parser.add_argument("--save", type=str, default=None, help="checkpoint folder name to save model")
parser.add_argument("--resume", type=str, default=None, help="checkpoint folder to resume training from")


def main(args):
    if args.cpu:
        mx.set_default_device(mx.cpu)
    mx.random.seed(args.seed)

    # If resuming, load metadata to use as defaults
    loaded_metadata = None
    if args.resume:
        from pathlib import Path
        import json
        import sys

        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"Error: Resume directory not found: {args.resume}")
            sys.exit(1)
        if not resume_dir.is_dir():
            print(f"Error: Resume path is not a directory: {args.resume}")
            sys.exit(1)

        history_path = resume_dir / "training_history.json"
        if not history_path.exists():
            print(f"Error: Training history not found in resume directory: {history_path}")
            print(f"Make sure {args.resume} contains a valid checkpoint with training_history.json")
            sys.exit(1)

        with open(history_path, "r") as f:
            history = json.load(f)
            loaded_metadata = {
                "dataset": history.get("dataset"),
                "batch_size": history.get("batch_size"),
                "learning_rate": history.get("learning_rate"),
                "model_config": history.get("model_config"),
            }
            print(f"Loaded training metadata from {history_path}")

    # Apply defaults from loaded metadata if resuming and not overridden
    dataset = args.dataset if args.dataset is not None else (loaded_metadata.get("dataset") if loaded_metadata and loaded_metadata.get("dataset") else "mnist")
    batch_size = args.batch_size if args.batch_size is not None else (loaded_metadata.get("batch_size") if loaded_metadata and loaded_metadata.get("batch_size") else 1024)
    learning_rate = args.lr

    if loaded_metadata and args.dataset is None and loaded_metadata.get("dataset"):
        print(f"  Using saved dataset: {dataset}")
    if loaded_metadata and args.batch_size is None and loaded_metadata.get("batch_size"):
        print(f"  Using saved batch_size: {batch_size}")

    # Load data with the specified or loaded dataset
    if dataset == "mnist":
        train_data, test_data, meta = mnist(batch_size)
    elif dataset == "cifar10":
        train_data, test_data, meta = cifar10(batch_size)
    else:
        raise NotImplementedError(f"{dataset=} is not implemented.")
    n_inputs = next(train_data)["image"].shape[1:]
    train_data.reset()

    # Create model config
    config = trm.ModelConfig(
        in_channels=n_inputs[-1],
        depth=2,
        dim=64,
        heads=4,
        patch_size=(4, 4),
        n_outputs=10,
    )

    # Convert config to dict for saving
    model_config_dict = {
        "in_channels": config.in_channels,
        "depth": config.depth,
        "dim": config.dim,
        "heads": config.heads,
        "patch_size": config.patch_size,
        "n_outputs": config.n_outputs,
        "pool": config.pool,
        "n": config.n,
        "T": config.T,
        "halt_max_steps": config.halt_max_steps,
        "halt_exploration_prob": config.halt_exploration_prob,
        "halt_follow_q": config.halt_follow_q,
    }

    model = trm.Model(config)
    model.summary()

    n_steps = args.epochs * meta["steps_per_epoch"]
    n_linear = n_steps * 0.10
    linear = optim.linear_schedule(0, learning_rate, steps=n_linear)
    cosine = optim.cosine_decay(learning_rate, n_steps - n_linear, 0)
    lr_schedule = optim.join_schedules([linear, cosine], [n_linear])
    optimizer = optim.AdamW(
        learning_rate=lr_schedule, betas=(0.9, 0.999), weight_decay=0.01
    )

    # If resuming without --save, save back to the resume folder
    save_dir = args.save if args.save else args.resume
    if args.resume and not args.save:
        print(f"No --save specified, will save to resume folder: {save_dir}")

    # Default validation frequency: no more than 1/10 of epochs
    val_freq = args.val_freq if args.val_freq is not None else max(1, args.epochs // 10)
    print(f"Validation frequency: every {val_freq} epoch(s)")

    manager = Trainer(model, optimizer, save_dir=save_dir, resume_dir=args.resume,
                     dataset=dataset, model_config=model_config_dict)

    print(f"Starting training for {args.epochs} additional epoch(s)...")
    manager.train(train_data, val=test_data, epochs=args.epochs, val_freq=val_freq,
                 batch_size=batch_size, learning_rate=learning_rate)

    #! plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    lw = 2

    # Create x-axis values that represent actual epoch numbers
    train_epochs = list(range(1, len(manager.train_acc_trace) + 1))

    # Plot training accuracy
    ax.plot(train_epochs, mx.array(manager.train_acc_trace) * 100, label="train", color="r", lw=lw)

    # Plot validation accuracy if we have the data
    if manager.val_acc_trace:
        # Check if we have validation epochs for all validation data
        n_val = len(manager.val_acc_trace)
        n_val_epochs = len(manager.val_epochs)

        if n_val_epochs == n_val:
            # Perfect case: we have epoch info for all validation points
            val_epochs = manager.val_epochs
        elif n_val_epochs > 0 and n_val_epochs < n_val:
            # Partial tracking: some old data without epochs, some new with epochs
            # Estimate the missing epochs and append the tracked ones
            n_missing = n_val - n_val_epochs
            n_train = len(manager.train_acc_trace)
            # Estimate spacing for the missing validation points
            # Assume they were evenly spaced in the earlier epochs
            if n_missing > 0:
                # Find the earliest tracked epoch
                first_tracked = min(manager.val_epochs) if manager.val_epochs else n_train
                # Space the missing points before the first tracked epoch
                spacing = first_tracked / (n_missing + 1)
                estimated_epochs = [int((i + 1) * spacing) for i in range(n_missing)]
            else:
                estimated_epochs = []
            val_epochs = estimated_epochs + manager.val_epochs
            print(f"Warning: Partial validation epoch tracking. Estimated epochs for old data: {estimated_epochs}")
        else:
            # No tracking at all: estimate all epochs
            n_train = len(manager.train_acc_trace)
            if n_val > 0:
                spacing = n_train / n_val
                val_epochs = [int((i + 1) * spacing) for i in range(n_val)]
                val_epochs = [min(e, n_train) for e in val_epochs]
            else:
                val_epochs = []
            print(f"Warning: No validation epoch tracking, estimated all epochs: {val_epochs}")

        ax.plot(val_epochs, mx.array(manager.val_acc_trace) * 100, label="val", color="b", lw=lw)

    # Add text with final accuracy and total epochs
    final_train_acc = manager.train_acc_trace[-1] * 100 if manager.train_acc_trace else 0
    final_val_acc = manager.val_acc_trace[-1] * 100 if manager.val_acc_trace else 0
    total_epochs = len(manager.train_acc_trace)

    info_text = f"Epochs: {total_epochs}\nFinal Train: {final_train_acc:.2f}%\nFinal Val: {final_val_acc:.2f}%"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    fig.tight_layout()

    # Save plot before showing (if save directory specified)
    if save_dir:
        from pathlib import Path
        plot_path = Path(save_dir) / "training_plot.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {plot_path}")

    plt.show()


if __name__ == "__main__":
    main(parser.parse_args())
