from contextlib import contextmanager
from functools import partial
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Optimizer, clip_grad_norm
from mlx.utils import tree_map
from tqdm import tqdm


def ema_update(ema_params, model, alpha=0.95):
    return tree_map(
        lambda a, b: a * alpha + (1 - alpha) * b, ema_params, model.parameters()
    )


@contextmanager
def use_ema(model, ema_params):
    orig = model.parameters()
    model.update(ema_params)
    try:
        yield
    finally:
        model.update(orig)


class Trainer:
    def __init__(self, model: nn.Module, optimizer: Optimizer, save_dir: str = None, resume_dir: str = None,
                 dataset: str = None, model_config: dict = None):
        self.model = model
        self.optimizer = optimizer
        self.save_dir = Path(save_dir) if save_dir else None
        self.resume_dir = Path(resume_dir) if resume_dir else None

        self.ema_params = model.parameters()

        self.train_error_trace: list[float] = []
        self.train_acc_trace: list[float] = []
        self.val_error_trace: list[float] = []
        self.val_acc_trace: list[float] = []
        self.val_epochs: list[int] = []  # Track which epochs had validation
        self.start_epoch = 0

        # Store metadata for saving
        self.dataset = dataset
        self.model_config = model_config

        # Load checkpoint if resuming
        if self.resume_dir:
            loaded_metadata = self.load_checkpoint()
            # Return loaded metadata so it can be used as defaults
            self.loaded_metadata = loaded_metadata

        # Create save directory if specified
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {self.save_dir}")

    def eval_fn(self, carry, batch):
        carry, outputs = self.model(carry, batch)
        label = carry["current_data"]["label"]
        pred = outputs["logits"]
        is_correct = mx.argmax(pred, axis=1) == label

        ce = nn.losses.cross_entropy(pred, label, reduction="mean")
        bce = nn.losses.binary_cross_entropy(
            outputs["q_halt_logits"], is_correct, with_logits=True, reduction="mean"
        )
        loss = ce + 0.5 * bce

        stats = {
            "q_prob_mean": mx.sigmoid(outputs["q_halt_logits"]).mean(),
            "frac_halted": carry["halted"].mean(),  # fraction halted
            "avg_steps": carry["steps"].mean(),  # average step index
        }

        return loss, carry, mx.sum(is_correct), stats

    def load_checkpoint(self):
        """Load model checkpoint from disk and return metadata"""
        if not self.resume_dir or not self.resume_dir.exists():
            raise ValueError(f"Resume directory not found: {self.resume_dir}")

        # Load training history
        history_path = self.resume_dir / "training_history.json"
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
                self.start_epoch = history["epoch"]
                self.train_error_trace = history["train_error_trace"]
                self.train_acc_trace = history["train_acc_trace"]
                self.val_error_trace = history["val_error_trace"]
                self.val_acc_trace = history["val_acc_trace"]
                self.val_epochs = history.get("val_epochs", [])

                # If val_epochs is empty but we have validation data, reconstruct it
                # This handles old checkpoints that didn't track val_epochs
                if not self.val_epochs and self.val_acc_trace:
                    # We don't know the exact epochs, so we'll just use indices for now
                    # The plotting code will need to handle this case
                    self.val_epochs = []

                # Load metadata if available
                metadata = {
                    "dataset": history.get("dataset"),
                    "model_config": history.get("model_config"),
                    "batch_size": history.get("batch_size"),
                    "learning_rate": history.get("learning_rate"),
                }

                print(f"Resuming from epoch {self.start_epoch}")
                if metadata["dataset"]:
                    print(f"  Dataset: {metadata['dataset']}")
                if metadata["model_config"]:
                    print(f"  Model config: {metadata['model_config']}")
        else:
            raise ValueError(f"Training history not found: {history_path}")

        # Find and load the latest checkpoint
        checkpoint_path = self.resume_dir / f"checkpoint_epoch_{self.start_epoch}.safetensors"
        if checkpoint_path.exists():
            self.model.load_weights(str(checkpoint_path))
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        return metadata

    def save_checkpoint(self, epoch: int, is_best: bool = False, batch_size: int = None, learning_rate: float = None):
        """Save model checkpoint to disk"""
        if not self.save_dir:
            return

        # Save model weights using MLX's save function
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.safetensors"
        self.model.save_weights(str(checkpoint_path))

        # Save training history as JSON with metadata
        history_path = self.save_dir / "training_history.json"
        history = {
            "epoch": epoch,
            "train_error_trace": self.train_error_trace,
            "train_acc_trace": self.train_acc_trace,
            "val_error_trace": self.val_error_trace,
            "val_acc_trace": self.val_acc_trace,
            "val_epochs": self.val_epochs,
            "dataset": self.dataset,
            "model_config": self.model_config,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Save best model if specified
        if is_best:
            best_path = self.save_dir / "best_model.safetensors"
            self.model.save_weights(str(best_path))
            print(f"  → Saved best model to {best_path}")

        print(f"  → Saved checkpoint to {checkpoint_path}")

    def train(self, train, val=None, epochs: int = 10, val_freq: int = 1, batch_size: int = None, learning_rate: float = None):
        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(carry, batch):
            train_step_fn = nn.value_and_grad(self.model, self.eval_fn)
            (loss, carry, correct, stats), grads = train_step_fn(carry, batch)
            grads, _ = clip_grad_norm(grads, max_norm=1.0)
            self.optimizer.update(self.model, grads)
            return loss, carry, correct, stats

        carry = None
        # Initialize best_val_acc from loaded history if resuming
        best_val_acc = max(self.val_acc_trace) if self.val_acc_trace else 0.0

        # When resuming, epochs means "additional epochs"
        # When not resuming, epochs means "total epochs"
        total_epochs = self.start_epoch + epochs
        epoch_bar = tqdm(range(self.start_epoch, total_epochs), desc="Training", unit="epoch", initial=self.start_epoch, total=total_epochs)
        for epoch_idx in epoch_bar:
            self.model.train()
            train.reset()
            total_loss, total_correct, n = 0, 0, 0

            q_prob_accum = 0.0
            frac_halted_accum = 0.0
            avg_steps_accum = 0.0
            n_batches = 0

            for batch in train:
                batch = {k: mx.array(v) for k, v in batch.items()}
                if (carry is None) or (
                    carry["halted"].shape[0] != batch["image"].shape[0]
                ):  # reset for different batch sizes
                    carry = self.model.initial_carry(batch)

                loss, carry, correct, stats = step(carry, batch)
                mx.eval(state)

                self.ema_params = ema_update(self.ema_params, self.model)

                total_loss += loss.item() * batch["image"].shape[0]
                total_correct += int(correct)
                n += batch["image"].shape[0]

                q_prob_accum += float(stats["q_prob_mean"])
                frac_halted_accum += float(stats["frac_halted"])
                avg_steps_accum += float(stats["avg_steps"])
                n_batches += 1

            avg_train_loss = total_loss / n
            avg_train_acc = total_correct / n

            self.train_error_trace.append(avg_train_loss)
            self.train_acc_trace.append(avg_train_acc)

            postfix = {
                "train_loss": f"{avg_train_loss:.3f}",
                "train_acc": f"{avg_train_acc:.3f}",
                "p_halt": f"{q_prob_accum / n_batches:.3f}",
                "frac_halt": f"{frac_halted_accum / n_batches:.3f}",
                "avg_steps": f"{avg_steps_accum / n_batches:.2f}",
            }

            if val is not None and (epoch_idx + 1) % val_freq == 0:
                avg_val_loss, avg_val_acc = self.evaluate(val)
                self.val_error_trace.append(avg_val_loss)
                self.val_acc_trace.append(avg_val_acc)
                self.val_epochs.append(epoch_idx + 1)
                postfix.update(
                    {"val_loss": f"{avg_val_loss:.3f}", "val_acc": f"{avg_val_acc:.3f}"}
                )

                # Check if this is the best model so far
                is_best = avg_val_acc > best_val_acc
                if is_best:
                    best_val_acc = avg_val_acc

                # Save checkpoint
                self.save_checkpoint(epoch_idx + 1, is_best=is_best, batch_size=batch_size, learning_rate=learning_rate)

            epoch_bar.set_postfix(postfix)

        # Save final checkpoint
        if self.save_dir:
            self.save_checkpoint(total_epochs, is_best=False, batch_size=batch_size, learning_rate=learning_rate)

    def evaluate(self, test):
        self.model.eval()
        test.reset()
        total_loss, total_correct, n = 0, 0, 0

        with use_ema(self.model, self.ema_params):
            val_bar = tqdm(test, desc="Validating", leave=False)
            for batch in val_bar:
                batch = {k: mx.array(v) for k, v in batch.items()}
                carry = self.model.initial_carry(batch)

                # Limit iterations to prevent infinite loops
                max_iterations = self.model.config.halt_max_steps
                for _ in range(max_iterations):
                    loss, carry, correct, stats = self.eval_fn(carry, batch)
                    if carry["halted"].all():
                        break

                total_loss += loss.item() * batch["image"].shape[0]
                total_correct += int(correct)
                n += batch["image"].shape[0]

                val_bar.set_postfix({
                    "val_loss": f"{total_loss / n:.3f}",
                    "val_acc": f"{total_correct / n:.3f}"
                })

        avg_loss = total_loss / n
        avg_acc = total_correct / n

        return avg_loss, avg_acc
