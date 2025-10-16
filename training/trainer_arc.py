"""Trainer for ARC models with sequence-to-sequence loss."""

from contextlib import contextmanager
from functools import partial
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import Optimizer, clip_grad_norm
from mlx.utils import tree_map, tree_flatten
from tqdm import tqdm


def ema_update(ema_params, model, alpha=0.999):
    """Exponential Moving Average update.

    Paper uses EMA coefficient of 0.999 for model parameters.
    """
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


class ARCTrainer:
    """Trainer for ARC sequence-to-sequence models."""

    def __init__(self, model: nn.Module, optimizer: Optimizer, save_dir: str = None, resume_dir: str = None,
                 use_github_loss: bool = False, resume_from_epoch: int = None, print_optimizer_dtypes: bool = False):
        self.model = model
        self.optimizer = optimizer
        # NOTE: Reference uses separate optimizers for puzzle embeddings (SGD, lr=1e-2)
        # MLX doesn't easily support parameter groups, so we use single optimizer
        self.save_dir = Path(save_dir) if save_dir else None
        self.resume_dir = Path(resume_dir) if resume_dir else None
        self.use_github_loss = use_github_loss  # Flag to use GitHub version (with 0.5 weight)
        self.resume_from_epoch = resume_from_epoch  # Specific epoch to load from
        self.print_optimizer_dtypes = print_optimizer_dtypes

        self.ema_params = model.parameters()

        self.train_error_trace: list[float] = []
        self.train_acc_trace: list[float] = []
        self.train_lm_loss_trace: list[float] = []
        self.train_q_halt_loss_trace: list[float] = []
        self.train_seq_correct_trace: list[float] = []
        self.train_p_halt_trace: list[float] = []
        self.train_avg_steps_trace: list[float] = []
        self.train_lr_trace: list[float] = []  # Track learning rate changes
        self.val_error_trace: list[float] = []
        self.val_acc_trace: list[float] = []
        self.val_exact_match_trace: list[float] = []
        self.val_exact_match_count_trace: list[int] = []  # Track actual puzzle counts
        self.val_total_samples: int = 0  # Total validation samples
        self.val_epochs: list[int] = []
        self.start_epoch = 0

        # Load checkpoint if resuming
        if self.resume_dir:
            self.load_checkpoint()

        # Create save directory if specified
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoints will be saved to: {self.save_dir}")

    def eval_fn(self, carry, batch):
        """Evaluation function for sequence-to-sequence task.

        Computes:
        - Cross-entropy loss between predicted and target sequences
        - Binary cross-entropy for Q-learning (based on correctness)
        - Token-level accuracy

        Note: Puzzle IDs are automatically handled by the model's embedding layer
        when present in the batch (matching TinyRecursiveModels).
        """
        carry, outputs = self.model(carry, batch)

        # Target tokens
        target = batch["output_tokens"]

        # Logits: (batch, seq_len, vocab_size)
        logits = outputs["logits"]

        # ============================================================================
        # LOSS COMPUTATION - Following TinyRecursiveModels Implementation
        # Reference: github.com/SamsungSAILMontreal/TinyRecursiveModels/models/losses.py
        # ============================================================================

        # 1. LANGUAGE MODEL LOSS (Cross-Entropy with padding masking)
        # GitHub uses ignore_index=-100, we use weights to mask padding tokens
        is_not_padding_flat = (target != 10).reshape(-1).astype(mx.float32)

        lm_loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            weights=is_not_padding_flat,  # Zero weight for padding token=10
            reduction="mean"
        )

        # 2. COMPUTE SEQUENCE CORRECTNESS (for halting target)
        # A sequence is correct only if ALL non-padding tokens match
        pred_tokens = mx.argmax(logits, axis=-1)
        is_not_padding = (target != 10)  # 10 is padding token
        correct_tokens = (pred_tokens == target) & is_not_padding

        # Token-level accuracy for metrics
        token_correct = correct_tokens.astype(mx.float32).sum()
        token_total = is_not_padding.astype(mx.float32).sum()
        token_accuracy = token_correct / mx.maximum(token_total, mx.array(1.0))

        # Sequence-level correctness (GitHub: seq_is_correct)
        seq_is_correct = (correct_tokens.sum(axis=1) == is_not_padding.sum(axis=1)).astype(mx.float32)

        # 3. Q-HALT LOSS (Binary Cross-Entropy with scalar logit)
        # Target: halt (1) when the entire sequence is correct, else continue (0)
        # outputs["q_halt_logits"] is (batch,) scalar logit for halt
        q_halt_target = seq_is_correct.astype(mx.float32)
        q_halt_loss = nn.losses.binary_cross_entropy(
            outputs["q_halt_logits"],
            q_halt_target,
            with_logits=True,
            reduction="mean",
        )

        # 4. Q-CONTINUE LOSS (Optional - present in GitHub but not in paper)
        # GitHub comment: "seems totally unnecessary"
        # Paper removes this for simplification
        # Continue loss: always present (target_q_continue will be zeros when disabled)
        q_continue_loss = nn.losses.binary_cross_entropy(
            outputs["q_continue_logits"],
            outputs["target_q_continue"],
            with_logits=True,
            reduction="mean"
        )

        # ============================================================================
        # LOSS COMBINATION - Key difference between GitHub and Paper
        # ============================================================================
        # GitHub version: lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        # Paper version:  lm_loss + q_halt_loss (removes q_continue, no 0.5 weight)
        #
        # We implement the PAPER version by default as it's simpler and achieves same results
        # Comment from Alexia (author): q_continue "seems totally unnecessary"
        # ============================================================================

        if self.use_github_loss:
            # GitHub version with 0.5 weight (from models/losses.py line ~300)
            total_loss = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
        else:
            # Paper version (simplified, recommended)
            total_loss = lm_loss + q_halt_loss

        # Scale loss by batch size (matching reference implementation)
        # Reference: ((1 / global_batch_size) * loss).backward()
        batch_size = target.shape[0]
        total_loss = total_loss / batch_size

        # Compute halt probability from scalar logit (sigmoid)
        q_halt_prob = mx.sigmoid(outputs["q_halt_logits"])

        stats = {
            "lm_loss": lm_loss,  # Language model loss (cross-entropy)
            "q_halt_loss": q_halt_loss,  # Halting loss
            "q_continue_loss": q_continue_loss,  # Continue loss (usually 0)
            "q_halt_prob_mean": q_halt_prob.mean(),  # Mean halt probability
            "frac_halted": carry["halted"].mean(),  # Fraction of sequences halted
            "avg_steps": carry["steps"].mean(),  # Average computation steps
            "seq_correct_rate": seq_is_correct.mean(),  # % of sequences fully correct
            "token_correct": token_correct,
            "token_total": token_total,
            "seq_correct_count": seq_is_correct.sum(),
        }

        # Only return logits during evaluation to avoid large tensors in training stats
        if not self.model.training:
            stats["logits"] = logits

        return total_loss, carry, token_accuracy, stats

    def load_checkpoint(self):
        """Load model checkpoint from disk.

        If resume_from_epoch is specified, loads that specific checkpoint.
        Otherwise, loads the last checkpoint from training history.
        """
        if not self.resume_dir or not self.resume_dir.exists():
            raise ValueError(f"Resume directory not found: {self.resume_dir}")

        # Determine which epoch to load from
        if self.resume_from_epoch is not None:
            # User specified a specific epoch
            checkpoint_epoch = self.resume_from_epoch
            print(f"Loading from specified epoch: {checkpoint_epoch}")

            # Load history but truncate to the specified epoch
            history_path = self.resume_dir / "training_history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    history = json.load(f)
                    # Truncate histories to the specified epoch
                    self.start_epoch = checkpoint_epoch
                    self.train_error_trace = history["train_error_trace"][:checkpoint_epoch]
                    self.train_acc_trace = history["train_acc_trace"][:checkpoint_epoch]
                    self.train_lm_loss_trace = history.get("train_lm_loss_trace", [])[:checkpoint_epoch]
                    self.train_q_halt_loss_trace = history.get("train_q_halt_loss_trace", [])[:checkpoint_epoch]
                    self.train_seq_correct_trace = history.get("train_seq_correct_trace", [])[:checkpoint_epoch]
                    self.train_p_halt_trace = history.get("train_p_halt_trace", [])[:checkpoint_epoch]
                    self.train_avg_steps_trace = history.get("train_avg_steps_trace", [])[:checkpoint_epoch]
                    self.train_lr_trace = history.get("train_lr_trace", [])[:checkpoint_epoch]
                    self.val_error_trace = history["val_error_trace"]
                    self.val_acc_trace = history["val_acc_trace"]
                    self.val_exact_match_trace = history.get("val_exact_match_trace", [])
                    self.val_exact_match_count_trace = history.get("val_exact_match_count_trace", [])
                    self.val_total_samples = history.get("val_total_samples", 0)
                    self.val_epochs = history.get("val_epochs", [])

                    # Restore optimizer step for LR schedule continuity
                    saved_optimizer_step = history.get("optimizer_step", None)
                    if saved_optimizer_step is not None:
                        optimizer_state = getattr(self.optimizer, "state", None)
                        if isinstance(optimizer_state, dict):
                            optimizer_state["step"] = mx.array(saved_optimizer_step)
                        else:
                            try:
                                self.optimizer.step = mx.array(saved_optimizer_step)
                            except AttributeError:
                                print("  Warning: Unable to restore optimizer step (no setter)")
                        print(f"  Restored optimizer step: {saved_optimizer_step}")

                    # Keep only validation results up to the specified epoch
                    self.val_epochs = [e for e in self.val_epochs if e <= checkpoint_epoch]
                    if self.val_epochs:
                        max_val_idx = len(self.val_epochs)
                        self.val_error_trace = self.val_error_trace[:max_val_idx]
                        self.val_acc_trace = self.val_acc_trace[:max_val_idx]
                        self.val_exact_match_trace = self.val_exact_match_trace[:max_val_idx]
            else:
                print(f"Warning: Training history not found, starting fresh from checkpoint")
                self.start_epoch = checkpoint_epoch
        else:
            # Load from last checkpoint in history
            history_path = self.resume_dir / "training_history.json"
            if history_path.exists():
                with open(history_path, "r") as f:
                    history = json.load(f)
                    checkpoint_epoch = history["epoch"]
                    self.start_epoch = checkpoint_epoch
                    self.train_error_trace = history["train_error_trace"]
                    self.train_acc_trace = history["train_acc_trace"]
                    self.train_lm_loss_trace = history.get("train_lm_loss_trace", [])
                    self.train_q_halt_loss_trace = history.get("train_q_halt_loss_trace", [])
                    self.train_seq_correct_trace = history.get("train_seq_correct_trace", [])
                    self.train_p_halt_trace = history.get("train_p_halt_trace", [])
                    self.train_avg_steps_trace = history.get("train_avg_steps_trace", [])
                    self.train_lr_trace = history.get("train_lr_trace", [])
                    self.val_error_trace = history["val_error_trace"]
                    self.val_acc_trace = history["val_acc_trace"]
                    self.val_exact_match_trace = history.get("val_exact_match_trace", [])
                    self.val_exact_match_count_trace = history.get("val_exact_match_count_trace", [])
                    self.val_total_samples = history.get("val_total_samples", 0)
                    self.val_epochs = history.get("val_epochs", [])

                    # Restore optimizer step for LR schedule continuity
                    saved_optimizer_step = history.get("optimizer_step", None)
                    if saved_optimizer_step is not None:
                        optimizer_state = getattr(self.optimizer, "state", None)
                        if isinstance(optimizer_state, dict):
                            optimizer_state["step"] = mx.array(saved_optimizer_step)
                        else:
                            try:
                                self.optimizer.step = mx.array(saved_optimizer_step)
                            except AttributeError:
                                print("  Warning: Unable to restore optimizer step (no setter)")
                        print(f"  Restored optimizer step: {saved_optimizer_step}")

                    # Ensure consistency between val_epochs and validation traces
                    val_trace_lengths = [len(self.val_error_trace), len(self.val_acc_trace), len(self.val_exact_match_trace)]
                    if len(self.val_epochs) != max(val_trace_lengths):
                        print(f"Warning: Mismatched validation trace lengths - epochs: {len(self.val_epochs)}, traces: {val_trace_lengths}")
                        # Truncate to the shortest length to maintain consistency
                        min_len = min(len(self.val_epochs), *val_trace_lengths)
                        if min_len > 0:
                            self.val_epochs = self.val_epochs[:min_len]
                            self.val_error_trace = self.val_error_trace[:min_len]
                            self.val_acc_trace = self.val_acc_trace[:min_len]
                            self.val_exact_match_trace = self.val_exact_match_trace[:min_len]
                        else:
                            self.val_epochs = []
                            self.val_error_trace = []
                            self.val_acc_trace = []
                            self.val_exact_match_trace = []

                    print(f"Resuming from epoch {checkpoint_epoch}")
            else:
                print(f"Warning: Training history not found: {history_path}")
                checkpoint_files = sorted(self.resume_dir.glob("checkpoint_epoch_*.safetensors"))
                if checkpoint_files:
                    latest_checkpoint = checkpoint_files[-1]
                    checkpoint_epoch = int(latest_checkpoint.stem.split("_")[-1])
                    self.start_epoch = checkpoint_epoch
                    self.model.load_weights(str(latest_checkpoint))
                    print(f"Loaded checkpoint from {latest_checkpoint}")
                    return

                best_path = self.resume_dir / "best_model.safetensors"
                if best_path.exists():
                    self.model.load_weights(str(best_path))
                    print(f"Loaded best model from {best_path}")
                    self.start_epoch = 0
                    return

                raise ValueError(
                    f"No checkpoints found in {self.resume_dir}. Provide a valid resume directory or rerun without --resume." 
                )

        # Load the checkpoint weights
        checkpoint_path = self.resume_dir / f"checkpoint_epoch_{checkpoint_epoch}.safetensors"
        if checkpoint_path.exists():
            self.model.load_weights(str(checkpoint_path))
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

    def save_checkpoint(self, epoch: int, is_best: bool = False, batch_size: int = None, learning_rate: float = None):
        """Save model checkpoint to disk.

        Args:
            epoch: The ABSOLUTE epoch number (including resumed epochs)
        """
        if not self.save_dir:
            return

        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.safetensors"
        self.model.save_weights(str(checkpoint_path))

        history_path = self.save_dir / "training_history.json"

        # Get optimizer step count for proper LR schedule resumption
        optimizer_state = getattr(self.optimizer, "state", None)
        optimizer_step = 0
        if isinstance(optimizer_state, dict) and "step" in optimizer_state:
            step_value = optimizer_state["step"]
            if hasattr(step_value, "item"):
                optimizer_step = int(step_value.item())
            else:
                optimizer_step = int(step_value)

        history = {
            "epoch": epoch,  # This should be the absolute epoch count
            "optimizer_step": optimizer_step,  # Save optimizer step for LR schedule
            "train_error_trace": self.train_error_trace,
            "train_acc_trace": self.train_acc_trace,
            "train_lm_loss_trace": self.train_lm_loss_trace,
            "train_q_halt_loss_trace": self.train_q_halt_loss_trace,
            "train_seq_correct_trace": self.train_seq_correct_trace,
            "train_p_halt_trace": self.train_p_halt_trace,
            "train_avg_steps_trace": self.train_avg_steps_trace,
            "train_lr_trace": self.train_lr_trace,
            "val_error_trace": self.val_error_trace,
            "val_acc_trace": self.val_acc_trace,
            "val_exact_match_trace": self.val_exact_match_trace,
            "val_exact_match_count_trace": self.val_exact_match_count_trace,
            "val_total_samples": self.val_total_samples,
            "val_epochs": self.val_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        }
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        if is_best:
            best_path = self.save_dir / "best_model.safetensors"
            self.model.save_weights(str(best_path))
            print(f"  → Saved best model to {best_path}")

        print(f"  → Saved checkpoint to {checkpoint_path}")

    def train(self, train, val=None, epochs: int = 10, val_freq: int = 1, batch_size: int = None, learning_rate: float = None, max_train_samples: int | None = None, skip_halt_check: bool = False):
        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(carry, batch):
            train_step_fn = nn.value_and_grad(self.model, self.eval_fn)
            (loss, carry, accuracy, stats), grads = train_step_fn(carry, batch)
            grads, _ = clip_grad_norm(grads, max_norm=1.0)
            self.optimizer.update(self.model, grads)
            return loss, carry, accuracy, stats

        # Forward-only compiled step (no gradients, no optimizer update)
        @partial(mx.compile, inputs=state, outputs=state)
        def step_fwd_only(carry, batch):
            loss, carry, accuracy, stats = self.eval_fn(carry, batch)
            return loss, carry, accuracy, stats

        best_val_acc = max(self.val_acc_trace) if self.val_acc_trace else 0.0

        total_epochs = self.start_epoch + epochs
        epoch_bar = tqdm(range(self.start_epoch, total_epochs), desc="Training", unit="epoch", initial=self.start_epoch, total=total_epochs)

        printed_dtype_info = False
        for epoch_idx in epoch_bar:
            import time
            epoch_start_time = time.perf_counter()

            self.model.train()
            train.reset()
            total_loss, n_batches = 0.0, 0
            token_correct_total, token_total_total = 0.0, 0.0
            total_samples = 0  # Track actual samples processed
            processed_samples = 0  # For throughput progress
            sample_bar = None
            if max_train_samples is not None:
                sample_bar = tqdm(total=max_train_samples, desc="Samples", unit="ex", leave=False)

            # Get current learning rate from optimizer
            current_lr = self.optimizer.learning_rate
            if callable(current_lr):
                # Learning rate schedule - evaluate at current step
                current_lr_value = float(current_lr(self.optimizer.step).item())
            else:
                # Fixed learning rate
                current_lr_value = float(current_lr)

            q_prob_accum = 0.0
            frac_halted_accum = 0.0
            avg_steps_accum = 0.0
            seq_correct_accum = 0.0
            lm_loss_accum = 0.0
            q_halt_loss_accum = 0.0

            printed_compile_note = False
            for batch in train:
                batch = {k: mx.array(v) for k, v in batch.items()}

                # Initialize carry for each batch
                carry = self.model.initial_carry(batch)

                # CRITICAL: Call model repeatedly until all sequences halt
                # This implements adaptive computation time (ACT) properly
                # Without this loop, model only runs for 1 step regardless of halt_max_steps!
                max_iterations = self.model.config.halt_max_steps
                # Run ACT iterations with a single gradient update on the final iteration
                for iteration in range(max_iterations):
                    if (not printed_compile_note) and (max_train_samples is not None):
                        print("Compiling kernels (first batch may be slow)...", flush=True)
                        printed_compile_note = True
                    is_last_iter = (iteration == max_iterations - 1)

                    if is_last_iter:
                        # Compute gradients and update once per batch
                        loss, carry, accuracy, stats = step(carry, batch)
                        break
                    else:
                        # Forward-only pass to update carry
                        loss, carry, accuracy, stats = step_fwd_only(carry, batch)
                        if not skip_halt_check and carry["halted"].all():
                            # One final gradient step using the halted carry
                            loss, carry, accuracy, stats = step(carry, batch)
                            break

                self.ema_params = ema_update(self.ema_params, self.model)

                # Print dtype info once after first update if requested
                if self.print_optimizer_dtypes and not printed_dtype_info:
                    try:
                        # Parameter dtypes
                        param_dtypes = sorted({str(p.dtype) for _, p in tree_flatten(self.model.parameters())})
                        # Optimizer state dtypes
                        opt_state = getattr(self.optimizer, "state", None)
                        if isinstance(opt_state, dict):
                            opt_dtypes = sorted({str(v.dtype) for _, v in tree_flatten(opt_state) if hasattr(v, "dtype")})
                        else:
                            opt_dtypes = []
                        print(f"\nDType info → params: {param_dtypes} | optimizer_state: {opt_dtypes}")
                    except Exception as e:
                        print(f"Warning: failed to print dtype info: {e}")
                    printed_dtype_info = True

                # Evaluate once per batch (after ACT loop) to avoid per-iteration sync
                mx.eval(loss)
                total_loss += loss.item()
                token_correct_total += float(stats["token_correct"])
                token_total_total += float(stats["token_total"])
                n_batches += 1
                batch_count = batch["input_tokens"].shape[0]
                total_samples += batch_count  # Track samples

                # Update throughput progress bar
                if sample_bar is not None:
                    if processed_samples < max_train_samples:
                        to_add = min(batch_count, max_train_samples - processed_samples)
                        if to_add > 0:
                            sample_bar.update(to_add)
                            processed_samples += to_add

                q_prob_accum += float(stats["q_halt_prob_mean"])
                frac_halted_accum += float(stats["frac_halted"])
                avg_steps_accum += float(stats["avg_steps"])
                seq_correct_accum += float(stats["seq_correct_rate"])
                lm_loss_accum += float(stats["lm_loss"])
                q_halt_loss_accum += float(stats["q_halt_loss"])

                # Early-exit throughput mode: stop after processing a target number of samples
                if max_train_samples is not None and total_samples >= max_train_samples:
                    epoch_time = time.perf_counter() - epoch_start_time
                    throughput = total_samples / epoch_time if epoch_time > 0 and total_samples > 0 else 0.0
                    print(f"\nThroughput: {throughput:.2f} samples/s over {total_samples} samples (target {max_train_samples})")
                    if sample_bar is not None:
                        sample_bar.close()
                    return

            avg_train_loss = total_loss / n_batches
            avg_train_acc = (token_correct_total / token_total_total) if token_total_total > 0 else 0.0

            # Calculate epoch time and throughput
            epoch_time = time.perf_counter() - epoch_start_time
            throughput = total_samples / epoch_time if epoch_time > 0 and total_samples > 0 else 0

            self.train_error_trace.append(avg_train_loss)
            self.train_acc_trace.append(avg_train_acc)
            self.train_lm_loss_trace.append(lm_loss_accum / n_batches)
            self.train_q_halt_loss_trace.append(q_halt_loss_accum / n_batches)
            self.train_seq_correct_trace.append(seq_correct_accum / n_batches)
            self.train_p_halt_trace.append(q_prob_accum / n_batches)
            self.train_avg_steps_trace.append(avg_steps_accum / n_batches)
            self.train_lr_trace.append(current_lr_value)

            # Simplified postfix to fit in terminal (full details saved to history)
            postfix = {
                "loss": f"{avg_train_loss:.3f}",
                "acc": f"{avg_train_acc:.3f}",
                "lm": f"{lm_loss_accum / n_batches:.3f}",
                "q_halt": f"{q_halt_loss_accum / n_batches:.3f}",
                "p_halt": f"{q_prob_accum / n_batches:.3f}",
                "steps": f"{avg_steps_accum / n_batches:.1f}",
                "seq_ok": f"{seq_correct_accum / n_batches:.2f}",
                "lr": f"{current_lr_value:.2e}",
            }

            # Add throughput info if available
            if throughput > 0:
                postfix["samples/s"] = f"{throughput:.2f}"

            # Print LR changes when significant (more than 20% change)
            if len(self.train_lr_trace) > 1:
                prev_lr = self.train_lr_trace[-2]
                if prev_lr != 0:
                    lr_change_pct = abs(current_lr_value - prev_lr) / prev_lr * 100
                    if lr_change_pct > 20:
                        print(f"\n  Learning rate changed: {prev_lr:.2e} → {current_lr_value:.2e} ({lr_change_pct:+.1f}%)")

            if val is not None and (epoch_idx + 1) % val_freq == 0:
                avg_val_loss, avg_val_acc, avg_exact_match, exact_count, val_samples = self.evaluate(val)
                self.val_error_trace.append(avg_val_loss)
                self.val_acc_trace.append(avg_val_acc)
                self.val_exact_match_trace.append(avg_exact_match)
                self.val_exact_match_count_trace.append(exact_count)
                self.val_total_samples = val_samples  # Store for plotting
                self.val_epochs.append(epoch_idx + 1)

                postfix.update(
                    {"val_loss": f"{avg_val_loss:.3f}",
                     "val_acc": f"{avg_val_acc:.3f}",
                     "exact": f"{exact_count}/{val_samples}",
                     "exact_rate": f"{avg_exact_match:.4f}"}
                )

                # Print throughput info
                if throughput > 0:
                    print(f"  Epoch {epoch_idx + 1}: {total_samples} samples in {epoch_time:.1f}s ({throughput:.2f} samples/s)")

                is_best = avg_val_acc > best_val_acc
                if is_best:
                    best_val_acc = avg_val_acc

                self.save_checkpoint(epoch_idx + 1, is_best=is_best, batch_size=batch_size, learning_rate=learning_rate)

            epoch_bar.set_postfix(postfix)

        # Save final checkpoint
        if self.save_dir:
            self.save_checkpoint(total_epochs, is_best=False, batch_size=batch_size, learning_rate=learning_rate)

    def evaluate(self, test):
        self.model.eval()
        test.reset()
        total_loss, n_batches = 0.0, 0
        total_sequence_correct = 0.0
        token_correct_total, token_total_total = 0.0, 0.0
        total_sequences = 0

        with use_ema(self.model, self.ema_params):
            val_bar = tqdm(test, desc="Validating", leave=False)
            for batch in val_bar:
                batch = {k: mx.array(v) for k, v in batch.items()}
                carry = self.model.initial_carry(batch)

                # Limit iterations
                max_iterations = self.model.config.halt_max_steps
                for _ in range(max_iterations):
                    loss, carry, accuracy, stats = self.eval_fn(carry, batch)
                    if carry["halted"].all():
                        break

                total_loss += loss.item()
                token_correct_total += float(stats["token_correct"])
                token_total_total += float(stats["token_total"])
                n_batches += 1
                total_sequences += batch["input_tokens"].shape[0]

                # Calculate exact match accuracy for this batch
                target = batch["output_tokens"]
                logits = stats["logits"]  # Get logits from eval_fn stats
                if logits is not None:
                    pred_tokens = mx.argmax(logits, axis=-1)
                    is_not_padding = (target != 10)
                    correct_tokens = (pred_tokens == target) & is_not_padding
                    # Check if entire sequence matches (excluding padding)
                    sequence_matches = (correct_tokens.sum(axis=1) == is_not_padding.sum(axis=1))
                    total_sequence_correct += float(sequence_matches.sum())

                current_token_acc = (token_correct_total / token_total_total) if token_total_total > 0 else 0.0
                current_exact = (total_sequence_correct / total_sequences) if total_sequences > 0 else 0.0
                val_bar.set_postfix({
                    "val_loss": f"{total_loss / n_batches:.3f}",
                    "token_acc": f"{current_token_acc:.3f}",
                    "exact_match": f"{current_exact:.3f}"
                })

        avg_loss = total_loss / n_batches
        avg_token_acc = (token_correct_total / token_total_total) if token_total_total > 0 else 0.0
        total_samples = total_sequences
        avg_exact_match = (total_sequence_correct / total_sequences) if total_sequences > 0 else 0.0

        # Print evaluation results in single line
        exact_match_count = int(total_sequence_correct)
        print(f"\n  Validation: Accuracy {avg_token_acc:.2%} ({exact_match_count} exact match{'es' if exact_match_count != 1 else ''} of {total_samples})")

        return avg_loss, avg_token_acc, avg_exact_match, exact_match_count, total_samples
