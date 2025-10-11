from contextlib import contextmanager
from functools import partial

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
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

        self.ema_params = model.parameters()

        self.train_error_trace: list[float] = []
        self.train_acc_trace: list[float] = []
        self.val_error_trace: list[float] = []
        self.val_acc_trace: list[float] = []

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

    def train(self, train, val=None, epochs: int = 10, val_freq: int = 1):
        state = [self.model.state, self.optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(carry, batch):
            train_step_fn = nn.value_and_grad(self.model, self.eval_fn)
            (loss, carry, correct, stats), grads = train_step_fn(carry, batch)
            grads, _ = clip_grad_norm(grads, max_norm=1.0)
            self.optimizer.update(self.model, grads)
            return loss, carry, correct, stats

        carry = None

        epoch_bar = tqdm(range(epochs), desc="Training", unit="epoch")
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
                postfix.update(
                    {"val_loss": f"{avg_val_loss:.3f}", "val_acc": f"{avg_val_acc:.3f}"}
                )

            epoch_bar.set_postfix(postfix)

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
