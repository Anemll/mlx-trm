#!/usr/bin/env python3
"""Test the updated plotting with exact match display."""

import mlx.core as mx
import numpy as np

# Simulate training history
class MockTrainer:
    def __init__(self):
        self.train_acc_trace = [0.1, 0.3, 0.5, 0.6, 0.65, 0.7, 0.72, 0.73]
        self.val_acc_trace = [0.08, 0.25, 0.4, 0.5]
        self.val_exact_match_trace = [0.0, 0.0, 0.01, 0.03]  # 0%, 0%, 1%, 3%
        self.val_epochs = [2, 4, 6, 8]

manager = MockTrainer()
test_data = type('obj', (object,), {'batch_size': 32, '__len__': lambda self: 11})()

# Generate plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5, 3))
lw = 2

train_epochs = list(range(1, len(manager.train_acc_trace) + 1))
ax.plot(train_epochs, mx.array(manager.train_acc_trace) * 100, label="train", color="r", lw=lw)

if manager.val_acc_trace:
    val_epochs = manager.val_epochs if manager.val_epochs else list(range(1, len(manager.val_acc_trace) + 1))
    ax.plot(val_epochs, mx.array(manager.val_acc_trace) * 100, label="val", color="b", lw=lw)

# Plot exact match percentage if available
if manager.val_exact_match_trace:
    val_epochs = manager.val_epochs if manager.val_epochs else list(range(1, len(manager.val_exact_match_trace) + 1))
    ax.plot(val_epochs, mx.array(manager.val_exact_match_trace) * 100, label="exact match", color="g", lw=lw, linestyle="--")

# Add stats
final_train_acc = manager.train_acc_trace[-1] * 100 if manager.train_acc_trace else 0
final_val_acc = manager.val_acc_trace[-1] * 100 if manager.val_acc_trace else 0
total_epochs = len(manager.train_acc_trace)

# Calculate exact match stats
final_exact_match = manager.val_exact_match_trace[-1] if manager.val_exact_match_trace else 0
# Estimate validation set size (usually 341 for ARC evaluation set)
val_set_size = len(test_data) * test_data.batch_size if hasattr(test_data, 'batch_size') else 341
exact_match_count = int(final_exact_match * val_set_size)

info_text = f"Epochs: {total_epochs}\nFinal Train: {final_train_acc:.2f}%\nFinal Val: {final_val_acc:.2f}%\nExact Match: {exact_match_count}/{val_set_size}"
ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax.legend()
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy (%)")
ax.set_title("ARC-AGI Training")
fig.tight_layout()

# Save test plot
fig.savefig("test_plot_with_exact.png", dpi=150, bbox_inches='tight')
print("Test plot saved to test_plot_with_exact.png")

print("\n✅ Plot now includes:")
print("  - Green dashed line for exact match %")
print("  - Stats box shows 'Exact Match: 10/352'")
print(f"  - From {final_exact_match:.1%} exact match rate on {val_set_size} samples")

plt.close()