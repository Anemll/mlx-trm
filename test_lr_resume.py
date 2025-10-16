"""Test script to verify LR schedule continuity when resuming training."""

import json
from pathlib import Path

def check_lr_continuity(checkpoint_dir: str):
    """Check if learning rate continues smoothly when resuming."""

    checkpoint_path = Path(checkpoint_dir)
    history_file = checkpoint_path / "training_history.json"

    if not history_file.exists():
        print(f"❌ History file not found: {history_file}")
        return

    with open(history_file, 'r') as f:
        history = json.load(f)

    lr_trace = history.get("train_lr_trace", [])
    optimizer_step = history.get("optimizer_step", None)
    epoch = history.get("epoch", 0)

    print("=" * 80)
    print(f"LR Resume Test: {checkpoint_dir}")
    print("=" * 80)

    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {epoch}")
    print(f"  Optimizer step: {optimizer_step}")
    print(f"  Total epochs trained: {len(lr_trace)}")

    if lr_trace:
        print(f"\nLearning Rate History:")
        print(f"  First epoch LR: {lr_trace[0]:.6e}")
        print(f"  Latest epoch LR: {lr_trace[-1]:.6e}")

        # Check for discontinuities
        print(f"\nLR Changes (last 10 epochs):")
        start_idx = max(0, len(lr_trace) - 10)
        for i in range(start_idx, len(lr_trace)):
            if i > 0:
                change = (lr_trace[i] - lr_trace[i-1]) / lr_trace[i-1] * 100 if lr_trace[i-1] != 0 else 0
                indicator = "⚠️" if abs(change) > 50 else "✓"
                print(f"  Epoch {i+1}: {lr_trace[i]:.6e} ({change:+.1f}%) {indicator}")
            else:
                print(f"  Epoch {i+1}: {lr_trace[i]:.6e}")

        # Detect abnormal jumps
        if len(lr_trace) > 1:
            max_change = 0
            max_change_epoch = 0
            for i in range(1, len(lr_trace)):
                if lr_trace[i-1] != 0:
                    change = abs((lr_trace[i] - lr_trace[i-1]) / lr_trace[i-1])
                    if change > max_change:
                        max_change = change
                        max_change_epoch = i + 1

            print(f"\nLargest LR change:")
            print(f"  Epoch {max_change_epoch}: {max_change*100:.1f}%")
            if max_change > 0.3:  # 30% change
                print(f"  ⚠️  WARNING: Large LR jump detected!")
            else:
                print(f"  ✓ LR schedule looks smooth")

    if optimizer_step is not None:
        print(f"\n✓ Optimizer step saved: {optimizer_step}")
        print(f"  When resuming, LR schedule will continue from step {optimizer_step}")
    else:
        print(f"\n❌ Optimizer step NOT saved!")
        print(f"  LR schedule will restart from step 0 (causing jumps)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_lr_resume.py <checkpoint_directory>")
        print("\nExample:")
        print("  python test_lr_resume.py my_arc_run_128")
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    check_lr_continuity(checkpoint_dir)
