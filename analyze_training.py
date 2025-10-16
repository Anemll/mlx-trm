"""Analyze detailed training metrics from training_history.json"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_training(history_path: str, save_plot: bool = True):
    """Analyze training history and create diagnostic plots."""

    with open(history_path, 'r') as f:
        history = json.load(f)

    print("=" * 80)
    print(f"Training Analysis: {history_path}")
    print("=" * 80)

    # Basic info
    total_epochs = history['epoch']
    print(f"\nTotal Epochs: {total_epochs}")
    print(f"Batch Size: {history.get('batch_size', 'N/A')}")

    if history.get('train_lr_trace'):
        initial_lr = history['train_lr_trace'][0]
        current_lr = history['train_lr_trace'][-1]
        print(f"Learning Rate: {initial_lr:.2e} (initial) → {current_lr:.2e} (current)")
        if current_lr < initial_lr * 0.1:
            print(f"  Note: LR has decayed significantly ({current_lr/initial_lr*100:.1f}% of initial)")
    else:
        print(f"Learning Rate: {history.get('learning_rate', 'N/A')}")

    # Training metrics
    print("\n" + "=" * 80)
    print("TRAINING METRICS")
    print("=" * 80)

    if history['train_acc_trace']:
        latest_train_acc = history['train_acc_trace'][-1] * 100
        print(f"\nToken Accuracy:")
        print(f"  Latest: {latest_train_acc:.2f}%")
        print(f"  Min: {min(history['train_acc_trace']) * 100:.2f}%")
        print(f"  Max: {max(history['train_acc_trace']) * 100:.2f}%")

    if history.get('train_seq_correct_trace'):
        latest_seq = history['train_seq_correct_trace'][-1] * 100
        print(f"\nSequence Correct Rate (all tokens match):")
        print(f"  Latest: {latest_seq:.2f}%")
        print(f"  Min: {min(history['train_seq_correct_trace']) * 100:.2f}%")
        print(f"  Max: {max(history['train_seq_correct_trace']) * 100:.2f}%")

    if history.get('train_p_halt_trace'):
        latest_p_halt = history['train_p_halt_trace'][-1]
        print(f"\nHalting Probability:")
        print(f"  Latest: {latest_p_halt:.4f}")
        print(f"  Min: {min(history['train_p_halt_trace']):.4f}")
        print(f"  Max: {max(history['train_p_halt_trace']):.4f}")
        print(f"  Mean: {np.mean(history['train_p_halt_trace']):.4f}")

        # Trend analysis
        first_half = np.mean(history['train_p_halt_trace'][:len(history['train_p_halt_trace'])//2])
        second_half = np.mean(history['train_p_halt_trace'][len(history['train_p_halt_trace'])//2:])
        trend = "increasing" if second_half > first_half else "decreasing"
        print(f"  Trend: {trend} (1st half: {first_half:.4f}, 2nd half: {second_half:.4f})")

    if history.get('train_avg_steps_trace'):
        latest_steps = history['train_avg_steps_trace'][-1]
        print(f"\nAverage Computation Steps:")
        print(f"  Latest: {latest_steps:.2f}")
        print(f"  Min: {min(history['train_avg_steps_trace']):.2f}")
        print(f"  Max: {max(history['train_avg_steps_trace']):.2f}")

    # Loss breakdown
    print("\n" + "=" * 80)
    print("LOSS BREAKDOWN")
    print("=" * 80)

    if history.get('train_lm_loss_trace'):
        latest_lm = history['train_lm_loss_trace'][-1]
        print(f"\nLanguage Model Loss:")
        print(f"  Latest: {latest_lm:.4f}")
        print(f"  Min: {min(history['train_lm_loss_trace']):.4f}")
        print(f"  Max: {max(history['train_lm_loss_trace']):.4f}")

    if history.get('train_q_halt_loss_trace'):
        latest_q = history['train_q_halt_loss_trace'][-1]
        print(f"\nQ-Halt Loss:")
        print(f"  Latest: {latest_q:.4f}")
        print(f"  Min: {min(history['train_q_halt_loss_trace']):.4f}")
        print(f"  Max: {max(history['train_q_halt_loss_trace']):.4f}")

        # Loss ratio
        if history.get('train_lm_loss_trace'):
            ratio = latest_q / latest_lm if latest_lm > 0 else 0
            print(f"  Ratio (Q-halt/LM): {ratio:.2f}x")

    # Validation metrics
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)

    if history['val_acc_trace']:
        val_accs = [acc * 100 for acc in history['val_acc_trace']]
        best_val_idx = np.argmax(history['val_acc_trace'])
        best_val_epoch = history['val_epochs'][best_val_idx] if history.get('val_epochs') else best_val_idx + 1

        print(f"\nValidation Accuracy:")
        print(f"  Latest: {val_accs[-1]:.2f}% (epoch {history['val_epochs'][-1] if history.get('val_epochs') else 'N/A'})")
        print(f"  Best: {val_accs[best_val_idx]:.2f}% (epoch {best_val_epoch})")
        print(f"  Min: {min(val_accs):.2f}%")

        # Overfitting check
        if len(val_accs) > 1:
            recent_trend = val_accs[-1] - val_accs[best_val_idx]
            if recent_trend < -1.0:
                print(f"\n  ⚠️  WARNING: Validation accuracy dropped {-recent_trend:.2f}% from best")
                print(f"      Model may be overfitting. Consider using checkpoint from epoch {best_val_epoch}")

    if history.get('val_exact_match_trace'):
        exact_matches = [em * 100 for em in history['val_exact_match_trace']]
        best_em_idx = np.argmax(history['val_exact_match_trace'])
        best_em_epoch = history['val_epochs'][best_em_idx] if history.get('val_epochs') else best_em_idx + 1

        print(f"\nExact Match Rate:")
        print(f"  Latest: {exact_matches[-1]:.2f}%")
        print(f"  Best: {exact_matches[best_em_idx]:.2f}% (epoch {best_em_epoch})")
        print(f"  Min: {min(exact_matches):.2f}%")

    # Diagnostic insights
    print("\n" + "=" * 80)
    print("DIAGNOSTIC INSIGHTS")
    print("=" * 80)

    # Check if halting is being learned
    if history.get('train_p_halt_trace'):
        if latest_p_halt < 0.03:
            print("\n⚠️  Low Halting Probability (<0.03):")
            print("    - Model rarely learns to halt early")
            print("    - May not be using adaptive computation effectively")
            print("    - Try increasing --halt-exploration (e.g., 0.2 or 0.3)")
        elif latest_p_halt > 0.15:
            print("\n✓ Good Halting Engagement (>0.15):")
            print("    - Model is learning when to stop reasoning")

    # Check training vs validation gap
    if history['train_acc_trace'] and history['val_acc_trace']:
        train_val_gap = (history['train_acc_trace'][-1] - history['val_acc_trace'][-1]) * 100
        if train_val_gap > 30:
            print(f"\n⚠️  Large Train-Val Gap ({train_val_gap:.1f}%):")
            print("    - Significant overfitting detected")
            print("    - Consider: higher weight decay, smaller model, more regularization")

    # Check sequence correctness
    if history.get('train_seq_correct_trace'):
        if latest_seq < 5:
            print(f"\n⚠️  Low Sequence Correctness ({latest_seq:.1f}%):")
            print("    - Most sequences have at least one error")
            print("    - Model struggling with full solution generation")
            print("    - This is normal for ARC-AGI early training")

    # Create diagnostic plots
    if save_plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Diagnostics - Epoch {total_epochs}', fontsize=14, fontweight='bold')

        epochs = list(range(1, len(history['train_acc_trace']) + 1))

        # Plot 1: Accuracy
        ax = axes[0, 0]
        ax.plot(epochs, [a * 100 for a in history['train_acc_trace']], label='Train Token Acc', color='blue')
        if history.get('train_seq_correct_trace'):
            ax.plot(epochs, [s * 100 for s in history['train_seq_correct_trace']], label='Train Seq Correct', color='cyan')
        if history['val_acc_trace'] and history.get('val_epochs'):
            ax.plot(history['val_epochs'], [a * 100 for a in history['val_acc_trace']], label='Val Acc', color='red', marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Halting Probability
        if history.get('train_p_halt_trace'):
            ax = axes[0, 1]
            ax.plot(epochs, history['train_p_halt_trace'], color='green')
            ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Target ~0.1')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Halting Probability')
            ax.set_title('Halting Behavior')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Average Steps
        if history.get('train_avg_steps_trace'):
            ax = axes[0, 2]
            ax.plot(epochs, history['train_avg_steps_trace'], color='purple')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Average Steps')
            ax.set_title('Computation Steps')
            ax.grid(True, alpha=0.3)

        # Plot 4: Loss Components
        if history.get('train_lm_loss_trace') and history.get('train_q_halt_loss_trace'):
            ax = axes[1, 0]
            ax.plot(epochs, history['train_lm_loss_trace'], label='LM Loss', color='blue')
            ax.plot(epochs, history['train_q_halt_loss_trace'], label='Q-Halt Loss', color='orange')
            ax.plot(epochs, history['train_error_trace'], label='Total Loss', color='red', linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Loss Components')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        # Plot 5: Exact Match
        if history.get('val_exact_match_trace') and history.get('val_epochs'):
            ax = axes[1, 1]
            ax.plot(history['val_epochs'], [em * 100 for em in history['val_exact_match_trace']],
                   color='green', marker='o', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Exact Match (%)')
            ax.set_title('Validation Exact Match')
            ax.grid(True, alpha=0.3)

        # Plot 6: Train-Val Gap
        if history['val_acc_trace'] and history.get('val_epochs'):
            ax = axes[1, 2]
            # Interpolate val accuracy to all epochs for gap calculation
            val_acc_interp = np.interp(epochs, history['val_epochs'], history['val_acc_trace'])
            gap = [(t - v) * 100 for t, v in zip(history['train_acc_trace'], val_acc_interp)]
            ax.plot(epochs, gap, color='red')
            ax.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Warning (20%)')
            ax.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Severe (30%)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Train-Val Gap (%)')
            ax.set_title('Overfitting Indicator')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_dir = Path(history_path).parent
        plot_path = plot_dir / "training_diagnostics.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Diagnostic plot saved to: {plot_path}")

        plt.show()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training history")
    parser.add_argument("history_path", type=str, help="Path to training_history.json")
    parser.add_argument("--no-plot", action="store_true", help="Don't create plots")

    args = parser.parse_args()
    analyze_training(args.history_path, save_plot=not args.no_plot)
