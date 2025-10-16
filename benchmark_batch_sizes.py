#!/usr/bin/env python3
"""Benchmark different batch sizes with the specified model configuration."""

import time
import sys
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from data.arc_json import arc_agi_json
from models.trm_arc import ARCModel, ARCModelConfig
from training.trainer_arc import ARCTrainer

def benchmark_batch_size(batch_size, dim=128, halt_max_steps=8, halt_exploration=0.2, use_bf16=True):
    """Run a quick benchmark for a given batch size."""
    print(f"\n{'='*60}")
    print(f"Benchmarking batch_size={batch_size}")
    print(f"Model config: dim={dim}, halt_max_steps={halt_max_steps}, bf16={use_bf16}")
    print(f"{'='*60}")

    # Load data
    train_data, test_data, meta = arc_agi_json(batch_size)

    print(f"Dataset: {meta['n_train_examples']} examples, {meta['steps_per_epoch']} batches/epoch")

    # Create model
    config = ARCModelConfig(
        vocab_size=meta['vocab_size'],
        max_seq_len=meta['max_seq_len'],
        depth=2,
        dim=dim,
        heads=8,
        n=6,
        T=3,
        halt_max_steps=halt_max_steps,
        halt_exploration_prob=halt_exploration,
        halt_follow_q=True,
    )

    model = ARCModel(config)

    if use_bf16:
        model.set_dtype(mx.bfloat16)

    # Count parameters
    def count_params(params):
        total = 0
        for k, v in params.items():
            if isinstance(v, dict):
                total += count_params(v)
            elif hasattr(v, 'size'):
                total += v.size
        return total

    n_params = count_params(model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Simple optimizer
    optimizer = optim.AdamW(learning_rate=1e-4, betas=[0.9, 0.95], weight_decay=0.01)

    # Create trainer (without saving)
    trainer = ARCTrainer(model, optimizer)

    # Time a few batches
    print("\nTiming first 10 batches...")
    model.train()

    batch_times = []
    total_examples = 0

    for i, batch in enumerate(train_data):
        if i >= 10:
            break

        batch = {k: mx.array(v) for k, v in batch.items()}
        carry = model.initial_carry(batch)

        start = time.time()
        loss, carry, accuracy, stats = trainer.eval_fn(carry, batch)
        mx.eval(loss)  # Force evaluation
        elapsed = time.time() - start

        batch_times.append(elapsed)
        total_examples += batch["input_tokens"].shape[0]

        print(f"  Batch {i+1}: {elapsed:.3f}s ({batch['input_tokens'].shape[0]} examples)")

    # Calculate statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    estimated_epoch_time = avg_batch_time * meta['steps_per_epoch']
    throughput = total_examples / sum(batch_times)

    print(f"\nResults:")
    print(f"  Average batch time: {avg_batch_time:.3f}s")
    print(f"  Estimated epoch time: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f} min)")
    print(f"  Throughput: {throughput:.1f} examples/sec")
    print(f"  Time per example: {1000/throughput:.2f}ms")

    return {
        'batch_size': batch_size,
        'avg_batch_time': avg_batch_time,
        'epoch_time': estimated_epoch_time,
        'throughput': throughput,
        'time_per_example': 1000/throughput
    }

if __name__ == "__main__":
    # Test different batch sizes
    batch_sizes = [32, 64, 128, 256]
    results = []

    for batch_size in batch_sizes:
        try:
            result = benchmark_batch_size(
                batch_size,
                dim=128,
                halt_max_steps=8,
                halt_exploration=0.2,
                use_bf16=True
            )
            results.append(result)
        except Exception as e:
            print(f"Error with batch_size={batch_size}: {e}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch Size':<12} {'Batch Time':<12} {'Epoch Time':<15} {'Throughput':<15} {'Time/Example':<12}")
    print(f"{'-'*60}")

    for r in results:
        print(f"{r['batch_size']:<12} {r['avg_batch_time']:<12.3f} "
              f"{r['epoch_time']:<8.1f}s ({r['epoch_time']/60:.1f}m) "
              f"{r['throughput']:<8.1f} ex/s    {r['time_per_example']:<8.2f}ms")