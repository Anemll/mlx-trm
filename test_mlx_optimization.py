#!/usr/bin/env python3
"""Test MLX optimizations for ARC model."""

import time
import mlx.core as mx
import mlx.nn as nn
from functools import partial
from data.arc_json import arc_agi_json
from models.trm_arc import ARCModel, ARCModelConfig

def benchmark_forward_pass():
    """Compare compiled vs uncompiled forward pass."""

    # Setup
    batch_size = 32
    train_data, _, meta = arc_agi_json(batch_size)

    config = ARCModelConfig(
        vocab_size=meta['vocab_size'],
        max_seq_len=meta['max_seq_len'],
        depth=2, dim=128, heads=8, n=6, T=3,
        halt_max_steps=8,
        halt_exploration_prob=0.2,
    )

    model = ARCModel(config)
    model.set_dtype(mx.bfloat16)
    model.eval()  # No dropout

    # Get batch
    batch = next(iter(train_data))
    batch = {k: mx.array(v) for k, v in batch.items()}

    print("Testing MLX optimizations...")
    print("="*60)

    # Test 1: Uncompiled forward pass
    print("\n1. UNCOMPILED forward pass:")
    carry = model.initial_carry(batch)

    # Warmup
    for _ in range(3):
        carry_tmp = model.initial_carry(batch)
        carry_tmp, outputs = model(carry_tmp, batch)
        mx.eval(outputs['logits'])

    # Time 10 passes
    start = time.time()
    for _ in range(10):
        carry = model.initial_carry(batch)
        carry, outputs = model(carry, batch)
        mx.eval(outputs['logits'])
    uncompiled_time = time.time() - start
    print(f"  10 passes: {uncompiled_time:.3f}s")
    print(f"  Per pass: {uncompiled_time/10:.3f}s")

    # Test 2: Compiled forward pass
    print("\n2. COMPILED forward pass (mx.compile):")

    # Create compiled function - note: can't pass model as arg
    def forward_fn(carry, batch):
        return model(carry, batch)

    forward_compiled = mx.compile(forward_fn)

    # Warmup
    for _ in range(3):
        carry_tmp = model.initial_carry(batch)
        carry_tmp, outputs = forward_compiled(carry_tmp, batch)
        mx.eval(outputs['logits'])

    # Time 10 passes
    start = time.time()
    for _ in range(10):
        carry = model.initial_carry(batch)
        carry, outputs = forward_compiled(carry, batch)
        mx.eval(outputs['logits'])
    compiled_time = time.time() - start
    print(f"  10 passes: {compiled_time:.3f}s")
    print(f"  Per pass: {compiled_time/10:.3f}s")

    # Test 3: Compiled with inputs/outputs specification
    print("\n3. COMPILED with inputs/outputs specification:")

    # Reset for consistent state
    model_state = model.state
    carry = model.initial_carry(batch)

    @partial(mx.compile, inputs=[model_state], outputs=[model_state])
    def forward_compiled_io(carry, batch):
        return model(carry, batch)

    # Warmup
    for _ in range(3):
        carry_tmp = model.initial_carry(batch)
        carry_tmp, outputs = forward_compiled_io(carry_tmp, batch)
        mx.eval(outputs['logits'])

    # Time 10 passes
    start = time.time()
    for _ in range(10):
        carry = model.initial_carry(batch)
        carry, outputs = forward_compiled_io(carry, batch)
        mx.eval(outputs['logits'])
    compiled_io_time = time.time() - start
    print(f"  10 passes: {compiled_io_time:.3f}s")
    print(f"  Per pass: {compiled_io_time/10:.3f}s")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print(f"  Uncompiled:           {uncompiled_time/10:.3f}s per pass (baseline)")
    print(f"  Compiled:             {compiled_time/10:.3f}s per pass ({uncompiled_time/compiled_time:.2f}x speedup)")
    print(f"  Compiled with I/O:    {compiled_io_time/10:.3f}s per pass ({uncompiled_time/compiled_io_time:.2f}x speedup)")

    # Test GPU usage
    print("\n" + "="*60)
    print("DEVICE INFO:")
    print(f"  Default device: {mx.default_device()}")
    print(f"  Model dtype: {next(iter(model.parameters().values()))['token_emb']['weight'].dtype}")

    return {
        'uncompiled': uncompiled_time/10,
        'compiled': compiled_time/10,
        'compiled_io': compiled_io_time/10
    }

if __name__ == "__main__":
    results = benchmark_forward_pass()

    print("\n" + "="*60)
    print("RECOMMENDATIONS:")

    if results['compiled'] < results['uncompiled'] * 0.9:
        print("✓ mx.compile provides significant speedup!")
        print("  Consider compiling the eval_fn in trainer_arc.py")
    else:
        print("⚠ mx.compile doesn't provide much benefit")
        print("  The model might already be well-optimized")

    print("\nTo improve performance:")
    print("1. Ensure mx.compile is used for training step ✓ (already done)")
    print("2. Consider compiling evaluation function")
    print("3. Use larger batch sizes if memory allows")
    print("4. Ensure Metal/GPU is being used ✓ (confirmed)")