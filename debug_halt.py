"""Debug script to test halt exploration logic."""
import sys
sys.path.insert(0, '.')

from models.trm_arc import ARCModel, ARCModelConfig
import mlx.core as mx

# Create a small model
config = ARCModelConfig(
    vocab_size=12,
    max_seq_len=100,
    depth=2,
    dim=128,
    heads=4,
    halt_max_steps=4,
    halt_exploration_prob=0.2,
)

model = ARCModel(config)
model.train()  # Enable training mode

# Create a dummy batch
batch_size = 16
batch = {
    "input_tokens": mx.random.randint(0, 10, (batch_size, 100)),
    "output_tokens": mx.random.randint(0, 10, (batch_size, 100)),
    "puzzle_ids": mx.array([1] * batch_size),
}

# Initialize carry
carry = model.initial_carry(batch)

print("=" * 70)
print("Testing Halt Exploration Logic")
print("=" * 70)
print(f"Config:")
print(f"  halt_max_steps: {config.halt_max_steps}")
print(f"  halt_exploration_prob: {config.halt_exploration_prob}")
print(f"  model.training: {model.training}")
print()

# Run forward pass
carry, outputs = model(carry, batch)

print(f"After 1st forward pass:")
print(f"  steps: {carry['steps']}")
print(f"  halted: {carry['halted']}")
print(f"  q_halt_logits: {outputs['q_halt_logits']}")
print()

# Run a few more steps
for i in range(2, 6):
    carry, outputs = model(carry, batch)
    print(f"After step {i}:")
    print(f"  steps: {carry['steps']}")
    print(f"  halted (count): {carry['halted'].sum().item()} / {batch_size}")
    print(f"  avg steps: {carry['steps'].mean().item():.2f}")
    print()

print("=" * 70)
print("Summary:")
print(f"  Final avg steps: {carry['steps'].mean().item():.2f}")
print(f"  Expected: > 1.0 if exploration is working")
if carry['steps'].mean().item() > 1.0:
    print("  ✅ Exploration IS working!")
else:
    print("  ❌ Exploration NOT working - all sequences halted at step 1")
