"""Test the halt exploration logic."""
import mlx.core as mx

# Simulate the halt logic
batch_size = 16
halt_max_steps = 4
halt_exploration_prob = 0.2

print("Testing halt logic with:")
print(f"  batch_size: {batch_size}")
print(f"  halt_max_steps: {halt_max_steps}")
print(f"  halt_exploration_prob: {halt_exploration_prob}")
print()

# Step 1: new_steps = 1
new_steps = mx.ones((batch_size,))
print(f"Step 1: new_steps shape: {new_steps.shape}")

# Check if this is last step
is_last_step = new_steps >= halt_max_steps
print(f"is_last_step: {is_last_step} (should be all False)")

# Q-head wants to continue (negative logits, like p_halt=0.007)
q_halt_logits = -2.0 * mx.ones((batch_size, 1))
print(f"q_halt_logits shape: {q_halt_logits.shape}")
print(f"q_halt_logits > 0: {(q_halt_logits > 0).squeeze()}")

halted = is_last_step | (q_halt_logits.squeeze() > 0)
print(f"After Q-learning, halted: {halted} (should be all False)")
print()

# Exploration: Force some to continue
print("Testing exploration:")
mx.random.seed(42)
random_uniform = mx.random.uniform(shape=q_halt_logits.shape)
print(f"random_uniform shape: {random_uniform.shape}")

should_explore = random_uniform < halt_exploration_prob
print(f"should_explore shape: {should_explore.shape}")
print(f"should_explore: {should_explore.squeeze()}")

min_halt_steps_random = mx.random.randint(
    low=2, high=halt_max_steps + 1, shape=new_steps.shape
)
print(f"min_halt_steps_random shape: {min_halt_steps_random.shape}")
print(f"min_halt_steps_random: {min_halt_steps_random}")

# THIS IS THE PROBLEM: shape mismatch!
try:
    min_halt_steps = should_explore * min_halt_steps_random
    print(f"min_halt_steps shape: {min_halt_steps.shape}")
except Exception as e:
    print(f"ERROR in multiplication: {e}")
    print(f"  should_explore shape: {should_explore.shape}")
    print(f"  min_halt_steps_random shape: {min_halt_steps_random.shape}")
