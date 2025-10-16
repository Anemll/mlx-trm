# Why MLP and Q-Head Were Different

## TL;DR

Both differences were **design choices from different sources**, not intentional optimizations. We should match CUDA exactly.

---

## 1. MLP Dimensions: 8/3 vs 3

### CUDA Implementation
```python
# MLP hidden dimension = 3 × model_dim
mlp_dim = 3 * 512 = 1536

gate_up_proj: (512, 3072)  # 512 → 3072 (6× for gate+up combined)
down_proj:    (1536, 512)  # 1536 → 512
```

### MLX Implementation (Before Fix)
```python
# MLP hidden dimension = 8/3 × model_dim
mlp_dim = int(8/3 * 512) = 1365

w1 (gate_up): (512, 2730)  # 512 → 2730 (2×1365 for gate+up combined)
w2 (down):    (1365, 512)  # 1365 → 512
```

### Why 8/3?

The `8/3` ratio comes from **LLaMA/LLaMA2 architecture**:

```python
# From LLaMA paper
hidden_dim = int(2/3 * 4 * dim)  # = 8/3 * dim
# Rationale: Approximate 4× expansion but with SwiGLU efficiency
```

This is a **common pattern** in modern LLMs:
- Original Transformer: 4× expansion (FFN dim = 4 × model_dim)
- SwiGLU variants: Often use 8/3× ≈ 2.67× (more efficient)
- CUDA TinyRecursiveModels: Uses 3× (simpler, slightly larger)

**Why it was used:** Likely copied from LLaMA-style implementations without checking CUDA reference.

### Why It Matters

```python
# Parameter difference per block:
CUDA: gate_up (3072×512) + down (512×1536) = 2,359,296
MLX:  gate_up (2730×512) + down (512×1365) = 2,096,640
Diff: 262,656 params per block × 2 blocks = 525,312 total
```

**Impact:** 7.7% fewer parameters in MLP layers, slightly less capacity.

---

## 2. Q-Head: Scalar vs 2-Class

### CUDA Implementation
```python
# Binary classification (halt vs continue)
q_head = nn.Linear(512, 2, bias=True)

# Forward:
logits = q_head(q_token)  # shape: (batch, 2)
# logits[0] = "continue", logits[1] = "halt"
# Halting decision: argmax(logits) or logits[1] > logits[0]
```

### MLX Implementation (Before Fix)
```python
# Scalar output (single halt score)
q_head = nn.Linear(512, 1, bias=True)

# Forward:
logit = q_head(q_token)  # shape: (batch,)
# Halting decision: logit > 0
```

### Why Scalar?

This was a **simplification/optimization**:

**Reasoning:**
- For binary classification, you only need 1 output
- The decision `logits[1] > logits[0]` is equivalent to `(logits[1] - logits[0]) > 0`
- Which is equivalent to `single_logit > 0` if we define `single_logit = logits[1] - logits[0]`

**Mathematically equivalent:**
```python
# 2-class approach:
class_0_logit, class_1_logit = model(x)
halt = class_1_logit > class_0_logit

# Scalar approach (setting class_0 = 0):
halt_logit = model(x)
halt = halt_logit > 0
```

### Why It Matters

```python
# Parameter difference:
CUDA: Linear(512, 2) + bias(2) = 1,024 + 2 = 1,026
MLX:  Linear(512, 1) + bias(1) = 512 + 1 = 513
Diff: 513 params
```

**Impact:** Functionally equivalent but not architecturally identical.

### Potential Subtle Differences

While mathematically equivalent, there could be **training dynamics differences**:

1. **Gradient flow:**
   - 2-class: Gradients split between two outputs
   - Scalar: All gradients go to single output

2. **Initialization:**
   - 2-class: Two separate weight vectors
   - Scalar: Single weight vector

3. **Softmax vs Sigmoid (if used):**
   - 2-class: Typically uses softmax
   - Scalar: Typically uses sigmoid

4. **Loss computation:**
   - 2-class: Cross-entropy loss
   - Scalar: Binary cross-entropy (which is equivalent, but computed differently)

---

## Should We Match CUDA Exactly?

**YES**, for these reasons:

### For MLP Dimensions:
✅ **Match reference implementation** - No reason to deviate
✅ **Slightly more capacity** - 3× vs 2.67× expansion
✅ **Simpler** - Clean integer (1536 vs 1365.33...)
✅ **Fair comparison** - Can't compare results if architectures differ

### For Q-Head:
✅ **Exact architectural match** - Eliminates any potential differences
✅ **Training dynamics** - Matches gradient flow exactly
✅ **Initialization** - Matches weight initialization strategy
✅ **Debugging** - Easier to compare if identical

---

## Summary Table

| Component | MLX (Before) | CUDA (Target) | Why Different? | Should Fix? |
|-----------|--------------|---------------|----------------|-------------|
| **MLP dim** | 1365 (8/3×) | 1536 (3×) | Copied from LLaMA | ✅ Yes |
| **Q-head** | 1 output | 2 outputs | Optimization attempt | ✅ Yes |
| **Impact** | -525,825 params | Reference | 7.5% smaller MLP | ✅ Yes |

---

## What We're Fixing

1. **MLP:** Change from `int(8/3 * dim)` → `3 * dim` = 1536
2. **Q-Head:** Change from `Linear(dim, 1)` → `Linear(dim, 2)`

After fix:
- MLX will have **525,825 MORE parameters** (matching CUDA MLP/Q-head)
- Core architecture will be **identical** to CUDA reference
- Only remaining difference: Puzzle embeddings (intentional design choice)
