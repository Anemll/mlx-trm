# CORRECTED: RMSNorm Parameter Comparison

## The Truth About RMSNorm Parameters

### CUDA Implementation
**Parameter-free RMSNorm** (no learnable scale/bias)
- Only uses `eps` for numerical stability
- No trainable parameters
- Effectively has gain = 1.0 (fixed)
- Configurable via `arch.rms_norm_eps`

```python
# Parameter-free RMSNorm
class RMSNorm:
    def __init__(self, eps=1e-6):
        self.eps = eps
        # NO learnable parameters!

    def forward(self, x):
        # Just normalize, no scaling
        rms = sqrt(mean(x²) + eps)
        return x / rms
```

**CUDA model norm parameters: 0** ❌

---

### MLX Implementation
**Standard RMSNorm with learnable scale** (`nn.RMSNorm`)
- Has a learnable `weight` parameter of shape `(dim,)`
- Initialized to ones
- Trainable via gradients

```python
# MLX's nn.RMSNorm (from source)
class RMSNorm:
    def __init__(self, dims: int, eps: float = 1e-5):
        self.weight = mx.ones((dims,))  # ← Learnable!
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)
```

**MLX model norm parameters: 2,048** ✅
- 2 blocks × 2 norms × 512 dims = 2,048 trainable parameters

---

## The Key Difference

| Aspect | CUDA | MLX |
|--------|------|-----|
| **RMSNorm type** | Parameter-free | With learnable scale |
| **Trainable params** | 0 | 2,048 |
| **Normalization** | ✅ Yes | ✅ Yes |
| **Learnable scaling** | ❌ No (fixed gain=1) | ✅ Yes (learned per dim) |

---

## Why This Matters

### Parameter-free RMSNorm (CUDA)
**Advantages:**
- ✅ Fewer parameters (saves 2K params)
- ✅ One less thing to tune
- ✅ Slightly faster (no extra multiply)
- ✅ Forces model to learn proper scale in other layers

**Disadvantages:**
- ❌ Less flexibility per layer
- ❌ Cannot learn different scales per feature

### Learnable RMSNorm (MLX)
**Advantages:**
- ✅ More flexible (can learn optimal scales)
- ✅ Standard in most modern transformers (LLaMA, GPT, etc.)
- ✅ Can help with optimization dynamics

**Disadvantages:**
- ❌ Extra 2K parameters
- ❌ Slightly more computation

---

## Impact on Your Comparison

### Updated Parameter Count

**CUDA (parameter-free norms):**
```
Token embeddings:        6,144
LM head:                 6,144
Q-head:                  1,026
Puzzle emb:         15,219,712
Carry init:              1,024
2× Transformer blocks:
  - QKV, O, MLP:     6,701,056
  - RMSNorm:                 0  ← No params!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:              21,935,106
Trainable:           6,829,058  (puzzle emb is sparse)
```

**MLX (learnable norms):**
```
Token embeddings:        6,144
LM head:                 6,144
Q-head:                    513
Puzzle emb:             24,192
Carry init:              1,024
Q-token:                   512
2× Transformer blocks:
  - QKV, O, MLP:     6,091,776
  - RMSNorm:             2,048  ← Has params!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total:               6,132,353
```

**Corrected difference:**
- MLX has **2,048 MORE parameters** from RMSNorm than CUDA
- This is opposite of what I initially claimed!

---

## Should You Match CUDA's Parameter-free Norms?

### Option 1: Keep MLX's Learnable Norms (Current)
**Recommended** ✅

**Reasons:**
- Standard practice in modern transformers
- Extra 2K params (~0.03% of model) is negligible
- May help training dynamics
- More flexible

### Option 2: Switch to Parameter-free Norms (Match CUDA)
To exactly match CUDA, you'd need to implement:

```python
# In models/trm_arc.py
class ParameterFreeRMSNorm(nn.Module):
    """RMSNorm without learnable parameters (matching CUDA impl)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Compute RMS
        ms = mx.mean(x * x, axis=-1, keepdims=True)
        rms = mx.sqrt(ms + self.eps)
        # Normalize (no learnable scale)
        return x / rms

# Then in Block class:
class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.n1 = ParameterFreeRMSNorm()  # ← No params
        self.attn = Attention(dim, heads)
        self.n2 = ParameterFreeRMSNorm()  # ← No params
        self.ff = SwiGLU(dim, int(8 / 3.0 * dim))
```

This would save 2,048 parameters and exactly match CUDA's approach.

---

## Recommendation

**Don't worry about this difference!** Here's why:

1. **2,048 parameters is tiny** (0.03% of your 6.3M model)
2. **Learnable norms are standard** in modern architectures (LLaMA, GPT-4, etc.)
3. **May actually improve training** by giving the model more flexibility
4. **The real difference** is still the puzzle embeddings (15.2M vs 24K)

**Focus on the puzzle embedding strategy**, not the norm layer choice!

---

## Summary

### What I Got Wrong Initially ❌
- I claimed both implementations trained norm layers
- I said the difference was just "reporting convention"
- **Wrong!** CUDA uses parameter-free norms, MLX uses learnable norms

### The Truth ✅
- **CUDA:** 0 norm parameters (parameter-free RMSNorm)
- **MLX:** 2,048 norm parameters (standard learnable RMSNorm)
- **Both normalize correctly**, but MLX adds learnable scaling
- **Impact:** Minimal (~0.03% of model size)
- **Recommendation:** Keep MLX's learnable norms (better/standard)

### What Still Matters 🎯
The **puzzle embedding** difference (15.2M → 24K) remains the primary architectural difference by far!
