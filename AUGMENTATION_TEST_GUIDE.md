# Testing Augmented Dataset (Without Training)

This guide shows you how to test and visualize the augmented ARC dataset without running any training.

---

## Quick Start

### 1. Test the Dataset Loader Directly

The simplest way is to run the augmented dataset module directly:

```bash
# Activate your environment first
source .venv/bin/activate

# Test the dataset loader
python data/arc_augmented.py
```

**Expected output:**
```
Testing augmented ARC data loader...
Loaded 123497 examples from 800 tasks (training + 400 extra)
  Unique puzzle IDs: 800
  Augmentations enabled: dihedral=True, color=True

Metadata:
  vocab_size: 12
  max_seq_len: 900
  n_train_examples: 123497
  n_test_examples: 1363
  ...

Fetching first batch...
Batch shapes:
  input_tokens: (4, 900)
  output_tokens: (4, 900)
  puzzle_ids: (4,)
```

---

## 2. Comprehensive Test Suite

Run the full test suite to verify all augmentation features:

```bash
python test_augmentation.py
```

**This will test:**
- ✅ Basic loading (no augmentation)
- ✅ Augmented loading (with transformations)
- ✅ Augmentation variations (same puzzle, different transforms)
- ✅ Puzzle ID mapping (train vs eval)
- ✅ Custom augmentation configs
- ✅ Batch iteration

**Expected output:**
```
==============================================================================
AUGMENTED DATASET TEST SUITE
==============================================================================

TEST 1: Basic Loading (No Augmentation)
--------------------------------------------------------------------------------
Loaded 1302 examples from 400 tasks (training)
...

TEST 2: Augmented Loading
--------------------------------------------------------------------------------
Loaded 123497 examples from 800 tasks (training + 400 extra)
Augmentation multiplier: 94.9x
...

✅ ALL TESTS PASSED!
```

---

## 3. Visualize Augmentations

Generate visual examples of augmentations:

```bash
python visualize_augmentations.py
```

**This will create:**
1. **`dihedral_transforms.png`** - Shows all 8 rotation/reflection transforms
2. **`color_permutations.png`** - Shows different color permutations
3. **`puzzle_*_variations.png`** - Shows original + augmented versions

**Example output:**
```
AUGMENTATION STATISTICS
==============================================================================

1. Without augmentation:
   Total examples: 1,302
   Unique puzzles: 800

2. With augmentation (max_aug=30):
   Total examples: 39,797
   Multiplier: 30.6x

3. With augmentation (max_aug=300, default):
   Total examples: 123,497
   Multiplier: 94.9x

4. With augmentation (max_aug=1000, paper):
   Total examples: 341,252
   Multiplier: 262.2x

GENERATING VISUALIZATIONS...
✅ ALL VISUALIZATIONS GENERATED!
```

---

## 4. Interactive Python Session

You can also test interactively in a Python shell:

```bash
python
```

```python
from data.arc_augmented import arc_agi_augmented

# Load dataset with augmentation
train, test, meta = arc_agi_augmented(
    batch_size=4,
    enable_augmentations=True,
    max_augmentations_per_puzzle=100
)

# Print metadata
print("Total examples:", meta['n_train_examples'])
print("Unique puzzles:", meta['num_puzzle_identifiers'])

# Get a batch
batch = next(iter(train))
print("Batch input shape:", batch['input_tokens'].shape)
print("Puzzle IDs:", batch['puzzle_ids'])

# Iterate through batches
for i, batch in enumerate(train):
    if i >= 3:
        break
    print(f"Batch {i}: {len(batch['puzzle_ids'])} examples")
```

---

## 5. Quick Test Commands

### Test without augmentation:
```python
from data.arc_augmented import arc_agi_augmented

train, test, meta = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=False  # ← No augmentation
)
print(f"Examples: {meta['n_train_examples']}")  # Should be ~1302
```

### Test with different augmentation levels:
```python
# Light augmentation (fast)
train, _, meta = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=True,
    max_augmentations_per_puzzle=30  # ← ~30x data
)
print(f"Examples: {meta['n_train_examples']}")  # ~39,000

# Standard augmentation (recommended)
train, _, meta = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=True,
    max_augmentations_per_puzzle=300  # ← ~95x data (default)
)
print(f"Examples: {meta['n_train_examples']}")  # ~123,000

# Paper-level augmentation (slow)
train, _, meta = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=True,
    max_augmentations_per_puzzle=1000  # ← ~260x data
)
print(f"Examples: {meta['n_train_examples']}")  # ~341,000
```

---

## Understanding the Output

### Dataset Statistics

| Metric | No Augmentation | With Augmentation (300) |
|--------|----------------|------------------------|
| **Training examples** | 1,302 | 123,497 |
| **Training tasks** | 400 | 800 (400 + 400 eval demos) |
| **Multiplier** | 1× | ~95× |
| **Unique puzzle IDs** | 800 | 800 |

### Augmentation Types

1. **Dihedral transformations (8 variations):**
   - Identity (no change)
   - Rotate 90°, 180°, 270°
   - Flip horizontal, vertical
   - Transpose, anti-diagonal flip

2. **Color permutations:**
   - Random shuffling of colors 1-9
   - Black (0) always stays black
   - Multiple random permutations per puzzle

### Puzzle IDs

- **Training puzzles:** IDs 1-400
- **Evaluation puzzles:** IDs 401-800
- Each puzzle keeps the same ID across all augmentations
- Augmentation info encoded in string version: `"123|||d1_c0123456789"`

---

## Checking Specific Features

### Check augmentation counts:
```python
from data.arc_augmented import arc_agi_augmented
import numpy as np

train, _, _ = arc_agi_augmented(
    batch_size=1000,
    enable_augmentations=True,
    max_augmentations_per_puzzle=100
)

# Collect all puzzle IDs
all_puzzle_ids = []
for batch in train:
    all_puzzle_ids.extend(batch['puzzle_ids'].tolist())

# Count occurrences per puzzle
unique, counts = np.unique(all_puzzle_ids, return_counts=True)
print("Puzzle ID | Count")
for pid, count in zip(unique[:10], counts[:10]):
    print(f"   {pid:3d}    |  {count:4d}")
```

### Verify transformations work:
```python
from data.arc_augmented import apply_dihedral_transform, apply_color_permutation
import numpy as np

# Create a simple test grid
grid = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Test rotation
rotated = apply_dihedral_transform(grid, 1)  # 90° rotation
print("Original:\n", grid)
print("\nRotated 90°:\n", rotated)

# Test color permutation
perm = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Reverse colors
permuted = apply_color_permutation(grid, perm)
print("\nColor permuted:\n", permuted)
```

---

## Common Use Cases

### 1. Quick sanity check:
```bash
python -c "from data.arc_augmented import arc_agi_augmented; t, _, m = arc_agi_augmented(4, enable_augmentations=True); print(f'✅ Loaded {m[\"n_train_examples\"]} examples')"
```

### 2. Compare augmentation strategies:
```bash
python test_augmentation.py
```

### 3. Visual inspection:
```bash
python visualize_augmentations.py
```

### 4. Profile loading time:
```python
import time
from data.arc_augmented import arc_agi_augmented

start = time.time()
train, test, meta = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=True,
    max_augmentations_per_puzzle=300
)
elapsed = time.time() - start

print(f"Loaded {meta['n_train_examples']} examples in {elapsed:.2f}s")
print(f"Speed: {meta['n_train_examples'] / elapsed:.0f} examples/sec")
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'numpy'`
**Solution:** Activate your virtual environment:
```bash
source .venv/bin/activate
```

### Issue: `FileNotFoundError: Task directory not found`
**Solution:** Make sure ARC data exists:
```bash
ls data/ARC-AGI/data/training/   # Should show .json files
```

If missing, clone the ARC data:
```bash
git clone https://github.com/fchollet/ARC-AGI.git data/ARC-AGI
```

### Issue: Dataset loading is slow
**Solution:** Reduce `max_augmentations_per_puzzle`:
```python
# Instead of 300 (default), use 30 for quick testing
train, _, meta = arc_agi_augmented(
    batch_size=32,
    enable_augmentations=True,
    max_augmentations_per_puzzle=30  # ← Faster!
)
```

---

## Summary

| Command | Purpose | Time |
|---------|---------|------|
| `python data/arc_augmented.py` | Quick test | <5s |
| `python test_augmentation.py` | Full test suite | ~30s |
| `python visualize_augmentations.py` | Generate images | ~60s |

All three scripts work **without training** and give you complete visibility into the augmentation pipeline!
