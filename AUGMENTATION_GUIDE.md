# Data Augmentation Guide for ARC-AGI Training

Based on [ARC Prize HRM Analysis](https://arcprize.org/blog/hrm-analysis).

## Overview

Data augmentation significantly improves model performance on ARC-AGI tasks by:
1. Increasing training dataset size
2. Teaching invariance to rotations, flips, and color permutations
3. Preventing overfitting

## Augmentation Settings

### `--num-aug` Parameter

Controls the maximum number of augmented versions per puzzle.

| Setting | Performance | Training Speed | Use Case |
|---------|-------------|----------------|----------|
| **30** | ~96% of max | Fastest | Quick experiments, prototyping |
| **100** | ~98% of max | Fast | Development, iteration |
| **300** | ~99.5% of max | Medium | **Recommended for most training** |
| **600** | ~99.9% of max | Slower | High-performance training |
| **1000** | 100% (max) | Slowest | Reproducing paper results |

### Key Findings from ARC Prize Analysis

> **300 augmentations achieves near-max performance** (within 0.5% of 1000 augmentations)
>
> **30 augmentations** (3% of paper's total) comes **within 4% of max performance**

This means you can train much faster with minimal performance loss!

## Augmentation Types

### 1. Dihedral Transformations (8 total)
- Identity (no change)
- 90° rotation
- 180° rotation
- 270° rotation
- Horizontal flip
- Vertical flip
- Diagonal flip (transpose)
- Anti-diagonal flip

### 2. Color Permutations
- Random shuffling of colors 1-9
- Black (0) always stays black
- Approximately 10 random color permutations per example

### Total Augmentations Per Example
With default settings: **1 (original) + 7 (dihedral) + ~10 (color) ≈ 18 augmentations**

With `--num-aug 300`, the dataset can generate up to 300 variations per puzzle, distributed across:
- Base examples from the task
- Dihedral transforms of each
- Color permutations of transforms

## Usage Examples

### Quick Experiment (Fast)
```bash
python train_arc_multi_opt.py \
  --augment \
  --num-aug 30 \
  -b 32 -e 20 --dim 128 --bf16
```
**Duration:** ~2-3 hours for 20 epochs
**Performance:** ~96% of max

### Recommended Training (Balanced)
```bash
python train_arc_multi_opt.py \
  --augment \
  --num-aug 300 \
  -b 32 -e 100 --dim 512 --bf16 \
  --halt-max-steps 16 --lr 0.0001
```
**Duration:** ~20-30 hours for 100 epochs
**Performance:** ~99.5% of max (near-optimal)

### High-Performance Training
```bash
python train_arc_multi_opt.py \
  --augment \
  --num-aug 600 \
  -b 32 -e 100 --dim 512 --bf16 \
  --halt-max-steps 16 --lr 0.0001
```
**Duration:** ~35-45 hours for 100 epochs
**Performance:** ~99.9% of max

### Paper Reproduction (Slowest)
```bash
python train_arc_multi_opt.py \
  --augment \
  --num-aug 1000 \
  -b 32 -e 100 --dim 512 --bf16 \
  --halt-max-steps 16 --lr 0.0001
```
**Duration:** ~50-60 hours for 100 epochs
**Performance:** 100% (max)

## Dataset Size Comparison

Without augmentation:
- **1,302 training examples** (400 tasks × ~3.3 examples each)

With augmentation (--num-aug 300):
- **~15,000-20,000 training examples**
- **~14× more data**

## Training Time Impact

Augmentation increases training time proportionally to dataset size:

| Config | Examples | Time/Epoch | 100 Epochs |
|--------|----------|------------|------------|
| No aug | 1,302 | ~1.3 min | ~2.2 hours |
| --num-aug 30 | ~5,000 | ~5 min | ~8 hours |
| --num-aug 300 | ~18,000 | ~18 min | ~30 hours |
| --num-aug 1000 | ~50,000 | ~50 min | ~83 hours |

## Recommendations

1. **For development/debugging:** Use `--num-aug 30`
   - Fast iteration
   - Still benefits from augmentation

2. **For serious training:** Use `--num-aug 300` (default)
   - Best performance/speed tradeoff
   - Recommended by ARC Prize analysis

3. **For paper reproduction:** Use `--num-aug 1000`
   - Maximum performance
   - Much slower training

## Performance Without Augmentation

Training without `--augment` flag leads to severe overfitting:
- Train accuracy: 80%+
- Test accuracy: 70%- (10+ point gap)
- Exact matches: 0/419
- Model memorizes training set instead of learning patterns

**Always use `--augment` for proper ARC training!**

## Implementation Details

Augmentations are generated on-the-fly during dataset loading:
- Each puzzle gets a consistent ID across all augmentations
- Model learns "this is the same puzzle, just transformed"
- Test set uses NO augmentation (only original puzzles)

See `data/arc_augmented.py` for implementation.

## References

- [ARC Prize HRM Analysis](https://arcprize.org/blog/hrm-analysis)
- [HRM Analysis GitHub](https://github.com/arcprize/hierarchical-reasoning-model-analysis)
- [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
