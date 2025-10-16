# MLX Tiny Recursive Models

> Experimental MLX implementation of Tiny Recursive Models (ARC + vision). Interfaces and performance are evolving; expect breaking changes.

Key files:
- Primary: `models/trm_arc.py` (ARC MLX model: embeddings, attention, blocks, recursion)
- Training loop: `training/trainer_arc.py` (loss, ACT loop, optimizer updates)
- Orchestration: `train_arc.py` and `train_arc_multi_opt.py` (CLI, data, call trainer)
- Vision model (non-ARC): `models/trm.py` Forked from [stockeh/mlx-trm](https://github.com/stockeh/mlx-trm) 

## Quick benchmarks

- Training throughput (multi-optimizer):
  ```bash
  python train_arc_multi_opt.py -b 32 \
    --dim 512 --depth 2 --n 6 --T 3 --ff-mult 3 \
    --bf16 --halt-max-steps 4 --throughput-samples 512
  ```
  (Disable fast RoPE with `--no-fast-rope` or revert to SwiGLU with `--no-mlx-ff` if needed.)

- HALT sweep:
  ```bash
  for h in 1 4 8 16; do
    python train_arc_multi_opt.py -b 32 \
      --dim 512 --depth 2 --n 6 --T 3 --ff-mult 3 \
      --bf16 --halt-max-steps $h --throughput-samples 512
  done
  ```

### Example training run (paper-like, quick experiment)

```bash
python train_arc_multi_opt.py --save my_arc_oct_16 \
  -b 16 --dim 512 --halt-max-steps 1 \
  --halt-exploration 0.2 \
  --bf16 -e 10 --val-freq 10 --puzzle-emb-lr 0.01 \
  --augment --num-aug 30
```

Notes:
- Uses multi-optimizer (SGD for sparse puzzle embeddings, AdamW for the model).
- `--bf16` matches the reference; `--num-aug 30` is a fast proxy for full training.
- Increase `--halt-max-steps` to 4/8/16 to study ACT cost/benefit.

Forked from [stockeh/mlx-trm](https://github.com/stockeh/mlx-trm) - [Original Tweet](https://x.com/itsstock/status/1977062337556214206)

Simplified reimplementation of [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) using [MLX](https://github.com/ml-explore/mlx).


Original implemnatation and paper
https://github.com/SamsungSAILMontreal/TinyRecursiveModels
https://arxiv.org/abs/2510.04871

## Usage

1. Setup the environment

   ```bash
   uv sync
   source .venv/bin/activate
   ```
2. Adjust model config in `train.py`

   ```python
   @dataclass
   class ModelConfig:
       in_channels: int
       depth: int
       dim: int
       heads: int
       patch_size: tuple
       n_outputs: int
       pool: str = "cls" # mean or cls
       n: int = 6  # latent steps
       T: int = 3  # deep steps
       halt_max_steps: int = 8  # maximum supervision steps
       halt_exploration_prob: float = 0.2  # exploratory q probability
       halt_follow_q: bool = True  # follow q (True) or max steps (False)
   ```
3. Train on MNIST or CIFAR-10 (see `python train.py --help`):
   ```bash
   python train.py --dataset mnist
   python train.py --dataset cifar10

   # With custom parameters
   python train.py --dataset cifar10 -e 50 -b 512 --lr 1e-3
   python train.py --dataset mnist --val-freq 5  # validate every 5 epochs

   # Save checkpoints during training
   python train.py --dataset cifar10 --save my_run
   # Creates: my_run/best_model.safetensors, my_run/checkpoint_epoch_N.safetensors,
   #          my_run/training_history.json, my_run/training_plot.png

   # Resume training from checkpoint (run 10 MORE epochs)
   python train.py -e 10 --resume my_run --save my_run_continued
   # Automatically uses saved dataset and batch_size, trains 10 additional epochs

   # Resume with different hyperparameters (overrides saved values)
   python train.py -e 5 --resume my_run --save my_run_v2 --dataset mnist -b 512 --lr 1e-4
   # Overrides saved values with new dataset, batch_size, and learning_rate
   ```

   **Available arguments:**
   - `--dataset` - Dataset to use (`mnist`, `cifar10`, `arc-sample`, or `arc-agi-1`)
   - `-b`, `--batch_size` - Batch size (default: 1024)
   - `-e`, `--epochs` - Number of epochs (default: 15)
   - `--lr` - Learning rate (default: 3e-4)
   - `--seed` - Random seed (default: 0)
   - `--cpu` - Use CPU only (default: use GPU/Metal)
   - `--val-freq` - Validation frequency in epochs (default: `max(1, epochs // 10)`)
   - `--save NAME` - Save checkpoints to folder NAME (saves best model + epoch checkpoints)
   - `--resume NAME` - Resume training from checkpoint folder NAME (loads weights and history, `-e` specifies additional epochs)

## ARC-AGI-1 Support

The Abstraction and Reasoning Corpus (ARC) is a challenging benchmark for AI reasoning. This repository includes a complete token-based implementation matching the original TinyRecursiveModels paper.

### Two Implementations Available

This repository provides **two separate implementations**:

| Feature | Vision Model (`train.py`) | ARC Model (`train_arc.py`) |
|---------|--------------------------|----------------------------|
| **Architecture** | Patch embeddings (2D vision) | Token embeddings (1D sequences) |
| **Use Case** | MNIST, CIFAR-10 | ARC-AGI tasks |
| **Input Format** | Images (3D tensors) | Flattened grids (1D sequences) |
| **Output** | Class labels | Output token sequences |
| **Embedding Type** | `PatchEmbedding` | `TokenEmbedding` |
| **Loss** | Cross-entropy (classification) | Seq2seq + Q-learning |
| **Metric** | Image accuracy | Token-level accuracy |

### ARC Data

The official ARC-AGI dataset is included in this repository:
```bash
# Already cloned in data/ARC-AGI/
ls data/ARC-AGI/data/training/   # 400 training tasks → 1302 examples
ls data/ARC-AGI/data/evaluation/ # 400 evaluation tasks → 1363 examples
```

Each task is a JSON file with:
- `"train"`: Multiple input/output demonstration pairs
- `"test"`: Test input/output pairs to solve
- Grids are 2D arrays with values 0-9 (colors)
- Grid sizes vary from 1x1 to 30x30
- Converted to 900-token sequences (30×30 flattened)

### Training ARC Models

Use the dedicated `train_arc.py` script with token-based architecture:

```bash
# Basic training (uses paper defaults: n=6, T=3, depth=2)
python train_arc.py -e 15 -b 32

# With data augmentation and puzzle IDs (matching TinyRecursiveModels)
python train_arc.py -e 50 -b 32 --augment --dim 512 --bf16
# Enables dihedral transformations, color permutations, and puzzle embeddings

# With custom learning rate and validation
python train_arc.py -e 50 -b 64 --lr 1e-4 --val-freq 5

# Save checkpoints
python train_arc.py -e 20 -b 32 --save my_arc_run
# Creates: my_arc_run/best_model.safetensors, checkpoint_epoch_N.safetensors,
#          training_history.json, training_plot.png

# Resume training (run 10 MORE epochs)
python train_arc.py -e 10 --resume my_arc_run --save my_arc_continued

# Paper defaults with BF16 (matches original implementation)
python train_arc.py -e 15 -b 32 --bf16
# Uses bfloat16 for 2x faster training and 50% less memory

# Speed testing with smaller model (faster training)
python train_arc.py -e 5 -b 64 --dim 256 --halt-max-steps 8 --bf16
# Smaller dim=256 reduces parameters, halt-max-steps=8 reduces computation

# Fast prototyping with minimal model
python train_arc.py -e 3 -b 128 --dim 128 --halt-max-steps 4 --halt-exploration 0.0 --bf16
# Tiny model for quick iteration

# Full training with augmentation (recommended)
python train_arc.py -e 100 -b 32 --augment --dim 512 --halt-max-steps 16 --bf16 --save arc_augmented
# Uses all augmentations and puzzle embeddings for best performance
```

**Available arguments for `train_arc.py`:**
- `-b`, `--batch_size` - Batch size (default: 32)
- `-e`, `--epochs` - Number of epochs (default: 15)
- `--lr` - Learning rate (default: 1e-4)
- `--seed` - Random seed (default: 0)
- `--cpu` - Use CPU only
- `--val-freq` - Validation frequency in epochs (default: `max(1, epochs // 10)`)
- `--save NAME` - Save checkpoints to folder NAME
- `--resume NAME` - Resume training from checkpoint folder NAME
- `--data-dir` - Path to ARC data directory (default: `data/ARC-AGI/data`)

**Model architecture arguments:**
- `--dim` - Model hidden dimension (default: 512)
- `--halt-max-steps` - Maximum adaptive computation steps (default: 16)
- `--halt-exploration` - Q-learning exploration probability (default: 0.1)
- `--bf16` - Use bfloat16 precision (paper default, faster training, 50% memory reduction)

**Data augmentation arguments (matching TinyRecursiveModels):**
- `--augment` - Enable data augmentations and puzzle IDs (recommended)
- `--puzzle-emb-dim` - Dimension for puzzle embeddings (default: 16)

### Hyperparameter Explanation

The ARC model uses hyperparameters matching the original TinyRecursiveModels paper for ARC-AGI-1:

| Parameter | Default | Paper Name | Description |
|-----------|---------|------------|-------------|
| `depth` | 2 | L_layers | Number of transformer blocks (one 2-layer network) |
| `T` | 3 | H_cycles | Deep recursion steps (with gradient stopping) |
| `n` | 6 | L_cycles | Latent recursion steps (per deep step) |
| `dim` | 512 | hidden_size | Model hidden dimension |
| `heads` | 8 | num_heads | Number of attention heads |
| `halt_max_steps` | 16 | - | Maximum adaptive computation steps |
| `halt_exploration_prob` | 0.1 | - | Q-learning exploration probability |

**Key recursive structure:**
- Each forward pass runs `T=3` deep recursion steps
- Each deep step runs `n=6` latent recursion steps
- Total recursion per pass: 3 × 6 = 18 latent updates
- Adaptive halting controlled by Q-learning (max 16 passes)

**Architecture details:**
- **Attention**: Multi-head self-attention with RoPE (Rotary Position Embedding)
- **Feedforward**: SwiGLU activation (expansion ratio 4)
- **Normalization**: RMSNorm
- **Token embeddings**: Scaled by sqrt(dim)
- **Positional encoding**: RoPE applied to queries and keys
- **Precision**: FP32 by default, BF16 with `--bf16` (paper uses BF16)

**Optimizer configuration (matching paper):**
- **Optimizer**: AdamW with β₁=0.9, β₂=0.95
- **Weight decay**: 0.01
- **Learning rate warmup**: 10% of total steps (min 10, max 2000)
  - Paper uses 2K steps, but ARC has only 1302 examples
  - We scale warmup to ~1.5 epochs for ARC dataset
- **LR schedule**: Linear warmup → Cosine decay to 0
- **EMA coefficient**: 0.999 for model parameters
- **Gradient clipping**: Max norm 1.0

The paper uses batch size 768, but we default to batch=32 for compatibility with consumer hardware.

### Data Augmentation and Puzzle Identifiers

This implementation includes advanced data augmentation and puzzle identifier embeddings matching the TinyRecursiveModels reference implementation.

#### Data Augmentations

When using `--augment`, the following augmentations are applied:

**1. Dihedral Transformations (8 geometric variations):**
- Identity (no change)
- 90°, 180°, 270° rotations
- Horizontal and vertical flips
- Diagonal and anti-diagonal flips

**2. Color Permutations:**
- Randomly permutes colors 1-9 (keeping black/0 unchanged)
- Preserves the logical structure while changing appearance
- Generates diverse color mappings for each puzzle

**3. Augmentation Strategy:**
- Training data can be expanded up to 100x with augmentations
- Each puzzle maintains its unique identifier across augmentations
- Augmentation details encoded in identifier string (e.g., "123|||d3_c0142356789")

#### Puzzle Identifier Embeddings

The model uses puzzle-specific embeddings to provide task context:

```python
# Puzzle embedding architecture:
- Separate embedding table: (num_puzzles, puzzle_emb_dim)
- Default dimension: 16 (configurable with --puzzle-emb-dim)
- Projection layer: Linear(16 → 512) to match model dimension
- Added to all token positions for task-specific context
```

**Benefits:**
- Model learns puzzle-specific patterns
- Improves generalization across similar tasks
- Enables per-puzzle performance tracking
- Matches TinyRecursiveModels approach for ARC

**Training with augmentation:**
```bash
# Standard dataset (1,302 training examples)
python train_arc.py -e 50 -b 32

# Augmented dataset (~13,000+ training examples)
python train_arc.py -e 50 -b 32 --augment

# With custom puzzle embedding dimension
python train_arc.py -e 50 -b 32 --augment --puzzle-emb-dim 32
```

### Model Architecture Details

**Token-based sequence modeling:**
```python
# Input: 30x30 grid → 900 tokens
# Vocabulary: {0-9: colors, 10: padding, 11: EOS} = 12 tokens
# Embedding: (batch, 900) → (batch, 901, 512)  # +1 for q_token
# Output: (batch, 900, 12) logits

# Components:
- TokenEmbedding: vocab_size=12, scaled embeddings (scaled by sqrt(512))
- PuzzleEmbedding: num_puzzles × 16, projected to model dimension
- Q-token: prepended for adaptive halting decisions
- Transformer blocks: RMSNorm + Attention(RoPE) + SwiGLU
- OutputHead: Linear(512 → 12) for token prediction
- QHead: Linear(512 → 1) for halt logits
```

**Loss function:**
```python
total_loss = ce_loss + 0.5 * bce_loss

# ce_loss: cross-entropy between predicted and target sequences
# bce_loss: Q-learning loss (halt when sequence is correct)
```

**Metrics:**
- **Token accuracy**: Percentage of correct tokens (ignoring padding)
- **Sequence accuracy**: Would require all tokens matching (very strict)
- Current results: ~3-4% token accuracy after initial epochs

### Evaluation System

The evaluation system implements **exact-match** evaluation identical to the original TinyRecursiveModels implementation, with full support for puzzle identifier tracking.

#### Key Concepts

**Training vs Evaluation Data:**
- **Training**: Uses `task["train"]` examples from 400 training tasks
  - Standard: ~1,302 examples
  - With augmentation: ~13,000+ examples (10x expansion)
- **Evaluation**: Uses `task["test"]` examples from 400 evaluation tasks → ~1,363 examples
- The model never sees test examples during training

**Puzzle Identifier Tracking:**
- Each unique puzzle gets a numeric identifier (1-400)
- Evaluation tracks per-puzzle performance metrics
- Identifies which specific puzzles the model struggles with
- Enables analysis of puzzle difficulty and model capabilities

**Exact Match via Hashing:**
```python
# Evaluation uses SHA256 hashing for exact grid comparison
def grid_hash(grid: np.ndarray) -> str:
    """Create a unique hash for a grid."""
    grid = np.asarray(grid, dtype=np.int32)
    return hashlib.sha256(grid.tobytes()).hexdigest()

# A prediction is correct ONLY if hash(predicted) == hash(ground_truth)
# Even a single incorrect token causes failure (100% accuracy required)
```

**Pass@K Metrics:**
- **Pass@1**: Model's first prediction must be exactly correct
- **Pass@2**: One of top-2 predictions must be correct
- **Pass@5**: One of top-5 predictions must be correct
- Uses confidence scores from Q-learning to rank predictions

#### Running Evaluation

Evaluate a trained model on the ARC test set:

```bash
# Basic evaluation (uses default model configuration)
python evaluate_arc.py checkpoints/best_model.safetensors

# With custom data directory
python evaluate_arc.py model.safetensors --data-dir path/to/arc/data

# Evaluate multiple K values
python evaluate_arc.py model.safetensors --pass-k 1 2 5 10

# Use bfloat16 for faster evaluation
python evaluate_arc.py model.safetensors --bf16

# Quiet mode (suppress detailed output)
python evaluate_arc.py model.safetensors --quiet
```

**Example output:**
```
Loading model from checkpoints/best_model.safetensors...
Model loaded. Parameters: 10,234,567
Loaded 419 test examples from evaluation

Evaluating on 419 test examples...
  Processing example 50/419...
  Processing example 100/419...

==================================================
Evaluation Results (419 test examples):
  Pass@1: 2.15%
  Pass@2: 3.10%
  Pass@5: 4.77%

==================================================
FINAL RESULTS:
==================================================
pass@1: 2.15%
pass@2: 3.10%
pass@5: 4.77%
```

#### Implementation Details

The evaluation system consists of four main components:

**1. Grid Hashing (`evaluators/arc.py:14-26`)**
- Converts grids to numpy int32 arrays for consistent representation
- Creates SHA256 hash from bytes representation
- Ensures deterministic comparison across platforms

**2. ARCEvaluator Class with Puzzle ID Support (`evaluators/arc_evaluator.py`)**
- Tracks predictions grouped by puzzle identifier
- Calculates per-puzzle exact match rates
- Identifies worst-performing puzzles for analysis
- Provides detailed puzzle-specific metrics

**3. ARCTestDataset Class (`evaluators/arc.py:249-313`)**
- Loads actual test examples from `task["test"]` (not training examples!)
- Includes demonstration pairs for reference (though current model doesn't use them)
- Converts JSON grids to numpy arrays

**4. Augmented Dataset with Puzzle IDs (`data/arc_augmented.py`)**
- Assigns unique identifiers to each puzzle
- Tracks augmentation transformations in ID strings
- Maintains puzzle ID consistency across augmentations
- Enables puzzle-specific learning and evaluation

#### Training vs TinyRecursiveModels Approach

**Our Implementation:**
- Trains on individual input→output pairs from `task["train"]`
- Does NOT use demonstration pairs for few-shot learning
- Treats ARC as supervised sequence-to-sequence prediction

**True ARC Intent:**
- Meant for few-shot learning: given 2-3 demonstration pairs, predict test output
- Should concatenate demonstrations as context for test prediction
- Current implementation ignores this aspect (matching TinyRecursiveModels)

**Future Work:**
Consider implementing true few-shot learning by:
1. Concatenating demonstration pairs as context
2. Using special tokens to separate examples
3. Predicting test output given demonstrations

### Testing with Sample Data

For quick pipeline testing with synthetic data:
```bash
python data/create_sample_arc.py  # Generate synthetic test data
python train.py --dataset arc-sample -e 5 -b 32  # Uses patch-based model
```

**Note**: `train.py --dataset arc-sample` uses the **vision model** (patch embeddings), not the proper token-based model. For real ARC training, always use `train_arc.py`.

## Notes

- Hyperparams are currently hardcoded for faster experimentation.
- Supported datasets: MNIST, CIFAR-10, and ARC-AGI-1 (with sample data generator).
- `uv` handles virtual environment creation automatically.

## CoreML/ANE export (optional)

This repository includes `export_arc_coreml.py`, a small PyTorch model that mimics the ARC token architecture and can be exported to CoreML for Apple Silicon ANE/GPU experiments.

### Install optional dependencies

These are not part of the default environment. Activate your venv and install:

```bash
source .venv/bin/activate
uv pip install torch coremltools
```

Verify the install:

```bash
python -c "import torch, coremltools as ct; print('torch', torch.__version__, 'coremltools', ct.__version__)"
```

### Export a CoreML model

By default the script creates a randomly-initialized ARC-like model and converts it to CoreML:

```bash
python export_arc_coreml.py \
  --output arc_model.mlpackage \
  --seq-len 900 \
  --dim 256 \
  --depth 2 \
  --heads 8 \
  --precision float16
```

Notes:
- Use `--precision float16` for iOS 16+ targets (preferred for ANE); use `--precision float32` for older targets.
- Disable puzzle ID embeddings with `--no-puzzle-ids` if desired; see `--puzzle-emb-dim` and `--num-puzzles` for configuration.
- To export learned weights, adapt the script to load a PyTorch checkpoint before conversion.

Troubleshooting:
- If you see `ModuleNotFoundError: No module named 'torch'` or `No module named 'coremltools'`, ensure the venv is active and reinstall the optional deps:
  ```bash
  source .venv/bin/activate
  uv pip install torch coremltools
  ```

## Benchmarks

### Dataset setup

- ARC-AGI dataset is already included in this repo at `data/ARC-AGI/data`.
- Verify presence:
  ```bash
  ls data/ARC-AGI/data/training | head
  ls data/ARC-AGI/data/evaluation | head
  ```
- Optional synthetic data for quick tests:
  ```bash
  python data/create_sample_arc.py
  # Then use --dataset arc-sample with train.py (vision baseline), or keep default for ARC using train_arc*.py
  ```

### Training throughput (MLX)

- Quick bench:
  ```bash
  python train_arc_multi_opt.py -b 32 \
    --dim 512 --depth 2 --n 6 --T 3 --ff-mult 3 \
    --bf16 --halt-max-steps 4 --throughput-samples 512
  ```

- Single optimizer (no saving/val, fixed samples):
  ```bash
  python train_arc.py -b 32 \
    --dim 512 --depth 2 --n 6 --T 3 --ff-mult 3 \
    --halt-max-steps 1 --bf16 --throughput-samples 512
  ```

- Multi-optimizer (SGD for puzzle_emb + AdamW for model):
  ```bash
  python train_arc_multi_opt.py -b 32 \
    --dim 512 --depth 2 --n 6 --T 3 --ff-mult 3 \
    --halt-max-steps 1 --bf16 --throughput-samples 512
  ```

- Sweep HALT (adaptive compute) impact:
  ```bash
  for h in 1 4 8 16; do
    python train_arc_multi_opt.py -b 32 \
      --dim 512 --depth 2 --n 6 --T 3 --ff-mult 3 \
      --bf16 --halt-max-steps $h --throughput-samples 512
  done
  ```

Notes:
- For ANE/CoreML parity during comparisons, you can use `--fp16` instead of `--bf16`.
- Set `--fixed-act` to run exactly `halt_max_steps` iterations without intermediate halting checks.
- Default behavior keeps gradients only on the last latent step for memory efficiency.

### Forward-only ANE/CoreML throughput

1) Export CoreML (FP16, set batch shape):
```bash
python export_arc_coreml.py \
  --output arc_model.mlpackage \
  --seq-len 900 --dim 256 --depth 2 --heads 8 \
  --precision float16 --batch 64
```

2) Benchmark on ANE/GPU:
```bash
python bench_coreml.py arc_model.mlpackage --batch 64 --seq-len 900 --warmup 5 --iters 50
```
