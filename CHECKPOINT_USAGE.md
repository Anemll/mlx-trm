# Checkpoint Management Guide

This guide explains how to resume training from checkpoints in the ARC-AGI training scripts.

## Available Checkpoints

When training, the following checkpoints are saved:

1. **`checkpoint_epoch_N.safetensors`** - Checkpoint saved every `val_freq` epochs
2. **`best_model.safetensors`** - Best model based on validation accuracy
3. **`training_history.json`** - Training metrics and epoch information

## Resume Training Methods

### Method 1: Resume from Latest Checkpoint (Default)

Resume from the last completed epoch:

```bash
python train_arc.py --resume my_arc_run -b 16 --dim 512 \
  --halt-max-steps 16 --bf16 -e 50 --val-freq 10
```

This will:
- Load `training_history.json` to find the last epoch
- Load `checkpoint_epoch_{last_epoch}.safetensors`
- Continue training from epoch `last_epoch + 1`

### Method 2: Resume from Specific Epoch (NEW)

Resume from a specific epoch checkpoint (e.g., epoch 180 where validation peaked):

```bash
python train_arc.py --resume my_arc_multi_512_EMB \
  --resume-from-epoch 180 \
  -b 16 --dim 512 --halt-max-steps 16 --bf16 -e 100 --val-freq 10
```

This will:
- Load `checkpoint_epoch_180.safetensors` specifically
- Truncate training history to epoch 180
- Continue training from epoch 181

**Use Case**: Your validation accuracy peaked at epoch 180 (44.19%) but training continued to epoch 368 (40.80%). Resume from epoch 180 to try different hyperparameters.

### Method 3: Start Fresh with Pretrained Weights

Copy checkpoint to new directory and start fresh training:

```bash
# Create new directory with checkpoint from epoch 180
mkdir my_arc_fresh_start
cp my_arc_multi_512_EMB/checkpoint_epoch_180.safetensors my_arc_fresh_start/checkpoint_epoch_0.safetensors

# Create minimal history file
echo '{"epoch": 0, "train_error_trace": [], "train_acc_trace": [], "val_error_trace": [], "val_acc_trace": [], "val_exact_match_trace": [], "val_epochs": []}' > my_arc_fresh_start/training_history.json

# Start training with different hyperparameters
python train_arc.py --resume my_arc_fresh_start \
  --save my_arc_fresh_start \
  -b 32 --dim 512 --halt-max-steps 16 \
  --halt-exploration 0.2 --bf16 -e 200 --val-freq 10
```

**Use Case**: Use pretrained weights but start new training with different learning rate, batch size, or exploration probability.

## Practical Examples

### Example 1: Recover from Overfitting

Your training shows overfitting after epoch 180:
- Epoch 180: 44.19% validation accuracy
- Epoch 368: 40.80% validation accuracy (dropped)

```bash
# Resume from best checkpoint and train with higher regularization
python train_arc.py --resume my_arc_multi_512_EMB \
  --resume-from-epoch 180 \
  --save my_arc_recovered \
  -b 16 --dim 512 --halt-max-steps 16 \
  --halt-exploration 0.2 \
  --weight-decay 0.15 \
  --bf16 -e 50 --val-freq 5
```

### Example 2: Try Different Halting Strategy

Your model has low halting probability (p_halt=0.02). Resume from a good checkpoint and increase exploration:

```bash
python train_arc.py --resume my_arc_multi_512_EMB \
  --resume-from-epoch 180 \
  --save my_arc_higher_exploration \
  -b 16 --dim 512 --halt-max-steps 16 \
  --halt-exploration 0.3 \
  --bf16 -e 100 --val-freq 10
```

### Example 3: Multi-Optimizer with Specific Epoch

```bash
python train_arc_multi_opt.py --resume my_arc_multi_512_EMB \
  --resume-from-epoch 180 \
  --save my_arc_multi_continued \
  -b 16 --dim 512 --halt-max-steps 16 \
  --puzzle-emb-lr 0.005 \
  --bf16 -e 50 --val-freq 10
```

## Finding Available Checkpoints

List all checkpoints in a directory:

```bash
ls -lh my_arc_multi_512_EMB/*.safetensors
```

Check which epoch had best validation:

```bash
# View training history
python -c "import json; print(json.load(open('my_arc_multi_512_EMB/training_history.json'))['val_epochs'])"
python -c "import json; h=json.load(open('my_arc_multi_512_EMB/training_history.json')); print(list(zip(h['val_epochs'], h['val_acc_trace'])))"
```

## Important Notes

1. **Matching Hyperparameters**: When resuming, you must use the same model architecture:
   - Same `--dim` (model dimension)
   - Same `--halt-max-steps`
   - Same `--puzzle-emb-dim`
   - Can change: learning rate, batch size, weight decay, exploration probability

2. **Checkpoint Files**: The trainer looks for `checkpoint_epoch_{N}.safetensors`. Make sure this file exists for the epoch you're resuming from.

3. **Training History**: When using `--resume-from-epoch`, the training history is automatically truncated to that epoch, so your plots will be consistent.

4. **Save Directory**: If you don't specify `--save`, checkpoints will be saved to the `--resume` directory (overwrites existing checkpoints).

## Troubleshooting

**Error: "Checkpoint not found: checkpoint_epoch_180.safetensors"**
- The checkpoint file doesn't exist. List available checkpoints with `ls my_arc_run/*.safetensors`

**Error: "Training history not found"**
- The `training_history.json` file is missing or corrupted

**Model architecture mismatch**
- You're trying to load a checkpoint with different model dimensions. Check the model config used during original training.
