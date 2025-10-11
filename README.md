# MLX Tiny Recursive Models

Forked from [stockeh/mlx-trm](https://github.com/stockeh/mlx-trm) - [Original Tweet](https://x.com/itsstock/status/1977062337556214206)

Simplified reimplementation of [TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels) using [MLX](https://github.com/ml-explore/mlx).

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
   - `--dataset` - Dataset to use (`mnist` or `cifar10`, auto-loaded from checkpoint if resuming)
   - `-b`, `--batch_size` - Batch size (default: 1024, auto-loaded from checkpoint if resuming)
   - `-e`, `--epochs` - Number of epochs (default: 15)
   - `--lr` - Learning rate (default: 3e-4)
   - `--seed` - Random seed (default: 0)
   - `--cpu` - Use CPU only (default: use GPU/Metal)
   - `--val-freq` - Validation frequency in epochs (default: `max(1, epochs // 10)`)
   - `--save NAME` - Save checkpoints to folder NAME (saves best model + epoch checkpoints + metadata)
   - `--resume NAME` - Resume training from checkpoint folder NAME (auto-loads dataset/batch_size/model_config, CLI args override)

## Notes

- Hyperparams are currently hardcoded for faster experimentation.
- Only MNIST and CIFAR-10 are supported at the moment.
- `uv` handles virtual environment creation automatically.
