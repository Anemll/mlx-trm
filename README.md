# MLX Tiny Recursive Models

Forked from:
https://github.com/stockeh/mlx-trm
https://x.com/itsstock/status/1977062337556214206

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
   ```

   **Available arguments:**
   - `--dataset` - Dataset to use (`mnist` or `cifar10`)
   - `-b`, `--batch_size` - Batch size (default: 1024)
   - `-e`, `--epochs` - Number of epochs (default: 15)
   - `--lr` - Learning rate (default: 3e-4)
   - `--seed` - Random seed (default: 0)
   - `--cpu` - Use CPU only (default: use GPU/Metal)
   - `--val-freq` - Validation frequency in epochs (default: `max(1, epochs // 10)`)

## Notes

- Hyperparams are currently hardcoded for faster experimentation.
- Only MNIST and CIFAR-10 are supported at the moment.
- `uv` handles virtual environment creation automatically.
