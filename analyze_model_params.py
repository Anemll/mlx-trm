"""Analyze MLX model parameters and compare with CUDA implementation."""

import mlx.core as mx
from models.trm_arc import ARCModel, ARCModelConfig

def analyze_model():
    """Analyze MLX model parameters."""

    # Configuration matching CUDA implementation
    config = ARCModelConfig(
        vocab_size=12,  # 0-9 colors + padding + EOS
        max_seq_len=900,  # 30x30 grid
        depth=2,  # 2 transformer blocks (L_layers)
        dim=512,  # hidden dimension
        heads=8,
        n=6,  # latent recursion steps
        T=3,  # deep recursion steps
        halt_max_steps=16,
        halt_exploration_prob=0.1,
        num_puzzle_identifiers=1000,
        puzzle_emb_ndim=16,
        use_sparse_embeddings=True,
    )

    model = ARCModel(config)

    # Get all parameters
    params = model.parameters()

    print("MLX Model Architecture:")
    print("=" * 80)
    print(f"ARCModel(")
    print(f"  (embed): TokenEmbedding(")
    print(f"    (token_emb): Embedding({config.vocab_size}, {config.dim})")
    print(f"    (puzzle_emb): SparseEmbedding({config.num_puzzle_identifiers}, {config.puzzle_emb_ndim})")
    print(f"    (puzzle_proj): Linear({config.puzzle_emb_ndim}, {config.dim})")
    print(f"    (q_token): Parameter([1, 1, {config.dim}])")
    print(f"  )")
    print(f"  (blocks): Sequential(")
    for i in range(config.depth):
        print(f"    ({i}): Block(")
        print(f"      (n1): RMSNorm({config.dim})")
        print(f"      (attn): Attention(")
        print(f"        (qkv): Linear({config.dim}, {3 * config.dim})")
        print(f"        (out): Linear({config.dim}, {config.dim})")
        print(f"        (rope): RoPE({config.dim // config.heads})")
        print(f"      )")
        print(f"      (n2): RMSNorm({config.dim})")
        print(f"      (ff): SwiGLU(")
        mlp_dim = int(8 / 3.0 * config.dim)
        print(f"        (w1): Linear({config.dim}, {mlp_dim * 2})")
        print(f"        (w2): Linear({mlp_dim}, {config.dim})")
        print(f"      )")
        print(f"    )")
    print(f"  )")
    print(f"  (out_head): OutputHead(")
    print(f"    (out): Linear({config.dim}, {config.vocab_size})")
    print(f"  )")
    print(f"  (q_head): QHead(")
    print(f"    (out): Linear({config.dim}, 1)")
    print(f"  )")
    print(f"  (_y_init): Parameter([{config.dim}])")
    print(f"  (_z_init): Parameter([{config.dim}])")
    print(f")")
    print()

    # Count parameters
    total_params = 0
    trainable_params = 0

    print("Parameter shapes/sizes:")
    print("=" * 80)

    param_list = []
    for name, param in params.items():
        numel = param.size
        total_params += numel
        trainable_params += numel
        param_list.append((name, param.shape, numel))

    print(f"  Total tensors: {len(param_list)} | Total numel: {total_params:,}")

    # Sort by name for better readability
    param_list.sort(key=lambda x: x[0])

    for name, shape, numel in param_list:
        shape_str = f"shape={list(shape)}"
        print(f"  - {name:<55} {shape_str:<30} numel={numel:,}")

    print()
    print(f"Number of parameters: {trainable_params:,} (trainable: {trainable_params:,})")
    print()

    return model, config, trainable_params, total_params

def compare_with_cuda():
    """Compare MLX implementation with CUDA implementation."""

    print("\n" + "=" * 80)
    print("COMPARISON: MLX vs CUDA Implementation")
    print("=" * 80)
    print()

    # CUDA stats
    cuda_trainable = 6_829_058
    cuda_total_numel = 22_049_794
    cuda_tensors = 15

    # MLX stats
    model, config, mlx_trainable, mlx_total_numel = analyze_model()
    mlx_tensors = len(model.parameters())

    print("CUDA Implementation:")
    print(f"  Trainable parameters: {cuda_trainable:,}")
    print(f"  Total numel: {cuda_total_numel:,}")
    print(f"  Number of tensors: {cuda_tensors}")
    print()

    print("MLX Implementation:")
    print(f"  Trainable parameters: {mlx_trainable:,}")
    print(f"  Total numel: {mlx_total_numel:,}")
    print(f"  Number of tensors: {mlx_tensors}")
    print()

    print("Differences:")
    print(f"  Trainable parameters: {mlx_trainable - cuda_trainable:+,} ({(mlx_trainable / cuda_trainable - 1) * 100:+.2f}%)")
    print(f"  Total numel: {mlx_total_numel - cuda_total_numel:+,} ({(mlx_total_numel / cuda_total_numel - 1) * 100:+.2f}%)")
    print(f"  Number of tensors: {mlx_tensors - cuda_tensors:+}")
    print()

    # Detailed comparison
    print("=" * 80)
    print("DETAILED ARCHITECTURE COMPARISON")
    print("=" * 80)
    print()

    print("CUDA Model Structure:")
    print("""
  - H_init (512)                          -> MLX: _y_init
  - L_init (512)                          -> MLX: _z_init
  - embed_tokens (12, 512)                -> MLX: token_emb
  - lm_head (12, 512)                     -> MLX: out_head.out
  - q_head (2, 512) + bias (2)            -> MLX: q_head.out (DIFFERENT OUTPUT DIM)
  - puzzle_emb (29726, 512)               -> MLX: puzzle_emb (DIFFERENT SIZE)
  - 2x Transformer blocks:
      - qkv_proj (1536, 512)              -> MLX: qkv (3*dim, dim)
      - o_proj (512, 512)                 -> MLX: out
      - gate_up_proj (3072, 512)          -> MLX: w1 (2*mlp_dim, dim)
      - down_proj (512, 1536)             -> MLX: w2 (dim, mlp_dim)
    """)

    print("\nKEY DIFFERENCES:")
    print("-" * 80)
    print()

    # 1. Puzzle embeddings
    cuda_puzzle_size = 29_726 * 512
    mlx_puzzle_size = 1_000 * 16  # Before projection
    print(f"1. Puzzle Embeddings:")
    print(f"   CUDA: CastedSparseEmbedding(29726, 512) = {cuda_puzzle_size:,} params")
    print(f"   MLX:  SparseEmbedding(1000, 16) + Linear(16, 512)")
    print(f"         = {1000 * 16:,} + {16 * 512:,} = {1000 * 16 + 16 * 512:,} params")
    print(f"   Difference: {cuda_puzzle_size - (1000 * 16 + 16 * 512):,} params")
    print(f"   Reason: MLX uses smaller puzzle vocabulary (1000 vs 29726) and")
    print(f"           low-rank factorization (16 -> 512 vs direct 512)")
    print()

    # 2. Q-head
    print(f"2. Q-Head (Halting Decision):")
    print(f"   CUDA: Linear(512, 2) + bias = {512 * 2 + 2:,} params (binary classification)")
    print(f"   MLX:  Linear(512, 1) + bias = {512 * 1 + 1:,} params (scalar output)")
    print(f"   Difference: {(512 * 2 + 2) - (512 * 1 + 1):,} params")
    print(f"   Reason: MLX uses scalar halt logit, CUDA uses 2-class logits")
    print()

    # 3. MLP dimensions
    mlp_dim_mlx = int(8 / 3.0 * 512)
    mlp_dim_cuda = 1536
    print(f"3. MLP Dimensions (per block):")
    print(f"   CUDA: gate_up_proj (3072, 512) + down_proj (512, 1536)")
    print(f"         = {3072 * 512 + 512 * 1536:,} params")
    print(f"   MLX:  w1 (2*{mlp_dim_mlx}, 512) + w2 ({mlp_dim_mlx}, 512)")
    print(f"         = {2 * mlp_dim_mlx * 512 + mlp_dim_mlx * 512:,} params")
    print(f"   Difference per block: {(3072 * 512 + 512 * 1536) - (2 * mlp_dim_mlx * 512 + mlp_dim_mlx * 512):,} params")
    print(f"   Reason: MLX uses mlp_dim={mlp_dim_mlx} (8/3 * dim), CUDA uses 1536 (3 * dim)")
    print()

    # 4. RMSNorm parameters
    print(f"4. Normalization Layers:")
    print(f"   CUDA: Not shown in parameter list (likely not counting norm params)")
    print(f"   MLX:  RMSNorm layers ARE counted in parameters")
    print(f"         2 blocks * 2 norms * 512 dims = {2 * 2 * 512:,} params")
    print()

    # 5. Q-token
    print(f"5. Q-Token (Halting Token):")
    print(f"   CUDA: Not shown in parameter list")
    print(f"   MLX:  q_token (1, 1, 512) = 512 params")
    print(f"   Note: CUDA may initialize this differently or not count it")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("The MLX implementation has FEWER parameters mainly due to:")
    print()
    print("1. MUCH smaller puzzle embedding (factorized 1000x16 vs 29726x512)")
    print("   Savings: ~15.2M parameters")
    print()
    print("2. Smaller MLP dimensions (1365 vs 1536 intermediate dim)")
    print("   Savings: ~0.36M parameters per block")
    print()
    print("3. Simpler Q-head (scalar vs 2-class)")
    print("   Savings: ~514 parameters")
    print()
    print("The MLX implementation has MORE counted parameters due to:")
    print()
    print("1. RMSNorm parameters being counted (+2048 params)")
    print("2. Q-token being counted as parameter (+512 params)")
    print()
    print("Overall: MLX is more parameter-efficient while maintaining")
    print("similar architectural capacity for the core reasoning.")

if __name__ == "__main__":
    compare_with_cuda()
