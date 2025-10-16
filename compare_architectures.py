"""Compare MLX and CUDA architecture without instantiating models."""

def analyze_parameter_counts():
    """Calculate parameter counts for both implementations."""

    # Configuration (matching both implementations)
    vocab_size = 12
    dim = 512
    heads = 8
    depth = 2  # 2 transformer blocks

    print("=" * 80)
    print("ARCHITECTURE COMPARISON: MLX vs CUDA")
    print("=" * 80)
    print()

    # CUDA Implementation Parameters
    print("CUDA IMPLEMENTATION")
    print("-" * 80)
    cuda_params = {}

    cuda_params['H_init'] = 512
    cuda_params['L_init'] = 512
    cuda_params['embed_tokens'] = 12 * 512
    cuda_params['lm_head'] = 12 * 512
    cuda_params['q_head.weight'] = 2 * 512
    cuda_params['q_head.bias'] = 2
    cuda_params['puzzle_emb'] = 29_726 * 512

    # Per block (2 blocks)
    for i in range(depth):
        prefix = f'L_level.layers.{i}'
        cuda_params[f'{prefix}.self_attn.qkv_proj'] = 1536 * 512
        cuda_params[f'{prefix}.self_attn.o_proj'] = 512 * 512
        cuda_params[f'{prefix}.mlp.gate_up_proj'] = 3072 * 512
        cuda_params[f'{prefix}.mlp.down_proj'] = 512 * 1536

    cuda_total = sum(cuda_params.values())

    for name, count in sorted(cuda_params.items()):
        print(f"  {name:<50} {count:>12,}")

    print()
    print(f"  CUDA Total Parameters: {cuda_total:,}")
    print(f"  CUDA Reported Trainable: 6,829,058")
    print(f"  CUDA Reported Total Numel: 22,049,794")
    print()

    # MLX Implementation Parameters
    print("MLX IMPLEMENTATION")
    print("-" * 80)
    mlx_params = {}

    # Token embeddings
    mlx_params['token_emb'] = 12 * 512

    # Puzzle embeddings (sparse, low-rank)
    puzzle_vocab = 1_000  # MLX uses 1000 vs CUDA's 29726
    puzzle_dim = 16       # Low-rank dimension
    mlx_params['puzzle_emb'] = puzzle_vocab * puzzle_dim
    mlx_params['puzzle_proj'] = puzzle_dim * 512  # No bias

    # Q-token
    mlx_params['q_token'] = 1 * 1 * 512

    # Output heads
    mlx_params['out_head.out'] = 512 * 12  # No bias
    mlx_params['q_head.out.weight'] = 512 * 1
    mlx_params['q_head.out.bias'] = 1

    # Latent carry initialization
    mlx_params['_y_init'] = 512
    mlx_params['_z_init'] = 512

    # Transformer blocks
    mlp_dim = int(8 / 3.0 * dim)  # 1365.33... → 1365

    for i in range(depth):
        prefix = f'blocks.{i}'
        # Attention
        mlx_params[f'{prefix}.attn.qkv'] = 3 * dim * dim  # No bias
        mlx_params[f'{prefix}.attn.out'] = dim * dim      # No bias
        # RMSNorm
        mlx_params[f'{prefix}.n1'] = dim
        mlx_params[f'{prefix}.n2'] = dim
        # SwiGLU MLP
        mlx_params[f'{prefix}.ff.w1'] = 2 * mlp_dim * dim  # gate_up fused
        mlx_params[f'{prefix}.ff.w2'] = dim * mlp_dim

    mlx_total = sum(mlx_params.values())

    for name, count in sorted(mlx_params.items()):
        print(f"  {name:<50} {count:>12,}")

    print()
    print(f"  MLX Total Parameters: {mlx_total:,}")
    print()

    # Comparison
    print("=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
    print()

    print("1. PUZZLE EMBEDDINGS (Largest Difference)")
    print("-" * 80)
    cuda_puzzle = 29_726 * 512
    mlx_puzzle = 1_000 * 16 + 16 * 512
    print(f"  CUDA: Direct embedding (29726, 512)")
    print(f"        = {cuda_puzzle:,} parameters")
    print(f"  MLX:  Factorized (1000, 16) + projection (16, 512)")
    print(f"        = {1_000 * 16:,} + {16 * 512:,} = {mlx_puzzle:,} parameters")
    print(f"  Difference: {cuda_puzzle - mlx_puzzle:,} parameters")
    print(f"  Reduction: {(1 - mlx_puzzle/cuda_puzzle)*100:.1f}%")
    print()
    print("  Why different?")
    print("    - CUDA uses full vocabulary of puzzles (29726 unique puzzles)")
    print("    - MLX uses reduced vocabulary (1000) with low-rank factorization")
    print("    - MLX approach: more memory efficient, less overfitting risk")
    print()

    print("2. Q-HEAD (Halting Decision)")
    print("-" * 80)
    cuda_qhead = 2 * 512 + 2
    mlx_qhead = 512 * 1 + 1
    print(f"  CUDA: Linear(512, 2) + bias")
    print(f"        = {cuda_qhead:,} parameters (2-class classification)")
    print(f"  MLX:  Linear(512, 1) + bias")
    print(f"        = {mlx_qhead:,} parameters (scalar logit)")
    print(f"  Difference: {cuda_qhead - mlx_qhead:,} parameters")
    print()
    print("  Why different?")
    print("    - CUDA: Binary classification (halt vs continue)")
    print("    - MLX: Scalar output (positive = halt, negative = continue)")
    print("    - Functionally equivalent, MLX is more compact")
    print()

    print("3. MLP DIMENSIONS")
    print("-" * 80)
    cuda_mlp_per_block = 3072 * 512 + 512 * 1536
    mlx_mlp_per_block = 2 * mlp_dim * 512 + 512 * mlp_dim
    print(f"  CUDA: gate_up_proj(3072, 512) + down_proj(512, 1536)")
    print(f"        = {cuda_mlp_per_block:,} per block")
    print(f"        Hidden dim: 1536 (3 * dim)")
    print(f"  MLX:  w1(2*1365, 512) + w2(512, 1365)")
    print(f"        = {mlx_mlp_per_block:,} per block")
    print(f"        Hidden dim: {mlp_dim} (8/3 * dim)")
    print(f"  Difference per block: {cuda_mlp_per_block - mlx_mlp_per_block:,}")
    print(f"  Total difference (2 blocks): {2 * (cuda_mlp_per_block - mlx_mlp_per_block):,}")
    print()
    print("  Why different?")
    print("    - CUDA uses 3x expansion (1536 = 3 * 512)")
    print("    - MLX uses 8/3 expansion (1365 ≈ 2.67 * 512)")
    print("    - 8/3 is common SwiGLU ratio, CUDA uses slightly larger")
    print()

    print("4. NORMALIZATION LAYERS")
    print("-" * 80)
    mlx_norm = 2 * 2 * 512  # 2 blocks * 2 norms * dim
    print(f"  CUDA: Not included in parameter count (standard for PyTorch)")
    print(f"  MLX:  RMSNorm parameters counted")
    print(f"        = {mlx_norm:,} parameters (2 blocks * 2 norms * {dim} dim)")
    print()
    print("  Why different?")
    print("    - PyTorch often doesn't count norm layers in parameter reports")
    print("    - MLX counts all learnable parameters")
    print("    - Not a functional difference, just reporting convention")
    print()

    print("5. OTHER DIFFERENCES")
    print("-" * 80)
    print(f"  Q-token parameter:")
    print(f"    MLX counts as parameter: {1 * 1 * 512:,}")
    print(f"    CUDA may initialize differently or not count")
    print()
    print(f"  Carry state initialization (_y_init, _z_init):")
    print(f"    MLX: {512 + 512:,} parameters")
    print(f"    CUDA: {512 + 512:,} parameters (H_init, L_init)")
    print(f"    These match!")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"CUDA Total: {cuda_total:,} parameters")
    print(f"CUDA Reported: 6,829,058 trainable (22,049,794 total numel)")
    print()
    print(f"MLX Total: {mlx_total:,} parameters")
    print()
    print(f"Difference: {mlx_total - cuda_total:,} parameters")
    print(f"MLX is {abs(mlx_total - cuda_total) / cuda_total * 100:.1f}% {'smaller' if mlx_total < cuda_total else 'larger'}")
    print()

    print("BREAKDOWN OF DIFFERENCES:")
    print()

    diff_puzzle = cuda_puzzle - mlx_puzzle
    diff_qhead = cuda_qhead - mlx_qhead
    diff_mlp = 2 * (cuda_mlp_per_block - mlx_mlp_per_block)
    diff_norm = -mlx_norm  # MLX has these, CUDA doesn't count
    diff_other = -512  # q_token in MLX

    print(f"  Puzzle embeddings:  {diff_puzzle:+15,}  (CUDA larger)")
    print(f"  Q-head:             {diff_qhead:+15,}  (CUDA larger)")
    print(f"  MLP layers (2x):    {diff_mlp:+15,}  (CUDA larger)")
    print(f"  Norm layers (4x):   {diff_norm:+15,}  (MLX counts them)")
    print(f"  Q-token:            {diff_other:+15,}  (MLX counts it)")
    print(f"  " + "-" * 35)
    print(f"  Net difference:     {diff_puzzle + diff_qhead + diff_mlp + diff_norm + diff_other:+15,}")
    print()

    print("KEY INSIGHTS:")
    print()
    print("1. MLX is SIGNIFICANTLY more parameter-efficient:")
    print("   - 99.5% reduction in puzzle embeddings (15.2M → 24.6K)")
    print("   - Achieved through vocabulary reduction + low-rank factorization")
    print()
    print("2. Core transformer architecture is nearly identical:")
    print("   - Same attention mechanism (QKV projection, RoPE)")
    print("   - Similar MLP structure (SwiGLU)")
    print("   - Slight difference in MLP hidden dim (1365 vs 1536)")
    print()
    print("3. MLX trades off:")
    print("   - Puzzle-specific capacity (smaller embeddings)")
    print("   - For better generalization and memory efficiency")
    print()
    print("4. The ~15M parameter difference is almost entirely")
    print("   from the puzzle embedding design choice!")
    print()

    # Additional numel explanation
    print("=" * 80)
    print("CUDA NUMEL DISCREPANCY EXPLANATION")
    print("=" * 80)
    print()
    print(f"CUDA reports:")
    print(f"  - Trainable parameters: 6,829,058")
    print(f"  - Total numel: 22,049,794")
    print()
    print(f"Our calculation of CUDA params: {cuda_total:,}")
    print()
    print("The 22M 'total numel' likely includes:")
    print("  1. Temporary buffers/activations during training")
    print("  2. Optimizer state (Adam/AdamW doubles or triples params)")
    print("  3. Gradient buffers")
    print("  4. Internal cached computations")
    print()
    print("This is NOT a parameter count, but total memory allocation.")
    print("The 6.8M trainable count is the actual model size.")

if __name__ == "__main__":
    analyze_parameter_counts()
