import argparse
import time

import coremltools as ct
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Benchmark CoreML ARC forward throughput")
    parser.add_argument("model", type=str, help="Path to .mlpackage or .mlmodel")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=900)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--use-puzzle-ids", action="store_true")
    args = parser.parse_args()

    mlmodel = ct.models.MLModel(args.model)

    token_ids = np.random.randint(0, 12, size=(args.batch, args.seq_len), dtype=np.int64)
    if args.use_puzzle_ids:
        puzzle_ids = np.random.randint(0, 1000, size=(args.batch,), dtype=np.int64)
        inputs = {"token_ids": token_ids, "puzzle_ids": puzzle_ids}
    else:
        inputs = {"token_ids": token_ids}

    # Warmup
    for _ in range(args.warmup):
        _ = mlmodel.predict(inputs, useCPUOnly=False)

    start = time.perf_counter()
    total = 0
    for _ in range(args.iters):
        _ = mlmodel.predict(inputs, useCPUOnly=False)
        total += args.batch
    elapsed = time.perf_counter() - start

    sps = total / elapsed
    print(f"Throughput (CoreML): {sps:.2f} samples/s  (batch={args.batch}, seq={args.seq_len})")


if __name__ == "__main__":
    main()


