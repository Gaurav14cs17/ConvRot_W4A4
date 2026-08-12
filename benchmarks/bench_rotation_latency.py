"""
Benchmark: Rotation latency comparison.

Compares:
  - Group-wise RHT (ConvRot) at different group sizes
  - Standard Hadamard (Sylvester-type via FWHT if available)
  - Global matrix multiply rotation

Reproduces Table 5 / Figure 7 from the paper.
"""

import torch
import time
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convrot.convrot import group_rht, compute_outlier_amplitude
from convrot.hadamard import get_regular_hadamard


def benchmark_fn(fn, warmup=10, repeats=100):
    """Benchmark a function with warmup and timing."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed * 1000  # ms


def sylvester_hadamard(order):
    """Construct Sylvester-type Hadamard matrix."""
    import math
    H = torch.tensor([[1.0]])
    n = 1
    while n < order:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
        n *= 2
    return H / math.sqrt(order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4608, help="Batch size (tokens)")
    parser.add_argument("--K", type=int, default=15360, help="Feature dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    device = args.device
    dtype = getattr(torch, args.dtype)

    print(f"Rotation Latency Benchmark")
    print(f"{'=' * 60}")
    print(f"  Input shape: ({args.M}, {args.K})")
    print(f"  Device: {device}")
    print(f"  Dtype: {args.dtype}")
    print(f"{'=' * 60}")
    print()

    x = torch.randn(args.M, args.K, device=device, dtype=dtype)

    group_sizes = [16, 64, 256, 1024]
    results = []

    # Benchmark group-wise RHT
    print(f"{'Method':<30} {'Group Size':<12} {'Latency (ms)':<15} {'Outlier Amp':<12}")
    print(f"{'-' * 70}")

    original_amp = compute_outlier_amplitude(x)
    print(f"{'Original (no rotation)':<30} {'-':<12} {'-':<15} {original_amp:<12.2f}")

    for gs in group_sizes:
        if args.K % gs != 0:
            continue

        H = get_regular_hadamard(gs, device=device).to(dtype=dtype)

        def fn():
            return group_rht(x, group_size=gs, hadamard_matrix=H)

        latency = benchmark_fn(fn)
        rotated = fn()
        amp = compute_outlier_amplitude(rotated)
        reduction = (1 - amp / original_amp) * 100

        print(f"{'ConvRot (Regular)':<30} {gs:<12} {latency:<15.3f} {amp:<12.2f} ({reduction:+.0f}%)")
        results.append(("ConvRot", gs, latency, amp))

    # Benchmark Sylvester-type for comparison (only practical for smaller sizes)
    print()
    for gs in [64, 256]:
        if args.K % gs != 0:
            continue

        H_syl = sylvester_hadamard(gs).to(device=device, dtype=dtype)
        num_groups = args.K // gs

        def fn_syl():
            x_reshaped = x.view(args.M, num_groups, gs)
            return (x_reshaped @ H_syl.T).view(args.M, args.K)

        latency = benchmark_fn(fn_syl)
        rotated = fn_syl()
        amp = compute_outlier_amplitude(rotated)
        reduction = (1 - amp / original_amp) * 100

        print(f"{'Sylvester (Standard)':<30} {gs:<12} {latency:<15.3f} {amp:<12.2f} ({reduction:+.0f}%)")

    print(f"\n{'=' * 60}")
    print("Note: On RTX 4090, ConvRot N0=256 achieves ~1.55x speedup over FP16 baseline")


if __name__ == "__main__":
    main()
