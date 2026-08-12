"""
Benchmark: ConvLinear layer latency and memory.

Compares:
  - nn.Linear (BF16)
  - ConvLinear8bit (W8A8)
  - ConvLinear4bit (W4A4)

Reports latency and memory usage for typical FLUX layer dimensions.
"""

import torch
import torch.nn as nn
import time
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convrot.conv_linear import ConvLinear8bit, ConvLinear4bit
from convrot.quantize import get_model_size_mb


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


def get_param_bytes(module):
    """Get total parameter + buffer bytes."""
    total = 0
    for p in module.parameters():
        total += p.nelement() * p.element_size()
    for b in module.buffers():
        total += b.nelement() * b.element_size()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    # FLUX.1-dev typical layer dimensions
    layer_configs = [
        ("Attention QKV", 3072, 3072),
        ("Attention Out", 3072, 3072),
        ("FFN Up", 3072, 15360),
        ("FFN Down", 15360, 3072),
        ("Single Block Proj", 15360, 3072),
    ]

    batch_size = 4608  # Typical token count for 1024x1024 image

    print(f"ConvLinear Layer Benchmark")
    print(f"{'=' * 90}")
    print(f"  Device: {device}, Batch: {batch_size}")
    print(f"{'=' * 90}")
    print()
    print(f"{'Layer':<20} {'(M,K,N)':<20} {'BF16 (ms)':<12} {'W8A8 (ms)':<12} {'W4A4 (ms)':<12} {'BF16 MB':<10} {'W8A8 MB':<10} {'W4A4 MB':<10}")
    print(f"{'-' * 106}")

    for name, K, N in layer_configs:
        M = batch_size

        # BF16 baseline
        linear = nn.Linear(K, N, bias=False, dtype=dtype, device=device)
        x = torch.randn(M, K, device=device, dtype=dtype)

        lat_bf16 = benchmark_fn(lambda: linear(x))
        mem_bf16 = get_param_bytes(linear) / (1024 * 1024)

        # Determine valid group size
        gs = 256 if K % 256 == 0 else 64 if K % 64 == 0 else 16

        # W8A8
        conv8 = ConvLinear8bit.from_linear(linear.cpu(), group_size=gs).to(device)
        lat_w8a8 = benchmark_fn(lambda: conv8(x))
        mem_w8a8 = get_param_bytes(conv8) / (1024 * 1024)

        # W4A4
        wgs = 128 if K % 128 == 0 else 64
        conv4 = ConvLinear4bit.from_linear(linear.cpu(), group_size=gs, weight_group_size=wgs).to(device)
        lat_w4a4 = benchmark_fn(lambda: conv4(x))
        mem_w4a4 = get_param_bytes(conv4) / (1024 * 1024)

        shape_str = f"({M},{K},{N})"
        print(f"{name:<20} {shape_str:<20} {lat_bf16:<12.2f} {lat_w8a8:<12.2f} {lat_w4a4:<12.2f} {mem_bf16:<10.1f} {mem_w8a8:<10.1f} {mem_w4a4:<10.1f}")

    print(f"\n{'=' * 90}")
    print("Note: True INT4 speedup requires specialized CUDA kernels (cutlass/torchao).")
    print("Current implementation demonstrates correct numerics; kernel optimization is future work.")


if __name__ == "__main__":
    main()
