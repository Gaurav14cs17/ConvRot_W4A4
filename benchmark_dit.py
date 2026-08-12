"""
Benchmark ConvRot on FLUX.1 (DiT) alongside torchao baselines.

Compares BF16, torchao int8/int4 weight-only, and ConvRot W8A8/W4A4
on FLUX.1-schnell or FLUX.1-dev. Measures latency, peak memory,
model size, and image quality (PSNR, SSIM, LPIPS).

Usage:
    python benchmark_dit.py --model flux-schnell --num-images 3
    python benchmark_dit.py --model flux-dev --num-images 5 --steps 50
    python benchmark_dit.py --model flux-schnell --methods bf16 convrot-w8a8 convrot-w4a4-mixed
"""

import torch
import torch.nn as nn
import time
import json
import argparse
import gc
import os
import sys
import math
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))
from convrot import convrot_quantize_, ConvRotConfig
from convrot.quantize import (
    get_model_size_mb, print_quantization_summary,
    FLUX_SKIP_LAYERS, FLUX_MIXED_PRECISION_LAYERS,
)
from convrot.conv_linear import ConvLinear8bit, ConvLinear4bit


PROMPTS = [
    "A professional photograph of a fluffy orange tabby cat sitting on a sunlit windowsill, shallow depth of field",
    "An oil painting of a medieval castle perched on a cliff overlooking the ocean at sunset, dramatic sky with golden clouds",
    "A photorealistic portrait of an astronaut in a lush garden, helmet visor reflecting flowers, cinematic lighting",
    "A macro photograph of dewdrops on a spider web at dawn, bokeh background with soft morning light",
    "A steampunk clockwork owl perched on a stack of old leather-bound books, detailed brass gears and glowing amber eyes",
]


ALL_METHODS = [
    "bf16",
    "torchao-int8wo",
    "torchao-int4wo",
    "torchao-fp8wo",
    "convrot-w8a8",
    "convrot-w4a4-mixed",
]


@dataclass
class MethodResult:
    method: str
    model_size_mb: float
    peak_memory_gb: float
    avg_latency_sec: float
    throughput_img_per_sec: float
    quantize_time_sec: float
    num_layers_quantized: int
    psnr_vs_bf16: float = 0.0
    ssim_vs_bf16: float = 0.0
    lpips_vs_bf16: float = -1.0
    images: list = field(default_factory=list)
    error: str = ""


def get_gpu_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        return name, vram
    return "CPU", 0.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.ndim == 3:
        return np.mean([compute_ssim(img1[:, :, c], img2[:, :, c]) for c in range(img1.shape[2])])
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq, sigma2_sq = img1.var(), img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return float(((2*mu1*mu2+C1)*(2*sigma12+C2)) / ((mu1**2+mu2**2+C1)*(sigma1_sq+sigma2_sq+C2)))


def compute_lpips_score(img1_pil, img2_pil, lpips_fn, device):
    if lpips_fn is None:
        return -1.0
    arr1 = np.array(img1_pil).astype(np.float32) / 255.0 * 2.0 - 1.0
    arr2 = np.array(img2_pil).astype(np.float32) / 255.0 * 2.0 - 1.0
    t1 = torch.from_numpy(arr1).permute(2, 0, 1).unsqueeze(0).to(device)
    t2 = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        return lpips_fn(t1, t2).item()


def load_flux_pipeline(model_name, device):
    """Load FLUX pipeline in BF16."""
    from diffusers import FluxPipeline

    model_map = {
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
    }
    model_id = model_map[model_name]
    print(f"  Loading {model_id}...")
    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.to(device)
    return pipe


def generate_images(pipe, prompts, num_steps, seed, model_name, device):
    """Generate images with the pipeline, return list of PIL images and per-image times."""
    images = []
    times = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device=device).manual_seed(seed + i)
        steps = 4 if "schnell" in model_name else num_steps

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()

        img = pipe(prompt=prompt, num_inference_steps=steps, generator=gen).images[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - t0

        images.append(img)
        times.append(elapsed)
        print(f"    [{i+1}/{len(prompts)}] {elapsed:.1f}s — {prompt[:50]}...")

    return images, times


def count_quantized_layers(model):
    n_conv8, n_conv4, n_linear = 0, 0, 0
    for m in model.modules():
        if isinstance(m, ConvLinear8bit):
            n_conv8 += 1
        elif isinstance(m, ConvLinear4bit):
            n_conv4 += 1
        elif isinstance(m, nn.Linear):
            n_linear += 1
    return n_conv8 + n_conv4


def apply_torchao_quantization(model, method):
    """Apply torchao quantization to the transformer."""
    from torchao.quantization import quantize_, int8_weight_only, int4_weight_only

    if method == "torchao-int8wo":
        print("  Applying torchao int8_weight_only...")
        quantize_(model, int8_weight_only())
    elif method == "torchao-int4wo":
        print("  Applying torchao int4_weight_only (group_size=128)...")
        quantize_(model, int4_weight_only(group_size=128))
    elif method == "torchao-fp8wo":
        print("  Applying torchao float8_weight_only...")
        try:
            from torchao.quantization import float8_weight_only
            quantize_(model, float8_weight_only())
        except Exception as e:
            # Fallback for newer torchao API names
            try:
                from torchao.quantization import Float8WeightOnlyConfig
                quantize_(model, Float8WeightOnlyConfig())
            except Exception:
                raise RuntimeError(
                    f"torchao FP8 weight-only not available in this install: {e}"
                ) from e
    else:
        raise ValueError(f"Unknown torchao method: {method}")


def apply_convrot_quantization(model, method, group_size):
    """Apply ConvRot quantization to the transformer."""
    skip_layers = list(FLUX_SKIP_LAYERS)
    mixed_layers = []

    if method == "convrot-w8a8":
        precision = "w8a8"
    elif method == "convrot-w4a4-mixed":
        precision = "w4a4"
        mixed_layers = list(FLUX_MIXED_PRECISION_LAYERS)
    else:
        raise ValueError(f"Unknown ConvRot method: {method}")

    config = ConvRotConfig(
        precision=precision,
        group_size=group_size,
        mixed_precision_layers=mixed_layers,
        skip_layers=skip_layers,
    )

    print(f"  Applying ConvRot {precision.upper()}" +
          (f" + mixed precision ({len(mixed_layers)} patterns)" if mixed_layers else "") +
          f" (group_size={group_size})...")
    convrot_quantize_(model, config)
    print_quantization_summary(model, config)


def run_single_method(
    method, model_name, prompts, num_steps, seed, device, group_size, images_dir
):
    """Run a single quantization method end-to-end. Returns MethodResult."""
    print(f"\n{'='*70}")
    print(f"  METHOD: {method}")
    print(f"{'='*70}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

    result = MethodResult(
        method=method, model_size_mb=0, peak_memory_gb=0,
        avg_latency_sec=0, throughput_img_per_sec=0,
        quantize_time_sec=0, num_layers_quantized=0,
    )

    pipe = None
    try:
        pipe = load_flux_pipeline(model_name, device)
        transformer = pipe.transformer

        size_before = get_model_size_mb(transformer)
        print(f"  Transformer size (BF16): {size_before:.1f} MB")

        t_quant_start = time.time()

        if method == "bf16":
            pass
        elif method.startswith("torchao-"):
            apply_torchao_quantization(transformer, method)
        elif method.startswith("convrot-"):
            apply_convrot_quantization(transformer, method, group_size)

        quant_time = time.time() - t_quant_start
        result.quantize_time_sec = quant_time

        size_after = get_model_size_mb(transformer)
        result.model_size_mb = size_after
        result.num_layers_quantized = count_quantized_layers(transformer) if method.startswith("convrot-") else 0

        print(f"  Size after quant: {size_after:.1f} MB ({size_before/size_after:.2f}x reduction)")
        print(f"  Quantization time: {quant_time:.1f}s")

        # Warmup
        print("  Warmup run...")
        gen = torch.Generator(device=device).manual_seed(0)
        steps = 4 if "schnell" in model_name else num_steps
        _ = pipe(prompt="warmup", num_inference_steps=steps, generator=gen).images[0]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Generate
        print(f"  Generating {len(prompts)} images...")
        images, times = generate_images(pipe, prompts, num_steps, seed, model_name, device)

        # Save images
        method_images = []
        for i, img in enumerate(images):
            safe_method = method.replace("/", "-")
            img_path = images_dir / f"{safe_method}_{i:03d}.png"
            img.save(img_path, quality=95)
            method_images.append(str(img_path))
        result.images = method_images

        avg_time = sum(times) / len(times)
        result.avg_latency_sec = avg_time
        result.throughput_img_per_sec = 1.0 / avg_time if avg_time > 0 else 0

        if torch.cuda.is_available():
            result.peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        print(f"  Avg latency: {avg_time:.2f}s/image")
        print(f"  Throughput: {result.throughput_img_per_sec:.3f} img/s")
        print(f"  Peak memory: {result.peak_memory_gb:.2f} GB")

    except Exception as e:
        result.error = f"{type(e).__name__}: {str(e)}"
        print(f"  ERROR: {result.error}")
        traceback.print_exc()

    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def compute_quality_metrics(results, bf16_result, images_dir, device):
    """Compute PSNR/SSIM/LPIPS for each method vs BF16 baseline."""
    if not bf16_result.images:
        print("  No BF16 images to compare against, skipping quality metrics")
        return

    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        print("  LPIPS metric: enabled")
    except ImportError:
        print("  LPIPS metric: disabled (pip install lpips)")

    bf16_pil = [Image.open(p) for p in bf16_result.images]

    for res in results:
        if res.method == "bf16" or res.error or not res.images:
            continue

        psnr_scores, ssim_scores, lpips_scores = [], [], []
        for i, img_path in enumerate(res.images):
            if i >= len(bf16_pil):
                break
            img_q = Image.open(img_path)
            img_bf16 = bf16_pil[i]

            arr_bf16 = np.array(img_bf16)
            arr_q = np.array(img_q)
            if arr_bf16.shape != arr_q.shape:
                img_q = img_q.resize(img_bf16.size, Image.LANCZOS)
                arr_q = np.array(img_q)

            psnr_scores.append(compute_psnr(arr_bf16, arr_q))
            ssim_scores.append(compute_ssim(arr_bf16, arr_q))
            lp = compute_lpips_score(img_bf16, img_q, lpips_fn, device)
            if lp >= 0:
                lpips_scores.append(lp)

        if psnr_scores:
            res.psnr_vs_bf16 = float(np.mean(psnr_scores))
        if ssim_scores:
            res.ssim_vs_bf16 = float(np.mean(ssim_scores))
        if lpips_scores:
            res.lpips_vs_bf16 = float(np.mean(lpips_scores))

        print(f"  {res.method:>25}: PSNR={res.psnr_vs_bf16:.2f} dB, SSIM={res.ssim_vs_bf16:.4f}" +
              (f", LPIPS={res.lpips_vs_bf16:.4f}" if res.lpips_vs_bf16 >= 0 else ""))


def create_comparison_grid(results, num_images, output_path):
    """Create a visual comparison grid: columns=methods, rows=prompts."""
    valid = [r for r in results if not r.error and r.images]
    if not valid:
        print("  No valid results to create grid")
        return

    img_size = 512
    cols = len(valid)
    rows = min(num_images, min(len(r.images) for r in valid))

    header_height = 60
    label_width = 50
    grid_w = label_width + cols * img_size
    grid_h = header_height + rows * img_size
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    # Headers
    for col, res in enumerate(valid):
        label = res.method.upper()
        if res.method == "bf16":
            label = "BF16 (Baseline)"
        x = label_width + col * img_size + img_size // 2
        draw.text((x, 8), label, fill="black", anchor="mt", font=font)
        info = f"{res.model_size_mb:.0f}MB | {res.avg_latency_sec:.1f}s | {res.peak_memory_gb:.1f}GB"
        draw.text((x, 32), info, fill="gray", anchor="mt", font=font_small)

    # Images
    for row in range(rows):
        y_center = header_height + row * img_size + img_size // 2
        draw.text((5, y_center), f"#{row+1}", fill="gray", anchor="lm", font=font_small)

        for col, res in enumerate(valid):
            if row < len(res.images):
                img = Image.open(res.images[row]).resize((img_size, img_size), Image.LANCZOS)
                grid.paste(img, (label_width + col * img_size, header_height + row * img_size))

    grid.save(output_path, quality=95)
    print(f"  Grid saved: {output_path}")


def print_results_table(results, model_name):
    """Print a formatted results table to stdout."""
    valid = [r for r in results if not r.error]
    if not valid:
        return

    bf16 = next((r for r in valid if r.method == "bf16"), None)
    bf16_size = bf16.model_size_mb if bf16 else 1.0
    bf16_latency = bf16.avg_latency_sec if bf16 else 1.0
    bf16_memory = bf16.peak_memory_gb if bf16 else 1.0

    print(f"\n{'='*100}")
    print(f"  BENCHMARK RESULTS — {model_name.upper()}")
    print(f"{'='*100}")

    # Performance table
    print(f"\n  {'Method':<25} {'Size (MB)':>10} {'Reduction':>10} {'Latency':>10} {'Speedup':>10} {'Peak Mem':>10} {'Mem Save':>10} {'Quant(s)':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
    for r in valid:
        reduction = f"{bf16_size/r.model_size_mb:.2f}x" if r.model_size_mb > 0 else "—"
        speedup = f"{bf16_latency/r.avg_latency_sec:.2f}x" if r.avg_latency_sec > 0 else "—"
        mem_save = f"{bf16_memory/r.peak_memory_gb:.2f}x" if r.peak_memory_gb > 0 else "—"
        print(f"  {r.method:<25} {r.model_size_mb:>9.0f}  {reduction:>10} {r.avg_latency_sec:>9.2f}s {speedup:>10} {r.peak_memory_gb:>9.2f}G {mem_save:>10} {r.quantize_time_sec:>9.1f}")

    # Quality table
    has_quality = any(r.psnr_vs_bf16 > 0 for r in valid if r.method != "bf16")
    if has_quality:
        print(f"\n  {'Method':<25} {'PSNR↑ (dB)':>12} {'SSIM↑':>10} {'LPIPS↓':>10} {'Verdict':>15}")
        print(f"  {'─'*25} {'─'*12} {'─'*10} {'─'*10} {'─'*15}")
        for r in valid:
            if r.method == "bf16":
                print(f"  {r.method:<25} {'(reference)':>12} {'—':>10} {'—':>10} {'baseline':>15}")
            elif r.psnr_vs_bf16 > 0:
                verdict = "excellent" if r.psnr_vs_bf16 > 25 else ("good" if r.psnr_vs_bf16 > 20 else ("acceptable" if r.psnr_vs_bf16 > 15 else "degraded"))
                lpips_str = f"{r.lpips_vs_bf16:.4f}" if r.lpips_vs_bf16 >= 0 else "N/A"
                print(f"  {r.method:<25} {r.psnr_vs_bf16:>11.2f}  {r.ssim_vs_bf16:>9.4f}  {lpips_str:>10} {verdict:>15}")

    # Errored methods
    errored = [r for r in results if r.error]
    if errored:
        print(f"\n  FAILED METHODS:")
        for r in errored:
            print(f"    {r.method}: {r.error}")


def main():
    parser = argparse.ArgumentParser(
        description="ConvRot DiT Benchmark — FLUX.1 with torchao Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="flux-schnell",
                        choices=["flux-schnell", "flux-dev"],
                        help="FLUX model (DiT architecture)")
    parser.add_argument("--methods", type=str, nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS,
                        help="Quantization methods to benchmark")
    parser.add_argument("--group-size", type=int, default=256,
                        help="ConvRot Hadamard rotation group size (paper default: 256)")
    parser.add_argument("--num-images", type=int, default=3,
                        help="Number of images per method")
    parser.add_argument("--steps", type=int, default=50,
                        help="Diffusion steps (ignored for schnell, which uses 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/dit_benchmark")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name, gpu_vram = get_gpu_info()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images" / args.model
    images_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report").mkdir(parents=True, exist_ok=True)

    prompts = PROMPTS[:args.num_images]
    effective_steps = 4 if "schnell" in args.model else args.steps

    print("=" * 70)
    print("  ConvRot DiT Benchmark — FLUX.1 with torchao Comparison")
    print("=" * 70)
    print(f"  Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model:      {args.model} (Diffusion Transformer)")
    print(f"  GPU:        {gpu_name} ({gpu_vram:.1f} GB)")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  Steps:      {effective_steps}")
    print(f"  Images:     {args.num_images}")
    print(f"  Group size: {args.group_size} (ConvRot)")
    print(f"  Methods:    {', '.join(args.methods)}")
    print("=" * 70)

    # Ensure bf16 is always first (needed as quality reference)
    methods = list(args.methods)
    if "bf16" not in methods:
        methods.insert(0, "bf16")
    elif methods[0] != "bf16":
        methods.remove("bf16")
        methods.insert(0, "bf16")

    # Run each method
    results = []
    for method in methods:
        res = run_single_method(
            method=method,
            model_name=args.model,
            prompts=prompts,
            num_steps=args.steps,
            seed=args.seed,
            device=device,
            group_size=args.group_size,
            images_dir=images_dir,
        )
        results.append(res)

    # Quality metrics
    print(f"\n{'='*70}")
    print(f"  Computing Quality Metrics (vs BF16)")
    print(f"{'='*70}")
    bf16_result = next((r for r in results if r.method == "bf16" and not r.error), None)
    if bf16_result:
        compute_quality_metrics(results, bf16_result, images_dir, device)

    # Comparison grid
    print(f"\n{'='*70}")
    print(f"  Creating Comparison Grid")
    print(f"{'='*70}")
    grid_path = output_dir / f"comparison_{args.model}.png"
    create_comparison_grid(results, args.num_images, grid_path)

    # Print results
    print_results_table(results, args.model)

    # Save JSON report
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "architecture": "DiT (Diffusion Transformer)",
        "gpu": gpu_name,
        "gpu_vram_gb": gpu_vram,
        "torch_version": torch.__version__,
        "num_images": args.num_images,
        "num_steps": effective_steps,
        "group_size": args.group_size,
        "methods": [asdict(r) for r in results],
    }

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / "report" / f"dit_benchmark_{args.model}_{ts}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  OUTPUT FILES:")
    print(f"    Images:  {images_dir}/")
    print(f"    Grid:    {grid_path}")
    print(f"    Report:  {report_path}")
    print(f"\n{'='*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
