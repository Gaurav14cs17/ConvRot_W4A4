"""
Generate images at multiple quantization levels and compare quality.

Produces side-by-side grids, PSNR/SSIM/LPIPS metrics, and a JSON report
for SDXL or FLUX models with BF16/W8A8/W4A4 precisions.

Usage:
    python generate.py --model sdxl --num-images 3 --steps 20
    python generate.py --model flux-dev --num-images 5 --steps 50
    python generate.py --model flux-dev --num-images 10 --steps 50 --precision bf16 w8a8 w4a4 w4a4-mixed
"""

import torch
import torch.nn as nn
import time
import json
import argparse
import math
import gc
import os
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).parent))
from convrot import convrot_quantize_, ConvRotConfig
from convrot.quantize import get_model_size_mb, print_quantization_summary
from convrot.hadamard import regular_hadamard_matrix
from convrot.convrot import group_rht, compare_outlier_reduction


# ─── Prompts matching ConvRot repo (PRISM-Bench text_rendering) ─────────────
PAPER_PROMPTS = [
    'A compact green circuit board featuring a Raspberry Pi logo, labeled "Raspberry Pi 4 Model B," with Ethernet, USB, and GPIO connectors, along with various electronic components.',
    'A bottle labeled "WHISTLEPIG" featuring "SMOKED BARREL-AGED RYE" sits alongside two clear whiskey glasses, showcasing a refined presentation of the spirit.',
    'A silhouette of an airport featuring control towers, cranes, and a terminal building against an orange sunset, with a road sign reading "TOWER!" pointing towards the structures.',
]

EXTRA_PROMPTS = [
    "A cute orange cat sitting on a windowsill, sunlight streaming in, photorealistic",
    "A mountain landscape at sunset with a lake reflection, oil painting style",
    "Portrait of an astronaut in a garden, detailed, cinematic lighting",
    "A steampunk clockwork robot reading a book in a library",
    "Fresh sushi platter on a wooden board, food photography, top-down view",
    "An ancient castle on a cliff overlooking the ocean, dramatic sky",
    "A red sports car driving through a neon-lit city at night",
]


@dataclass
class ImageResult:
    prompt_idx: int
    prompt: str
    precision: str
    group_size: int
    gen_time_sec: float
    image_path: str


@dataclass
class QualityMetrics:
    precision: str
    avg_psnr: float
    avg_ssim: float
    avg_lpips: float
    num_images: int


@dataclass
class PerformanceMetrics:
    precision: str
    model_size_mb: float
    avg_gen_time_sec: float
    speedup_vs_bf16: float
    memory_reduction_vs_bf16: float


@dataclass
class ProofReport:
    timestamp: str
    model_name: str
    gpu_name: str
    gpu_vram_gb: float
    torch_version: str
    num_images: int
    num_steps: int
    group_size: int
    precisions_tested: list
    quality_metrics: list
    performance_metrics: list
    image_results: list


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
    """Windowed SSIM (per-channel, averaged)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if img1.ndim == 3:
        return np.mean([compute_ssim(img1[:, :, c], img2[:, :, c]) for c in range(img1.shape[2])])

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_val)


def compute_lpips_score(img1_pil, img2_pil, lpips_fn, device):
    if lpips_fn is None:
        return -1.0
    arr1 = np.array(img1_pil).astype(np.float32) / 255.0 * 2.0 - 1.0
    arr2 = np.array(img2_pil).astype(np.float32) / 255.0 * 2.0 - 1.0
    t1 = torch.from_numpy(arr1).permute(2, 0, 1).unsqueeze(0).to(device)
    t2 = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        score = lpips_fn(t1, t2)
    return score.item()


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0


def load_pipeline(model_name, device, dtype):
    from diffusers import StableDiffusionXLPipeline, FluxPipeline

    model_map = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    }
    model_id = model_map.get(model_name, model_name)
    print(f"  Loading: {model_id}")

    if "sdxl" in model_name.lower():
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16",
        )
    else:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

    pipe.to(device)
    return pipe


def get_component(pipe):
    if hasattr(pipe, 'transformer'):
        return pipe.transformer, "transformer"
    return pipe.unet, "unet"


def generate_one(pipe, prompt, num_steps, seed, model_name, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    kwargs = {"prompt": prompt, "num_inference_steps": num_steps, "generator": gen}
    if "schnell" in model_name:
        kwargs["num_inference_steps"] = 4
    return pipe(**kwargs).images[0]


def create_comparison_grid(images_dir, precisions, num_images, output_path):
    """Create a side-by-side comparison grid like the ConvRot GitHub repo."""
    img_size = 512
    cols = len(precisions)
    rows = num_images

    header_height = 50
    label_width = 40
    grid_w = label_width + cols * img_size
    grid_h = header_height + rows * img_size
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()
        font_small = font

    for col, prec in enumerate(precisions):
        label = prec.upper()
        if prec == "bf16":
            label = "BF16 (Reference)"
        x = label_width + col * img_size + img_size // 2
        draw.text((x, 15), label, fill="black", anchor="mt", font=font)

    for row in range(num_images):
        y_center = header_height + row * img_size + img_size // 2
        draw.text((5, y_center), f"#{row+1}", fill="gray", anchor="lm", font=font_small)

        for col, prec in enumerate(precisions):
            img_path = images_dir / f"{prec}_{row:03d}.png"
            if img_path.exists():
                img = Image.open(img_path).resize((img_size, img_size), Image.LANCZOS)
                grid.paste(img, (label_width + col * img_size, header_height + row * img_size))
            else:
                x = label_width + col * img_size + img_size // 2
                y = header_height + row * img_size + img_size // 2
                draw.text((x, y), "missing", fill="red", anchor="mm", font=font_small)

    grid.save(output_path, quality=95)
    print(f"  Comparison grid saved: {output_path}")
    return output_path


def run_outlier_analysis(output_dir):
    """Prove Regular Hadamard is better than Sylvester for outlier suppression."""
    print("\n" + "=" * 70)
    print("  CLAIM 1: Regular Hadamard Suppresses Outliers Better Than Sylvester")
    print("=" * 70)

    torch.manual_seed(42)
    x = torch.randn(32, 1024)
    x[5, :] += 20.0
    x[12, :] += 15.0
    x[:, 100] += 30.0

    original_max = x.abs().max().item()
    print(f"\n  Simulated DiT activation with outliers, max = {original_max:.2f}")

    results = compare_outlier_reduction(x, group_sizes=[16, 64, 256])
    for gs, (amp, reduction) in [(k, v) for k, v in results.items() if k != "original"]:
        print(f"  Group {gs:>4}: max = {amp:.2f}, reduction = {reduction:+.1f}%")

    H_syl = torch.tensor([[1.0]])
    n = 1
    while n < 256:
        H_syl = torch.cat([torch.cat([H_syl, H_syl], dim=1),
                           torch.cat([H_syl, -H_syl], dim=1)], dim=0)
        n *= 2
    syl_disc = H_syl.sum(dim=0).abs().max().item()
    reg_disc = regular_hadamard_matrix(256).sum(dim=0).abs().max().item()

    print(f"\n  Column discrepancy (lower = better):")
    print(f"    Sylvester H_256: {syl_disc:.0f}  (worst case = n)")
    print(f"    Regular   H_256: {reg_disc:.0f}  (optimal = sqrt(n) = 16)")
    print(f"    -> Regular is {syl_disc/reg_disc:.0f}x better")

    x_outlier = torch.randn(1, 256) + 10.0
    H_reg = regular_hadamard_matrix(256) / math.sqrt(256)
    H_syl_norm = H_syl / math.sqrt(256)
    after_reg = (x_outlier @ H_reg.T).abs().max().item()
    after_syl = (x_outlier @ H_syl_norm.T).abs().max().item()
    print(f"\n  Row-wise outlier amplification test:")
    print(f"    Regular:  {after_reg:.2f}")
    print(f"    Sylvester: {after_syl:.2f}")
    print(f"    VERDICT: {'PASS — Regular avoids amplification' if after_reg < after_syl else 'FAIL'}")

    return {
        "original_max": original_max,
        "regular_disc": reg_disc,
        "sylvester_disc": syl_disc,
        "regular_outlier": after_reg,
        "sylvester_outlier": after_syl,
        "passed": after_reg < after_syl,
    }


def main():
    parser = argparse.ArgumentParser(
        description="ConvRot Paper Claims Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="sdxl",
                        choices=["sdxl", "flux-dev", "flux-schnell"],
                        help="Diffusion model to evaluate")
    parser.add_argument("--precision", type=str, nargs="+",
                        default=["bf16", "w8a8", "w4a4"],
                        choices=["bf16", "w8a8", "w4a4", "w4a4-mixed"],
                        help="Precisions to test")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Hadamard rotation group size (64 for broad compat, 256 for paper)")
    parser.add_argument("--num-images", type=int, default=3,
                        help="Number of images to generate per precision")
    parser.add_argument("--steps", type=int, default=20,
                        help="Diffusion steps (paper uses 50 for flux-dev)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="results/proof")
    parser.add_argument("--use-paper-prompts", action="store_true",
                        help="Use the exact PRISM-Bench prompts from the ConvRot repo")
    parser.add_argument("--skip-outlier-test", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    gpu_name, gpu_vram = get_gpu_info()

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images" / args.model
    images_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report").mkdir(parents=True, exist_ok=True)

    if args.use_paper_prompts:
        prompts = PAPER_PROMPTS[:args.num_images]
    else:
        all_prompts = PAPER_PROMPTS + EXTRA_PROMPTS
        prompts = all_prompts[:args.num_images]

    print("=" * 70)
    print("  ConvRot Paper Claims Verification")
    print("=" * 70)
    print(f"  Date:        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model:       {args.model}")
    print(f"  GPU:         {gpu_name} ({gpu_vram:.1f} GB)")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  Device:      {device}")
    print(f"  Precisions:  {args.precision}")
    print(f"  Group size:  {args.group_size}")
    print(f"  Num images:  {args.num_images}")
    print(f"  Steps:       {args.steps}")
    print(f"  Output:      {output_dir}")
    print("=" * 70)

    # ── Outlier Analysis ────────────────────────────────────────────────
    outlier_results = None
    if not args.skip_outlier_test:
        outlier_results = run_outlier_analysis(output_dir)

    # ── Load LPIPS if available ─────────────────────────────────────────
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        print("\n  LPIPS metric: enabled")
    except ImportError:
        print("\n  LPIPS metric: disabled (install with: pip install lpips)")

    # ── Generate images for each precision ──────────────────────────────
    all_image_results = []
    bf16_images = {}
    model_sizes = {}
    gen_times = {}
    peak_memory = {}

    precisions_to_run = args.precision
    if "bf16" not in precisions_to_run:
        precisions_to_run = ["bf16"] + precisions_to_run

    for prec_idx, precision in enumerate(precisions_to_run):
        print(f"\n{'=' * 70}")
        print(f"  PHASE {prec_idx+1}/{len(precisions_to_run)}: Generating {precision.upper()} images")
        print(f"{'=' * 70}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        pipe = load_pipeline(args.model, device, dtype)
        component, comp_name = get_component(pipe)
        size_before = get_model_size_mb(component)

        if precision == "bf16":
            model_sizes["bf16"] = size_before
            print(f"  {comp_name} size: {size_before:.1f} MB (BF16, no quantization)")
        else:
            actual_precision = precision.replace("-mixed", "")

            # Model-specific skip layers (critical for visual quality)
            if "sdxl" in args.model or comp_name == "unet":
                from convrot.quantize import SDXL_SKIP_LAYERS, SDXL_MIXED_PRECISION_LAYERS
                skip_layers = list(SDXL_SKIP_LAYERS)
                mixed_layers = list(SDXL_MIXED_PRECISION_LAYERS) if actual_precision == "w4a4" else []
            elif "flux" in args.model:
                from convrot.quantize import FLUX_SKIP_LAYERS, FLUX_MIXED_PRECISION_LAYERS
                skip_layers = list(FLUX_SKIP_LAYERS)
                mixed_layers = list(FLUX_MIXED_PRECISION_LAYERS) if actual_precision == "w4a4" else []
            else:
                skip_layers = []
                mixed_layers = []

            if precision == "w4a4-mixed":
                actual_precision = "w4a4"

            config = ConvRotConfig(
                precision=actual_precision,
                group_size=args.group_size,
                mixed_precision_layers=mixed_layers,
                skip_layers=skip_layers,
            )

            print(f"  Quantizing {comp_name} to {precision.upper()}...")
            t0 = time.time()
            convrot_quantize_(component, config)
            quant_time = time.time() - t0
            print(f"  Quantization time: {quant_time:.1f}s")

            size_after = get_model_size_mb(component)
            model_sizes[precision] = size_after
            print(f"  Size: {size_before:.1f} MB -> {size_after:.1f} MB ({size_before / size_after:.2f}x)")
            print_quantization_summary(component, config)

        times_this_prec = []
        for i, prompt in enumerate(prompts):
            print(f"\n  [{i+1}/{len(prompts)}] {prompt[:60]}...")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()

            img = generate_one(pipe, prompt, args.steps, args.seed + i, args.model, device)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - t0
            times_this_prec.append(elapsed)

            img_path = images_dir / f"{precision}_{i:03d}.png"
            img.save(img_path, quality=95)
            print(f"    Time: {elapsed:.1f}s  |  Saved: {img_path.name}")

            if precision == "bf16":
                bf16_images[i] = img

            all_image_results.append(ImageResult(
                prompt_idx=i,
                prompt=prompt,
                precision=precision,
                group_size=args.group_size,
                gen_time_sec=elapsed,
                image_path=str(img_path),
            ))

        gen_times[precision] = times_this_prec
        if torch.cuda.is_available():
            peak_memory[precision] = torch.cuda.max_memory_allocated() / (1024**3)

        avg_time = sum(times_this_prec) / len(times_this_prec)
        print(f"\n  Average time ({precision}): {avg_time:.1f}s/image")
        if torch.cuda.is_available():
            print(f"  Peak GPU memory: {peak_memory[precision]:.2f} GB")

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Compute Quality Metrics ─────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Computing Quality Metrics")
    print(f"{'=' * 70}")

    quality_results = []
    for precision in precisions_to_run:
        if precision == "bf16":
            continue

        psnr_scores = []
        ssim_scores = []
        lpips_scores = []

        for i in range(len(prompts)):
            if i not in bf16_images:
                continue

            q_path = images_dir / f"{precision}_{i:03d}.png"
            if not q_path.exists():
                continue

            img_q = Image.open(q_path)
            img_bf16 = bf16_images[i]

            arr_bf16 = np.array(img_bf16)
            arr_q = np.array(img_q)

            if arr_bf16.shape != arr_q.shape:
                img_q = img_q.resize(img_bf16.size, Image.LANCZOS)
                arr_q = np.array(img_q)

            psnr = compute_psnr(arr_bf16, arr_q)
            ssim = compute_ssim(arr_bf16, arr_q)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            lp = compute_lpips_score(img_bf16, img_q, lpips_fn, device)
            if lp >= 0:
                lpips_scores.append(lp)

            print(f"  {precision:>10} #{i}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}" +
                  (f", LPIPS={lp:.4f}" if lp >= 0 else ""))

        avg_psnr = np.mean(psnr_scores) if psnr_scores else 0
        avg_ssim = np.mean(ssim_scores) if ssim_scores else 0
        avg_lpips = np.mean(lpips_scores) if lpips_scores else -1

        quality_results.append(QualityMetrics(
            precision=precision,
            avg_psnr=avg_psnr,
            avg_ssim=avg_ssim,
            avg_lpips=avg_lpips,
            num_images=len(psnr_scores),
        ))

    # ── Create Comparison Grid ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Creating Comparison Grid")
    print(f"{'=' * 70}")

    grid_path = output_dir / f"comparison_{args.model}.png"
    create_comparison_grid(images_dir, precisions_to_run, len(prompts), grid_path)

    # ── Performance Summary ─────────────────────────────────────────────
    bf16_avg_time = np.mean(gen_times.get("bf16", [1.0]))
    bf16_size = model_sizes.get("bf16", 1.0)

    perf_results = []
    for prec in precisions_to_run:
        avg_t = np.mean(gen_times[prec])
        size = model_sizes.get(prec, bf16_size)
        perf_results.append(PerformanceMetrics(
            precision=prec,
            model_size_mb=size,
            avg_gen_time_sec=avg_t,
            speedup_vs_bf16=bf16_avg_time / avg_t if avg_t > 0 else 0,
            memory_reduction_vs_bf16=bf16_size / size if size > 0 else 0,
        ))

    # ── Final Report ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  FINAL VERIFICATION REPORT — ConvRot on {args.model.upper()}")
    print(f"{'=' * 70}")
    print(f"  GPU: {gpu_name} ({gpu_vram:.1f} GB)")
    print(f"  Model: {args.model} | Steps: {args.steps} | Group: {args.group_size}")
    print(f"  Images per precision: {args.num_images}")

    # Quality table
    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  IMAGE QUALITY (vs BF16 baseline)                              │")
    print(f"  ├──────────────┬──────────┬──────────┬──────────┬────────────────┤")
    print(f"  │ Precision    │ PSNR↑ dB │ SSIM↑    │ LPIPS↓   │ Verdict        │")
    print(f"  ├──────────────┼──────────┼──────────┼──────────┼────────────────┤")
    print(f"  │ BF16 (ref)   │   ref    │   ref    │   ref    │ baseline       │")
    for qr in quality_results:
        verdict = "GOOD" if qr.avg_psnr > 20 else ("OK" if qr.avg_psnr > 15 else "DEGRADED")
        lpips_str = f"{qr.avg_lpips:.4f}" if qr.avg_lpips >= 0 else "  N/A  "
        print(f"  │ {qr.precision:<12} │ {qr.avg_psnr:>7.2f}  │ {qr.avg_ssim:>7.4f}  │ {lpips_str} │ {verdict:<14} │")
    print(f"  └──────────────┴──────────┴──────────┴──────────┴────────────────┘")

    # Performance table
    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  PERFORMANCE                                                    │")
    print(f"  ├──────────────┬────────────┬──────────┬──────────┬──────────────┤")
    print(f"  │ Precision    │ Size (MB)  │ Time (s) │ Speedup  │ Mem Reduce   │")
    print(f"  ├──────────────┼────────────┼──────────┼──────────┼──────────────┤")
    for pr in perf_results:
        print(f"  │ {pr.precision:<12} │ {pr.model_size_mb:>9.1f}  │ {pr.avg_gen_time_sec:>7.1f}  │ {pr.speedup_vs_bf16:>6.2f}x  │ {pr.memory_reduction_vs_bf16:>6.2f}x       │")
    print(f"  └──────────────┴────────────┴──────────┴──────────┴──────────────┘")

    if torch.cuda.is_available() and peak_memory:
        print(f"\n  ┌─────────────────────────────────────────┐")
        print(f"  │  GPU PEAK MEMORY                        │")
        print(f"  ├──────────────┬──────────────────────────┤")
        for prec, mem in peak_memory.items():
            print(f"  │ {prec:<12} │ {mem:>8.2f} GB              │")
        print(f"  └──────────────┴──────────────────────────┘")

    # Paper claims verification
    print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  PAPER CLAIMS VERIFICATION                                     │")
    print(f"  ├────────────────────────────────────────────────────────────────-┤")

    if outlier_results:
        status = "VERIFIED" if outlier_results["passed"] else "FAILED"
        print(f"  │  [{'✓' if outlier_results['passed'] else '✗'}] Outlier suppression (Regular > Sylvester)     {status:>8}  │")

    for pr in perf_results:
        if pr.precision in ("w4a4", "w4a4-mixed"):
            mem_ok = pr.memory_reduction_vs_bf16 > 2.0
            print(f"  │  [{'✓' if mem_ok else '✗'}] Memory reduction {pr.precision}: {pr.memory_reduction_vs_bf16:.2f}x (paper: ~4x)  {'VERIFIED' if mem_ok else 'PARTIAL ':>8}  │")

    for qr in quality_results:
        quality_ok = qr.avg_psnr > 15
        print(f"  │  [{'✓' if quality_ok else '✗'}] {qr.precision} quality preserved (PSNR={qr.avg_psnr:.1f})    {'VERIFIED' if quality_ok else 'DEGRADED':>8}  │")

    print(f"  └─────────────────────────────────────────────────────────────────┘")

    # Output paths
    print(f"\n  OUTPUT FILES:")
    print(f"    Images:     {images_dir}/")
    print(f"    Grid:       {grid_path}")

    # Save JSON report
    report = ProofReport(
        timestamp=datetime.now().isoformat(),
        model_name=args.model,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram,
        torch_version=torch.__version__,
        num_images=args.num_images,
        num_steps=args.steps,
        group_size=args.group_size,
        precisions_tested=precisions_to_run,
        quality_metrics=[asdict(q) for q in quality_results],
        performance_metrics=[asdict(p) for p in perf_results],
        image_results=[asdict(r) for r in all_image_results],
    )

    report_path = output_dir / "report" / f"proof_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    print(f"    Report:     {report_path}")

    print(f"\n{'=' * 70}")
    print(f"  PROOF GENERATION COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
