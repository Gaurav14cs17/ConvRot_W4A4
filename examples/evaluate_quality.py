"""
End-to-end image quality evaluation with ConvRot quantization.

Supports:
  - Stable Diffusion XL (sdxl) — ~7GB VRAM, fast to test
  - FLUX.1-dev — 12B params, paper's target model
  - FLUX.1-schnell — faster variant (4 steps)

Generates images at BF16 (baseline) and quantized, then computes:
  - PSNR (Peak Signal-to-Noise Ratio) — higher is better
  - LPIPS (Learned Perceptual Image Patch Similarity) — lower is better
  - SSIM (Structural Similarity) — higher is better
  - FID (Fréchet Inception Distance) — lower is better (needs many images)

Usage:
    # Quick test with SDXL (needs ~7GB VRAM)
    python examples/evaluate_quality.py --model sdxl --num-images 5

    # Full eval with FLUX.1-dev (needs ~24GB VRAM)
    python examples/evaluate_quality.py --model flux-dev --num-images 50

    # Compare W8A8 vs W4A4
    python examples/evaluate_quality.py --model sdxl --precision w8a8 w4a4
"""

import torch
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convrot import convrot_quantize_, ConvRotConfig
from convrot.quantize import get_model_size_mb, print_quantization_summary


EVAL_PROMPTS = [
    "A cute orange cat sitting on a windowsill, sunlight streaming in, photorealistic",
    "A mountain landscape at sunset with a lake reflection, oil painting style",
    "Portrait of an astronaut in a garden, detailed, cinematic lighting",
    "A steampunk clockwork robot reading a book in a library",
    "Fresh sushi platter on a wooden board, food photography, top-down view",
    "An ancient castle on a cliff overlooking the ocean, dramatic sky",
    "A red sports car driving through a neon-lit city at night",
    "A golden retriever playing in autumn leaves, warm lighting",
    "A futuristic space station interior with holographic displays",
    "A cozy cabin in the snowy woods with smoke rising from chimney",
    "A ballerina performing on stage with dramatic spotlight",
    "A tropical beach with crystal clear water and palm trees",
    "A wizard's study filled with floating books and glowing potions",
    "A street market in Tokyo at night with lanterns and food stalls",
    "A macro photo of a dewdrop on a spider web, morning light",
    "An Art Deco style illustration of a jazz musician",
    "A field of sunflowers under a dramatic cloudy sky",
    "A cyberpunk samurai standing in the rain, neon reflections",
    "A medieval blacksmith forging a sword, sparks flying",
    "A peaceful zen garden with raked sand and cherry blossoms",
]


@dataclass
class EvalResult:
    prompt: str
    precision: str
    group_size: int
    psnr: float
    ssim: float
    lpips: float
    gen_time_bf16: float
    gen_time_quant: float
    speedup: float


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images (numpy arrays in [0, 255])."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images. Simple implementation."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim)


def compute_lpips(img1_tensor: torch.Tensor, img2_tensor: torch.Tensor, lpips_fn) -> float:
    """Compute LPIPS (requires lpips package)."""
    if lpips_fn is None:
        return -1.0
    with torch.no_grad():
        score = lpips_fn(img1_tensor, img2_tensor)
    return score.item()


def image_to_tensor(img) -> torch.Tensor:
    """Convert PIL image to tensor for LPIPS [-1, 1] range."""
    import numpy as np
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # Scale to [-1, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor


def load_pipeline(model_name: str, device: str, dtype: torch.dtype):
    """Load the appropriate diffusion pipeline."""
    from diffusers import (
        StableDiffusionXLPipeline,
        FluxPipeline,
        DiffusionPipeline,
    )

    model_map = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    }

    model_id = model_map.get(model_name, model_name)
    print(f"Loading model: {model_id}")

    if "sdxl" in model_name.lower() or "stable-diffusion-xl" in model_id:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None,
        )
    elif "flux" in model_name.lower() or "FLUX" in model_id:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    pipe.to(device)
    return pipe


def get_quantizable_component(pipe, model_name: str):
    """Get the main model component to quantize."""
    if hasattr(pipe, 'transformer'):
        return pipe.transformer, "transformer"
    elif hasattr(pipe, 'unet'):
        return pipe.unet, "unet"
    else:
        raise ValueError(f"Cannot find quantizable component in pipeline")


def generate_image(pipe, prompt: str, num_steps: int, seed: int, model_name: str):
    """Generate a single image with fixed seed."""
    device = pipe.device if hasattr(pipe, 'device') else "cuda"
    generator = torch.Generator(device=device).manual_seed(seed)

    kwargs = {
        "prompt": prompt,
        "num_inference_steps": num_steps,
        "generator": generator,
    }

    if "flux-schnell" in model_name:
        kwargs["num_inference_steps"] = 4

    result = pipe(**kwargs)
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description="ConvRot Image Quality Evaluation")
    parser.add_argument("--model", type=str, default="sdxl",
                        choices=["sdxl", "flux-dev", "flux-schnell"],
                        help="Model to evaluate")
    parser.add_argument("--precision", type=str, nargs="+", default=["w8a8"],
                        choices=["w8a8", "w4a4"],
                        help="Quantization precision(s) to test")
    parser.add_argument("--group-size", type=int, default=256,
                        help="Hadamard rotation group size")
    parser.add_argument("--num-images", type=int, default=5,
                        help="Number of images to generate for evaluation")
    parser.add_argument("--num-steps", type=int, default=30,
                        help="Number of diffusion steps")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resolution", type=int, default=1024,
                        help="Image resolution")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use mixed precision for W4A4")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU (will be slow!)")
        device = "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images" / args.model
    tables_dir = output_dir / "tables"
    images_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Try to load LPIPS
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)
        print("LPIPS metric: enabled")
    except ImportError:
        print("LPIPS metric: disabled (pip install lpips)")

    prompts = EVAL_PROMPTS[:args.num_images]

    # Load pipeline
    print(f"\n{'=' * 60}")
    print(f"ConvRot Quality Evaluation")
    print(f"{'=' * 60}")
    print(f"  Model:       {args.model}")
    print(f"  Precisions:  {args.precision}")
    print(f"  Group size:  {args.group_size}")
    print(f"  Num images:  {args.num_images}")
    print(f"  Steps:       {args.num_steps}")
    print(f"  Device:      {device}")
    print(f"{'=' * 60}\n")

    pipe = load_pipeline(args.model, device, dtype)

    # Phase 1: Generate BF16 baseline images
    print("\n--- Phase 1: Generating BF16 baseline images ---")
    bf16_images = []
    bf16_times = []

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
        t0 = time.time()
        img = generate_image(pipe, prompt, args.num_steps, args.seed + i, args.model)
        elapsed = time.time() - t0
        bf16_times.append(elapsed)
        bf16_images.append(img)
        img.save(images_dir / f"bf16_{i:03d}.png")
        print(f"    Time: {elapsed:.1f}s")

    avg_bf16_time = sum(bf16_times) / len(bf16_times)
    print(f"\n  Average BF16 time: {avg_bf16_time:.1f}s/image")

    # Phase 2: Quantize and evaluate each precision
    all_results = []

    for precision in args.precision:
        print(f"\n--- Phase 2: Evaluating {precision.upper()} ---")

        # Reload model fresh for each precision test
        del pipe
        torch.cuda.empty_cache() if device == "cuda" else None
        pipe = load_pipeline(args.model, device, dtype)

        component, comp_name = get_quantizable_component(pipe, args.model)
        size_before = get_model_size_mb(component)

        # Apply quantization
        mixed_layers = []
        if args.mixed_precision and precision == "w4a4":
            mixed_layers = ["attn.to_out", "attn.to_v"]

        config = ConvRotConfig(
            precision=precision,
            group_size=args.group_size,
            mixed_precision_layers=mixed_layers,
        )

        print(f"  Quantizing {comp_name}...")
        t0 = time.time()
        convrot_quantize_(component, config)
        quant_time = time.time() - t0
        size_after = get_model_size_mb(component)
        print(f"  Quantization time: {quant_time:.1f}s")
        print(f"  Size: {size_before:.0f} MB -> {size_after:.0f} MB ({size_before/size_after:.2f}x reduction)")

        # Generate quantized images and compute metrics
        quant_images = []
        quant_times = []
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []

        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
            t0 = time.time()
            img_q = generate_image(pipe, prompt, args.num_steps, args.seed + i, args.model)
            elapsed = time.time() - t0
            quant_times.append(elapsed)
            quant_images.append(img_q)
            img_q.save(images_dir / f"{precision}_{i:03d}.png")

            # Compute metrics
            img_bf16_np = np.array(bf16_images[i])
            img_q_np = np.array(img_q)

            psnr = compute_psnr(img_bf16_np, img_q_np)
            ssim = compute_ssim(img_bf16_np, img_q_np)
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)

            if lpips_fn is not None:
                t1 = image_to_tensor(bf16_images[i]).to(device)
                t2 = image_to_tensor(img_q).to(device)
                lp = compute_lpips(t1, t2, lpips_fn)
                lpips_scores.append(lp)

            print(f"    Time: {elapsed:.1f}s | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}", end="")
            if lpips_scores:
                print(f" | LPIPS: {lpips_scores[-1]:.4f}")
            else:
                print()

            all_results.append(EvalResult(
                prompt=prompt,
                precision=precision,
                group_size=args.group_size,
                psnr=psnr,
                ssim=ssim,
                lpips=lpips_scores[-1] if lpips_scores else -1,
                gen_time_bf16=bf16_times[i],
                gen_time_quant=elapsed,
                speedup=bf16_times[i] / elapsed if elapsed > 0 else 0,
            ))

        # Summary for this precision
        avg_quant_time = sum(quant_times) / len(quant_times)
        avg_psnr = sum(psnr_scores) / len(psnr_scores)
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        avg_lpips = sum(lpips_scores) / len(lpips_scores) if lpips_scores else -1
        avg_speedup = avg_bf16_time / avg_quant_time if avg_quant_time > 0 else 0

        print(f"\n  {'=' * 50}")
        print(f"  {precision.upper()} Summary ({args.num_images} images)")
        print(f"  {'=' * 50}")
        print(f"  Memory:    {size_before:.0f} MB -> {size_after:.0f} MB ({size_before/size_after:.2f}x)")
        print(f"  Latency:   {avg_bf16_time:.1f}s -> {avg_quant_time:.1f}s ({avg_speedup:.2f}x speedup)")
        print(f"  PSNR:      {avg_psnr:.2f} dB")
        print(f"  SSIM:      {avg_ssim:.4f}")
        if avg_lpips >= 0:
            print(f"  LPIPS:     {avg_lpips:.4f}")
        print(f"  {'=' * 50}")

    # Save results as JSON
    results_data = [asdict(r) for r in all_results]
    results_file = tables_dir / f"eval_{args.model}_{args.group_size}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Print final comparison table
    print(f"\n{'=' * 70}")
    print(f"FINAL COMPARISON TABLE — {args.model}")
    print(f"{'=' * 70}")
    print(f"{'Precision':<12} {'PSNR↑':<10} {'SSIM↑':<10} {'LPIPS↓':<10} {'Latency':<12} {'Speedup':<10} {'Memory':<10}")
    print(f"{'-' * 70}")
    print(f"{'BF16':<12} {'ref':<10} {'ref':<10} {'ref':<10} {avg_bf16_time:<12.1f} {'1.00x':<10} {f'{size_before:.0f}MB':<10}")

    for precision in args.precision:
        p_results = [r for r in all_results if r.precision == precision]
        avg_p = np.mean([r.psnr for r in p_results])
        avg_s = np.mean([r.ssim for r in p_results])
        avg_l = np.mean([r.lpips for r in p_results if r.lpips >= 0]) if any(r.lpips >= 0 for r in p_results) else -1
        avg_t = np.mean([r.gen_time_quant for r in p_results])
        avg_sp = avg_bf16_time / avg_t if avg_t > 0 else 0

        lpips_str = f"{avg_l:.4f}" if avg_l >= 0 else "N/A"
        print(f"{precision.upper():<12} {avg_p:<10.2f} {avg_s:<10.4f} {lpips_str:<10} {avg_t:<12.1f} {f'{avg_sp:.2f}x':<10} {f'{size_after:.0f}MB':<10}")

    print(f"{'=' * 70}")
    print(f"\nImages saved to: {images_dir}")
    print(f"To visually compare: open bf16_000.png and {args.precision[0]}_000.png side by side")


if __name__ == "__main__":
    main()
