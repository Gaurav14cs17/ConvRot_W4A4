"""
Quick test: Apply ConvRot to a real diffusion model and generate one image.

This is the simplest possible end-to-end test. Requires:
    pip install diffusers transformers accelerate

Works with ~7GB VRAM (SDXL) or even CPU (slow but works).

Usage:
    python examples/quick_test.py
    python examples/quick_test.py --model flux-schnell --device cuda
    python examples/quick_test.py --precision w4a4 --mixed-precision
"""

import torch
import time
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convrot import convrot_quantize_, ConvRotConfig
from convrot.quantize import get_model_size_mb


def main():
    parser = argparse.ArgumentParser(description="Quick ConvRot test")
    parser.add_argument("--model", type=str, default="sdxl",
                        choices=["sdxl", "flux-dev", "flux-schnell"])
    parser.add_argument("--precision", type=str, default="w8a8",
                        choices=["w8a8", "w4a4"])
    parser.add_argument("--group-size", type=int, default=64,
                        help="Use 64 for broader compatibility (256 requires in_features divisible by 256)")
    parser.add_argument("--prompt", type=str,
                        default="A cute cat sitting on a sunny windowsill, photorealistic")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"ConvRot Quick Test")
    print(f"  Model:     {args.model}")
    print(f"  Precision: {args.precision}")
    print(f"  Group:     {args.group_size}")
    print(f"  Device:    {device}")
    print()

    try:
        from diffusers import StableDiffusionXLPipeline, FluxPipeline
    except ImportError:
        print("Install diffusers first:")
        print("  pip install diffusers transformers accelerate")
        return

    # Load model
    print("Loading model...")
    model_ids = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "flux-dev": "black-forest-labs/FLUX.1-dev",
        "flux-schnell": "black-forest-labs/FLUX.1-schnell",
    }

    model_id = model_ids[args.model]

    if args.model == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype)
    else:
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)

    pipe.to(device)

    # Get the component to quantize
    if hasattr(pipe, 'transformer'):
        component = pipe.transformer
        name = "transformer"
    else:
        component = pipe.unet
        name = "unet"

    size_bf16 = get_model_size_mb(component)
    print(f"  {name} size (BF16): {size_bf16:.0f} MB")

    # --- Generate BF16 baseline ---
    print(f"\nGenerating BF16 image...")
    generator = torch.Generator(device=device).manual_seed(42)
    steps = 4 if "schnell" in args.model else args.steps

    t0 = time.time()
    img_bf16 = pipe(args.prompt, num_inference_steps=steps, generator=generator).images[0]
    time_bf16 = time.time() - t0
    print(f"  Time: {time_bf16:.1f}s")

    output_dir = Path("results/images/quick_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    img_bf16.save(output_dir / "bf16.png")

    # --- Apply ConvRot quantization ---
    print(f"\nApplying ConvRot {args.precision.upper()} (group_size={args.group_size})...")
    mixed_layers = ["attn.to_out", "attn.to_v"] if args.mixed_precision else []

    config = ConvRotConfig(
        precision=args.precision,
        group_size=args.group_size,
        mixed_precision_layers=mixed_layers,
    )
    convrot_quantize_(component, config)

    size_quant = get_model_size_mb(component)
    print(f"  {name} size ({args.precision}): {size_quant:.0f} MB")
    print(f"  Compression: {size_bf16/size_quant:.2f}x")

    # --- Generate quantized image ---
    print(f"\nGenerating quantized image...")
    generator = torch.Generator(device=device).manual_seed(42)

    t0 = time.time()
    img_q = pipe(args.prompt, num_inference_steps=steps, generator=generator).images[0]
    time_q = time.time() - t0
    print(f"  Time: {time_q:.1f}s")

    img_q.save(output_dir / f"{args.precision}.png")

    # --- Compute basic metrics ---
    import numpy as np
    arr_bf16 = np.array(img_bf16).astype(float)
    arr_q = np.array(img_q).astype(float)

    mse = np.mean((arr_bf16 - arr_q) ** 2)
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')

    # --- Results ---
    print(f"\n{'=' * 50}")
    print(f"RESULTS")
    print(f"{'=' * 50}")
    print(f"  Model:        {args.model}")
    print(f"  Precision:    BF16 -> {args.precision.upper()}")
    print(f"  Memory:       {size_bf16:.0f} MB -> {size_quant:.0f} MB ({size_bf16/size_quant:.2f}x reduction)")
    print(f"  Latency:      {time_bf16:.1f}s -> {time_q:.1f}s ({time_bf16/time_q:.2f}x speedup)")
    print(f"  PSNR:         {psnr:.2f} dB")
    print(f"  Images saved: {output_dir}/")
    print(f"{'=' * 50}")
    print(f"\nOpen both images to visually compare quality:")
    print(f"  {output_dir}/bf16.png")
    print(f"  {output_dir}/{args.precision}.png")


if __name__ == "__main__":
    main()
