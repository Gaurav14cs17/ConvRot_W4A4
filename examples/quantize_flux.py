"""
Example: Apply ConvRot quantization to FLUX.1-dev for W4A4 inference.

This script demonstrates how to:
1. Load a FLUX.1-dev model via diffusers
2. Apply ConvRot W4A4 quantization with mixed precision
3. Generate images and compare quality

Requirements:
    pip install diffusers transformers accelerate
"""

import torch
import time
import argparse
from pathlib import Path

# ConvRot imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from convrot import convrot_quantize_, ConvRotConfig
from convrot.quantize import get_model_size_mb, print_quantization_summary


def parse_args():
    parser = argparse.ArgumentParser(description="ConvRot quantization for FLUX.1")
    parser.add_argument(
        "--model-id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="w8a8",
        choices=["w8a8", "w4a4"],
        help="Quantization precision",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=256,
        help="Hadamard rotation group size (must be power of 4)",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision (20%% INT8 for sensitive layers)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cute cat sitting on a windowsill, sunlight streaming in",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/images",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--compare-bf16",
        action="store_true",
        help="Also generate BF16 reference image for comparison",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Group size: {args.group_size}")
    print(f"Mixed precision: {args.mixed_precision}")
    print()

    try:
        from diffusers import FluxPipeline
    except ImportError:
        print("ERROR: Please install diffusers: pip install diffusers transformers accelerate")
        return

    # Load model
    print("Loading FLUX.1-dev pipeline...")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
    )
    pipe.to(device)

    transformer = pipe.transformer
    size_before = get_model_size_mb(transformer)
    print(f"Transformer size (BF16): {size_before:.1f} MB")

    # Generate BF16 reference if requested
    if args.compare_bf16:
        print("\nGenerating BF16 reference image...")
        generator = torch.Generator(device=device).manual_seed(42)
        t0 = time.time()
        image_bf16 = pipe(
            args.prompt,
            num_inference_steps=args.num_steps,
            generator=generator,
        ).images[0]
        t_bf16 = time.time() - t0
        print(f"BF16 generation time: {t_bf16:.1f}s")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_bf16.save(output_dir / "flux_bf16.png")

    # Apply ConvRot quantization
    print("\nApplying ConvRot quantization...")
    mixed_layers = []
    if args.mixed_precision and args.precision == "w4a4":
        mixed_layers = [
            "attn.to_out",
            "attn.to_v",
        ]

    config = ConvRotConfig(
        precision=args.precision,
        group_size=args.group_size,
        mixed_precision_layers=mixed_layers,
    )

    t0 = time.time()
    convrot_quantize_(transformer, config)
    t_quant = time.time() - t0
    print(f"Quantization time: {t_quant:.1f}s")

    print_quantization_summary(transformer, config)

    # Generate quantized image
    print("\nGenerating quantized image...")
    generator = torch.Generator(device=device).manual_seed(42)
    t0 = time.time()
    image_q = pipe(
        args.prompt,
        num_inference_steps=args.num_steps,
        generator=generator,
    ).images[0]
    t_q = time.time() - t0
    print(f"Quantized generation time: {t_q:.1f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_q.save(output_dir / f"flux_{args.precision}_gs{args.group_size}.png")

    # Summary
    size_after = get_model_size_mb(transformer)
    print(f"\n{'=' * 50}")
    print(f"Results Summary")
    print(f"{'=' * 50}")
    print(f"  Model size:   {size_before:.1f} MB -> {size_after:.1f} MB ({size_before/size_after:.2f}x reduction)")
    if args.compare_bf16:
        print(f"  Latency:      {t_bf16:.1f}s (BF16) -> {t_q:.1f}s ({args.precision})")
        print(f"  Speedup:      {t_bf16/t_q:.2f}x")
    print(f"  Images saved to: {output_dir}")


if __name__ == "__main__":
    main()
