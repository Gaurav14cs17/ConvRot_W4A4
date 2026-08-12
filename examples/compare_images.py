"""
Create side-by-side comparison images for the README.

Usage:
    python examples/compare_images.py --results-dir results/images/sdxl
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_comparison_grid(
    images_dir: Path,
    output_path: Path,
    precisions: list[str] = ["w8a8", "w4a4"],
    num_images: int = 4,
    img_size: int = 512,
):
    """Create a grid comparing BF16 vs quantized images."""
    cols = 1 + len(precisions)  # BF16 + each precision
    rows = num_images

    grid_w = cols * img_size
    grid_h = rows * img_size + 60  # Extra space for headers
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)

    # Headers
    headers = ["BF16 (Reference)"] + [p.upper() for p in precisions]
    for col, header in enumerate(headers):
        x = col * img_size + img_size // 2
        draw.text((x, 20), header, fill="black", anchor="mt")

    # Fill images
    for row in range(num_images):
        # BF16
        bf16_path = images_dir / f"bf16_{row:03d}.png"
        if bf16_path.exists():
            img = Image.open(bf16_path).resize((img_size, img_size))
            grid.paste(img, (0, row * img_size + 60))

        # Quantized
        for col, precision in enumerate(precisions, 1):
            q_path = images_dir / f"{precision}_{row:03d}.png"
            if q_path.exists():
                img = Image.open(q_path).resize((img_size, img_size))
                grid.paste(img, (col * img_size, row * img_size + 60))

    grid.save(output_path)
    print(f"Comparison grid saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results/images/sdxl")
    parser.add_argument("--output", type=str, default="results/comparison_grid.png")
    parser.add_argument("--precisions", nargs="+", default=["w8a8", "w4a4"])
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    images_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist.")
        print("Run evaluate_quality.py first to generate images.")
        return

    create_comparison_grid(
        images_dir,
        output_path,
        precisions=args.precisions,
        num_images=args.num_images,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
