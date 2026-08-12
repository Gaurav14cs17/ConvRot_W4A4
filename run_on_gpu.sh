#!/bin/bash
set -e

# ConvRot GPU runner
#
# Usage:
#   bash run_on_gpu.sh quick        # SDXL proof (UNet, ~7GB)
#   bash run_on_gpu.sh dit          # FLUX.1-schnell benchmark (recommended)
#   bash run_on_gpu.sh dit-full     # FLUX.1-dev benchmark
#   bash run_on_gpu.sh all          # Everything
#
# Remote:
#   scp -r . user@gpu-server:~/convrot-torchao
#   ssh user@gpu-server "cd ~/convrot-torchao && bash run_on_gpu.sh dit"

echo "=== ConvRot GPU Runner ==="

echo ""
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi not found"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1024**3
    print(f'VRAM: {vram:.1f} GB')
" 2>/dev/null || true
echo ""

echo "--- Installing dependencies ---"
pip install -e . 2>/dev/null || pip install --user -e . 2>/dev/null || true
pip install diffusers transformers accelerate safetensors 2>/dev/null || \
    pip install --user diffusers transformers accelerate safetensors 2>/dev/null || true
pip install torchao 2>/dev/null || pip install --user torchao 2>/dev/null || echo "torchao install failed (needed for comparison)"
pip install lpips 2>/dev/null || pip install --user lpips 2>/dev/null || echo "LPIPS not installed (optional)"
echo ""

echo "--- Quick CPU verification ---"
python3 examples/cpu_demo.py
echo ""

MODE="${1:-dit}"

case "$MODE" in
    quick|sdxl)
        echo "--- SDXL Quick Proof (3 images, ~7GB VRAM) ---"
        python3 generate.py \
            --model sdxl \
            --precision bf16 w8a8 w4a4 \
            --group-size 64 \
            --num-images 3 \
            --steps 20 \
            --use-paper-prompts \
            --output-dir results/proof_sdxl
        ;;

    dit|flux-schnell|schnell)
        echo "--- FLUX.1-schnell DiT Benchmark ---"
        python3 benchmark_dit.py \
            --model flux-schnell \
            --methods bf16 torchao-int8wo torchao-int4wo torchao-fp8wo convrot-w8a8 convrot-w4a4-mixed \
            --group-size 256 \
            --num-images 3 \
            --output-dir results/dit_benchmark
        ;;

    dit-full|flux-dev|flux)
        echo "--- FLUX.1-dev DiT Benchmark ---"
        python3 benchmark_dit.py \
            --model flux-dev \
            --methods bf16 torchao-int8wo torchao-int4wo torchao-fp8wo convrot-w8a8 convrot-w4a4-mixed \
            --group-size 256 \
            --num-images 5 \
            --steps 50 \
            --output-dir results/dit_benchmark
        ;;

    paper)
        echo "--- Paper-matched FLUX.1-dev eval (quality/size; not fused-kernel latency) ---"
        python3 benchmark_dit.py \
            --model flux-dev \
            --methods bf16 torchao-fp8wo torchao-int8wo convrot-w8a8 convrot-w4a4-mixed \
            --group-size 256 \
            --num-images 5 \
            --steps 50 \
            --output-dir results/dit_benchmark_paper
        ;;

    convrot-only)
        echo "--- FLUX.1-schnell ConvRot Only (no torchao comparison) ---"
        python3 benchmark_dit.py \
            --model flux-schnell \
            --methods bf16 convrot-w8a8 convrot-w4a4-mixed \
            --group-size 256 \
            --num-images 3 \
            --output-dir results/dit_benchmark
        ;;

    all)
        echo "--- Running ALL benchmarks ---"
        echo ""
        echo "=== 1/3: SDXL (UNet baseline) ==="
        python3 generate.py \
            --model sdxl \
            --precision bf16 w8a8 w4a4 \
            --group-size 64 \
            --num-images 3 \
            --steps 20 \
            --use-paper-prompts \
            --output-dir results/proof_sdxl

        echo ""
        echo "=== 2/3: FLUX.1-schnell (DiT, 4 steps) ==="
        python3 benchmark_dit.py \
            --model flux-schnell \
            --methods bf16 torchao-int8wo torchao-int4wo convrot-w8a8 convrot-w4a4-mixed \
            --group-size 256 \
            --num-images 3 \
            --output-dir results/dit_benchmark

        echo ""
        echo "=== 3/3: FLUX.1-dev (DiT, 50 steps) ==="
        python3 benchmark_dit.py \
            --model flux-dev \
            --methods bf16 torchao-int8wo torchao-int4wo convrot-w8a8 convrot-w4a4-mixed \
            --group-size 256 \
            --num-images 5 \
            --steps 50 \
            --output-dir results/dit_benchmark
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Usage: bash run_on_gpu.sh [MODE]"
        echo ""
        echo "  quick           SDXL proof (UNet, ~7GB VRAM)"
        echo "  dit             FLUX.1-schnell DiT benchmark (recommended, ~24GB)"
        echo "  dit-full        FLUX.1-dev DiT benchmark (paper's target, ~32GB)"
        echo "  paper           FLUX.1-dev + FP8 baseline (paper-matched eval)"
        echo "  convrot-only    FLUX.1-schnell ConvRot methods only (no torchao)"
        echo "  all             All benchmarks"
        exit 1
        ;;
esac

echo ""
echo "Done. Copy results with:"
echo "  scp -r user@gpu-server:~/convrot-torchao/results/ ./results"
