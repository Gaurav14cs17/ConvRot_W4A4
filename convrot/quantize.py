"""
Model-level quantization API for ConvRot.

Provides convrot_quantize_() which replaces nn.Linear modules in a model
with ConvLinear4bit or ConvLinear8bit layers.

Usage:
    from convrot import convrot_quantize_, ConvRotConfig

    config = ConvRotConfig(precision="w8a8", group_size=256)
    convrot_quantize_(model, config)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Callable
from .conv_linear import ConvLinear8bit, ConvLinear4bit


@dataclass
class ConvRotConfig:
    """Configuration for ConvRot quantization.

    Args:
        precision: Quantization precision. One of "w8a8", "w4a4".
        group_size: Hadamard rotation group size. Must be a power of 4.
            Larger values provide better outlier suppression but more computation.
            Recommended: 256 (good balance of precision and efficiency).
        weight_group_size: Group size for weight quantization along K dimension.
            Only used for W4A4. Default: 128.
        mixed_precision_layers: List of layer name patterns that should use
            higher precision (W8A8) even when overall precision is W4A4.
            This implements the mixed-precision strategy from the paper.
        skip_layers: List of layer name patterns to skip (keep in original precision).
        filter_fn: Optional function (module, fqn) -> bool to select which
            modules to quantize.
    """
    precision: str = "w8a8"
    group_size: int = 256
    weight_group_size: int = 128
    mixed_precision_layers: list[str] = field(default_factory=list)
    skip_layers: list[str] = field(default_factory=list)
    filter_fn: Optional[Callable[[nn.Module, str], bool]] = None
    # Paper / official activation scale: per-token (True) or per-tensor (False)
    act_per_token: bool = True

    def __post_init__(self):
        if self.precision not in ("w8a8", "w4a4"):
            raise ValueError(f"precision must be 'w8a8' or 'w4a4', got '{self.precision}'")

        import math
        log4 = math.log2(self.group_size) / 2
        if log4 != int(log4) or self.group_size < 4:
            raise ValueError(
                f"group_size must be a power of 4 (4, 16, 64, 256, ...), got {self.group_size}"
            )


# FLUX.1 mixed-precision layers (paper Table 8, ~20% of layers kept at W8A8)
FLUX_MIXED_PRECISION_LAYERS = [
    "attn.to_out",           # transformer_blocks.{i}.attn.to_out.0
    "attn.to_v",             # single_transformer_blocks.{i}.attn.to_v
]

# FLUX.1 layers to keep at full precision.
# Final blocks from paper Section 9.2; embedders match official ConvRot config
# (x_embedder in_dim=64 is not divisible by rot_size=256).
FLUX_SKIP_LAYERS = [
    "x_embedder",
    "time_text_embed",
    "context_embedder",
    "transformer_blocks.18.ff_context.net.2",
    "transformer_blocks.18.ff.net.2",
    "single_transformer_blocks.37.proj_out",
]

# SDXL UNet skip layers
SDXL_SKIP_LAYERS = [
    "time_embedding",
    "add_embedding",
    "conv_in",
    "conv_out",
    "conv_norm_out",
    "class_embedding",
    "label_emb",
]

# SDXL UNet mixed-precision layers (W8A8 when rest is W4A4)
SDXL_MIXED_PRECISION_LAYERS = [
    "to_out",
    "to_v",
    "proj_out",
    "ff.net.2",
    "conv_shortcut",
]


def _should_quantize(module: nn.Module, fqn: str, config: ConvRotConfig) -> bool:
    """Determine if a module should be quantized."""
    if not isinstance(module, nn.Linear):
        return False

    # Check skip list
    for pattern in config.skip_layers:
        if pattern in fqn:
            return False

    # Check input features divisibility by group_size
    if module.in_features % config.group_size != 0:
        return False

    # For W4A4, also check weight_group_size divisibility
    if config.precision == "w4a4" and not _is_mixed_precision_layer(fqn, config):
        if module.in_features % config.weight_group_size != 0:
            return False

    # User-provided filter
    if config.filter_fn is not None:
        return config.filter_fn(module, fqn)

    return True


def _is_mixed_precision_layer(fqn: str, config: ConvRotConfig) -> bool:
    """Check if a layer should use higher precision in mixed-precision mode."""
    for pattern in config.mixed_precision_layers:
        if pattern in fqn:
            return True
    return False


def convrot_quantize_(
    model: nn.Module,
    config: ConvRotConfig,
) -> nn.Module:
    """Apply ConvRot quantization to a model in-place.

    Replaces nn.Linear modules with ConvLinear8bit or ConvLinear4bit based
    on the config.

    Args:
        model: The model to quantize.
        config: Quantization configuration.

    Returns:
        The quantized model (modified in-place).
    """
    replacements = {}

    for fqn, module in model.named_modules():
        if not _should_quantize(module, fqn, config):
            continue

        if config.precision == "w8a8":
            new_module = ConvLinear8bit.from_linear(module, group_size=config.group_size)
        elif config.precision == "w4a4":
            if _is_mixed_precision_layer(fqn, config):
                # Use W8A8 for sensitive layers
                new_module = ConvLinear8bit.from_linear(module, group_size=config.group_size)
            else:
                new_module = ConvLinear4bit.from_linear(
                    module,
                    group_size=config.group_size,
                    weight_group_size=config.weight_group_size,
                    act_per_token=config.act_per_token,
                )
        else:
            continue

        replacements[fqn] = new_module

    # Apply replacements
    for fqn, new_module in replacements.items():
        parts = fqn.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    return model


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB (parameter + buffer memory)."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    for b in model.buffers():
        total_bytes += b.nelement() * b.element_size()
    return total_bytes / (1024 * 1024)


def print_quantization_summary(model: nn.Module, config: ConvRotConfig):
    """Print a summary of quantization applied to the model."""
    n_linear = 0
    n_conv8 = 0
    n_conv4 = 0
    n_skipped = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n_linear += 1
            n_skipped += 1
        elif isinstance(module, ConvLinear8bit):
            n_conv8 += 1
        elif isinstance(module, ConvLinear4bit):
            n_conv4 += 1

    total_quantized = n_conv8 + n_conv4
    print(f"ConvRot Quantization Summary")
    print(f"{'=' * 40}")
    print(f"  Precision:     {config.precision}")
    print(f"  Group size:    {config.group_size}")
    print(f"  Quantized:     {total_quantized} layers")
    print(f"    - W8A8:      {n_conv8}")
    print(f"    - W4A4:      {n_conv4}")
    print(f"  Skipped:       {n_skipped} linear layers")
    print(f"  Model size:    {get_model_size_mb(model):.1f} MB")
    print(f"{'=' * 40}")
