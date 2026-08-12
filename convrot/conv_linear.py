"""
ConvLinear4bit and ConvLinear8bit: Plug-and-play quantized linear layers.

These modules replace nn.Linear and integrate:
  1. ConvRot (group-wise Regular Hadamard rotation)
  2. Quantization (per-token for activations, per-channel for weights)
  3. Low-precision GEMM (INT4 or INT8)
  4. Dequantization

The rotation is applied to weights offline (during module creation) and to
activations online (during forward pass). This preserves the linear
transformation equivalence: X @ W.T = RHT(X) @ RHT(W).T
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from .hadamard import get_regular_hadamard
from .convrot import group_rht, group_rht_weight


def symmetric_quantize(
    x: torch.Tensor,
    bits: int,
    per_token: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric quantization to n-bit integers.

    Args:
        x: Input tensor.
        bits: Number of bits (4 or 8).
        per_token: If True, compute scale per-row (for activations).
                   If False, compute scale per-tensor.

    Returns:
        (quantized_tensor, scale)
    """
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    if per_token:
        amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    else:
        amax = x.abs().amax().clamp(min=1e-8)

    scale = amax / qmax
    x_q = (x / scale).round().clamp(qmin, qmax).to(torch.int8)

    return x_q, scale


def symmetric_dequantize(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize integer tensor back to floating point."""
    return x_q.to(dtype) * scale.to(dtype)


def pack_int4(weight_q: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values stored in int8 into uint8 (2 values per byte).

    Layout: even index → low nibble, odd index → high nibble.
    Input shape (N, K) with K even → output (N, K // 2) uint8.
    """
    if weight_q.shape[-1] % 2 != 0:
        raise ValueError(f"last dim must be even to pack int4, got {weight_q.shape}")
    # Shift signed [-8, 7] → unsigned [0, 15] for nibble packing
    u = (weight_q.to(torch.int16) + 8).to(torch.uint8)
    lo = u[..., 0::2]
    hi = u[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, k: int) -> torch.Tensor:
    """Unpack uint8 nibble-packed weights back to signed int8 in [-8, 7]."""
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(*packed.shape[:-1], k, dtype=torch.int8, device=packed.device)
    out[..., 0::2] = lo.to(torch.int8)
    out[..., 1::2] = hi.to(torch.int8)
    return out


class ConvLinear8bit(nn.Module):
    """W8A8 quantized linear layer with ConvRot rotation.

    Applies group-wise Regular Hadamard rotation to suppress outliers,
    then performs INT8 matrix multiplication.

    The weight is pre-rotated and quantized during initialization.
    Activations are rotated and quantized on-the-fly during forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 256,
        weight: Optional[torch.Tensor] = None,
        bias_data: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        H = get_regular_hadamard(group_size)
        self.register_buffer("hadamard", H)

        if weight is not None:
            self._quantize_weight(weight)
        else:
            # Placeholder — will be set via from_linear()
            self.register_buffer("weight_q", torch.zeros(out_features, in_features, dtype=torch.int8))
            self.register_buffer("weight_scale", torch.ones(out_features, 1))

        if bias and bias_data is not None:
            self.register_buffer("bias", bias_data.clone())
        elif bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    def _quantize_weight(self, weight: torch.Tensor):
        """Rotate and quantize weight (offline)."""
        # Apply group-wise RHT to weight
        H = self.hadamard.to(dtype=weight.dtype, device=weight.device)
        weight_rotated = group_rht_weight(weight, group_size=self.group_size, hadamard_matrix=H)

        # Per-channel symmetric quantization (per-row of W)
        weight_q, weight_scale = symmetric_quantize(weight_rotated, bits=8, per_token=True)

        self.register_buffer("weight_q", weight_q)
        self.register_buffer("weight_scale", weight_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        H = self.hadamard.to(dtype=x.dtype, device=x.device)
        x_rotated = group_rht(x, group_size=self.group_size, hadamard_matrix=H)
        x_q, x_scale = symmetric_quantize(x_rotated, bits=8, per_token=True)

        # Dequantize-then-matmul (simulated INT8 GEMM; real INT8 needs cutlass/torchao kernels)
        x_deq = x_q.to(original_dtype) * x_scale.to(original_dtype)
        w_deq = self.weight_q.to(original_dtype) * self.weight_scale.to(original_dtype)
        result = x_deq @ w_deq.T

        if self.bias is not None:
            result = result + self.bias.to(original_dtype)

        return result

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 256,
    ) -> "ConvLinear8bit":
        """Convert a nn.Linear module to ConvLinear8bit."""
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            group_size=group_size,
            weight=linear.weight.data,
            bias_data=linear.bias.data if has_bias else None,
        )
        return layer


class ConvLinear4bit(nn.Module):
    """W4A4 quantized linear layer with ConvRot rotation.

    Weights are group-quantized to INT4 and stored packed (2 values / byte)
    so DiT parameter memory approaches the paper's ~4x reduction vs BF16.
    Forward still uses dequantize-then-matmul unless a fused INT4/NVFP4
    kernel is available (see README — paper speedup needs that path).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 256,
        weight_group_size: int = 128,
        weight: Optional[torch.Tensor] = None,
        bias_data: Optional[torch.Tensor] = None,
        act_per_token: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.weight_group_size = weight_group_size
        self.act_per_token = act_per_token

        H = get_regular_hadamard(group_size)
        self.register_buffer("hadamard", H)

        if weight is not None:
            self._quantize_weight(weight)
        else:
            self.register_buffer(
                "weight_packed",
                torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            )
            self.register_buffer(
                "weight_scale",
                torch.ones(out_features, in_features // weight_group_size),
            )

        if bias and bias_data is not None:
            self.register_buffer("bias", bias_data.clone())
        elif bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    @property
    def weight_q(self) -> torch.Tensor:
        """Unpacked signed INT4 weights in int8, shape (N, K)."""
        return unpack_int4(self.weight_packed, self.in_features)

    def _quantize_weight(self, weight: torch.Tensor):
        """Rotate, quantize to INT4, and pack (2 nibbles per byte)."""
        H = self.hadamard.to(dtype=weight.dtype, device=weight.device)
        weight_rotated = group_rht_weight(weight, group_size=self.group_size, hadamard_matrix=H)

        N, K = weight_rotated.shape
        wgs = self.weight_group_size
        num_groups = K // wgs

        weight_reshaped = weight_rotated.view(N, num_groups, wgs)
        amax = weight_reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / 7.0  # INT4 range: [-8, 7]

        weight_q = (weight_reshaped / scale).round().clamp(-8, 7).to(torch.int8)
        weight_q = weight_q.view(N, K)
        scale = scale.squeeze(-1)  # (N, num_groups)

        self.register_buffer("weight_packed", pack_int4(weight_q))
        self.register_buffer("weight_scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        H = self.hadamard.to(dtype=x.dtype, device=x.device)
        x_rotated = group_rht(x, group_size=self.group_size, hadamard_matrix=H)
        x_q, x_scale = symmetric_quantize(
            x_rotated, bits=4, per_token=self.act_per_token
        )

        # Unpack + dequantize-then-matmul (simulated W4A4 GEMM)
        weight_q = self.weight_q
        N, K = weight_q.shape
        wgs = self.weight_group_size
        num_groups = K // wgs

        w_deq = weight_q.view(N, num_groups, wgs).to(original_dtype) * self.weight_scale.unsqueeze(-1).to(original_dtype)
        w_deq = w_deq.view(N, K)
        x_deq = x_q.to(original_dtype) * x_scale.to(original_dtype)
        result = x_deq @ w_deq.T

        if self.bias is not None:
            result = result + self.bias.to(original_dtype)

        return result

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 256,
        weight_group_size: int = 128,
        act_per_token: bool = True,
    ) -> "ConvLinear4bit":
        """Convert a nn.Linear module to ConvLinear4bit."""
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            group_size=group_size,
            weight_group_size=weight_group_size,
            weight=linear.weight.data,
            bias_data=linear.bias.data if has_bias else None,
            act_per_token=act_per_token,
        )
        return layer
