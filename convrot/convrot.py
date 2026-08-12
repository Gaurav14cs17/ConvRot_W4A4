"""
ConvRot: Group-wise Regular Hadamard Transform via matmul.

The key insight is that a group-wise Hadamard rotation of size N_0 can be
expressed as a convolution-like operation:
  - Reshape X from (M, K) to (M, K/N_0, N_0)
  - Multiply each block by the N_0 x N_0 Regular Hadamard matrix
  - Reshape back to (M, K)

This is equivalent to applying rotation within non-overlapping windows
of size N_0 along the channel dimension.

For the linear layer Y = X @ W.T:
  Y = sum_i RHT(X_i) @ RHT(W_i).T

where X_i are column blocks of size N_0.
"""

import torch
import math
from typing import Optional
from .hadamard import get_regular_hadamard


def group_rht(
    x: torch.Tensor,
    group_size: int = 256,
    hadamard_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply group-wise Regular Hadamard Transform to activations.

    Partitions the last dimension into groups of size `group_size` and
    applies a normalized regular Hadamard rotation within each group
    using batched matrix multiplication.

    Args:
        x: Input tensor of shape (..., K) where K is divisible by group_size.
        group_size: Size of each rotation group. Must be a power of 4.
        hadamard_matrix: Pre-computed normalized Hadamard matrix of shape
            (group_size, group_size). If None, constructed automatically.

    Returns:
        Rotated tensor of same shape as input.
    """
    K = x.shape[-1]
    if K % group_size != 0:
        raise ValueError(
            f"Last dimension ({K}) must be divisible by group_size ({group_size})"
        )

    if hadamard_matrix is None:
        hadamard_matrix = get_regular_hadamard(group_size, device=x.device)
    hadamard_matrix = hadamard_matrix.to(dtype=x.dtype, device=x.device)

    original_shape = x.shape
    num_groups = K // group_size

    # Reshape: (..., K) -> (..., num_groups, group_size)
    x = x.view(*original_shape[:-1], num_groups, group_size)

    # Batched matmul: each group is multiplied by H
    # (..., num_groups, group_size) @ (group_size, group_size) -> (..., num_groups, group_size)
    x = x @ hadamard_matrix.T

    # Reshape back: (..., num_groups, group_size) -> (..., K)
    x = x.view(original_shape)

    return x


def group_rht_weight(
    weight: torch.Tensor,
    group_size: int = 256,
    hadamard_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply group-wise RHT to weight matrix (offline, during quantization).

    For W of shape (N, K), partitions along K dimension and rotates each block.
    This is done once during model preparation and the rotated weights are stored.

    Args:
        weight: Weight tensor of shape (N, K).
        group_size: Size of each rotation group.
        hadamard_matrix: Pre-computed normalized Hadamard matrix.

    Returns:
        Rotated weight tensor of same shape.
    """
    return group_rht(weight, group_size=group_size, hadamard_matrix=hadamard_matrix)


def compute_outlier_amplitude(x: torch.Tensor) -> float:
    """Compute the outlier amplitude (max absolute value) of a tensor."""
    return x.abs().max().item()


def compare_outlier_reduction(
    x: torch.Tensor,
    group_sizes: list[int] = [16, 64, 256, 1024],
) -> dict:
    """Compare outlier amplitude before and after RHT at various group sizes.

    Useful for analyzing outlier suppression effectiveness.

    Returns:
        Dictionary mapping group_size -> (amplitude_after, reduction_pct)
    """
    original_amp = compute_outlier_amplitude(x)
    results = {"original": original_amp}

    for gs in group_sizes:
        if x.shape[-1] % gs != 0:
            continue
        rotated = group_rht(x, group_size=gs)
        amp = compute_outlier_amplitude(rotated)
        reduction = (1 - amp / original_amp) * 100
        results[gs] = (amp, reduction)

    return results
