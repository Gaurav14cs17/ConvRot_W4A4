"""
Regular Hadamard matrix construction via Kronecker product.

A Regular Hadamard matrix H_n has the property that every row and column
sums to ±√n, which gives minimal column discrepancy √n. This prevents
amplification of row-wise outliers during rotation (unlike Sylvester-type
matrices where column discrepancy = n).

Construction (Theorem 3.3 from paper):
  H_4 is the base regular matrix, H_{4^{k+1}} = H_{4^k} ⊗ H_4
  This yields regular matrices for all orders n = 4^k.
"""

import torch
from functools import lru_cache
from typing import Optional


H4_BASE = torch.tensor(
    [
        [1, 1, 1, -1],
        [1, 1, -1, 1],
        [1, -1, 1, 1],
        [-1, 1, 1, 1],
    ],
    dtype=torch.float32,
)


def regular_hadamard_matrix(order: int) -> torch.Tensor:
    """Construct a regular Hadamard matrix of given order.

    Args:
        order: Must be a power of 4 (4, 16, 64, 256, 1024, ...).

    Returns:
        Regular Hadamard matrix H of shape (order, order) such that
        H @ H.T = order * I and each row/column sums to ±√order.
    """
    if order < 4 or (order & (order - 1)) != 0:
        raise ValueError(f"Order must be a power of 4, got {order}")

    k = 0
    temp = order
    while temp > 1:
        if temp % 4 != 0 and temp != 1:
            # Check it's actually a power of 4, not just power of 2
            if temp % 2 == 0:
                raise ValueError(
                    f"Order must be a power of 4 (4, 16, 64, 256, ...), got {order}"
                )
        temp //= 4
        k += 1
        if temp == 1:
            break
        if temp < 1:
            raise ValueError(f"Order must be a power of 4, got {order}")

    # Verify it's a power of 4
    import math
    log4 = math.log2(order) / 2
    if log4 != int(log4):
        raise ValueError(f"Order must be a power of 4, got {order}")

    H = H4_BASE.clone()
    n = 4
    while n < order:
        H = torch.kron(H, H4_BASE)
        n *= 4

    return H


@lru_cache(maxsize=16)
def get_regular_hadamard(order: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Get a cached, normalized regular Hadamard matrix.

    Returns H / √order so that the matrix is orthogonal (H @ H.T = I).
    """
    import math
    H = regular_hadamard_matrix(order)
    H = H / math.sqrt(order)
    if device is not None:
        H = H.to(device)
    return H


def verify_regular(H: torch.Tensor) -> bool:
    """Verify that a matrix is a valid regular Hadamard matrix.

    Checks:
    1. All entries are ±1 (before normalization)
    2. H @ H.T = n * I (orthogonality)
    3. Each column sum has absolute value √n (regularity)
    """
    import math
    n = H.shape[0]
    assert H.shape == (n, n), "Matrix must be square"

    scale = math.sqrt(n)

    # Denormalize if needed
    H_int = H * scale if torch.allclose(H @ H.T, torch.eye(n, dtype=H.dtype), atol=1e-4) else H

    # Check ±1 entries
    if not torch.allclose(H_int.abs(), torch.ones_like(H_int), atol=1e-4):
        return False

    # Check orthogonality: H @ H.T = n * I
    product = H_int @ H_int.T
    expected = n * torch.eye(n, dtype=H_int.dtype)
    if not torch.allclose(product, expected, atol=1e-2):
        return False

    # Check regularity: each column sum = ±√n
    col_sums = H_int.sum(dim=0)
    sqrt_n = math.sqrt(n)
    if not torch.allclose(col_sums.abs(), torch.full((n,), sqrt_n, dtype=H_int.dtype), atol=1e-4):
        return False

    return True
