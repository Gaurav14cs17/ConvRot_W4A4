"""Tests for Regular Hadamard matrix construction."""

import torch
import math
import pytest
from convrot.hadamard import regular_hadamard_matrix, get_regular_hadamard, verify_regular


class TestRegularHadamardMatrix:
    """Test the Kronecker construction of Regular Hadamard matrices."""

    @pytest.mark.parametrize("order", [4, 16, 64, 256, 1024])
    def test_correct_size(self, order):
        H = regular_hadamard_matrix(order)
        assert H.shape == (order, order)

    @pytest.mark.parametrize("order", [4, 16, 64, 256])
    def test_entries_are_pm1(self, order):
        H = regular_hadamard_matrix(order)
        assert torch.allclose(H.abs(), torch.ones_like(H))

    @pytest.mark.parametrize("order", [4, 16, 64, 256])
    def test_orthogonality(self, order):
        """H @ H.T = n * I"""
        H = regular_hadamard_matrix(order)
        product = H @ H.T
        expected = order * torch.eye(order)
        assert torch.allclose(product, expected, atol=1e-4)

    @pytest.mark.parametrize("order", [4, 16, 64, 256])
    def test_regularity(self, order):
        """Each column sum should be ±√n."""
        H = regular_hadamard_matrix(order)
        col_sums = H.sum(dim=0)
        sqrt_n = math.sqrt(order)
        assert torch.allclose(col_sums.abs(), torch.full((order,), sqrt_n), atol=1e-4)

    @pytest.mark.parametrize("order", [4, 16, 64, 256])
    def test_row_regularity(self, order):
        """Each row sum should also be ±√n (regularity is symmetric)."""
        H = regular_hadamard_matrix(order)
        row_sums = H.sum(dim=1)
        sqrt_n = math.sqrt(order)
        assert torch.allclose(row_sums.abs(), torch.full((order,), sqrt_n), atol=1e-4)

    @pytest.mark.parametrize("order", [4, 16, 64, 256])
    def test_column_discrepancy_minimal(self, order):
        """Column discrepancy should equal √n (minimal possible)."""
        H = regular_hadamard_matrix(order)
        col_sums = H.sum(dim=0)
        discrepancy = col_sums.abs().max().item()
        assert abs(discrepancy - math.sqrt(order)) < 1e-4

    def test_invalid_order_not_power_of_4(self):
        with pytest.raises(ValueError):
            regular_hadamard_matrix(8)  # Power of 2 but not 4

    def test_invalid_order_not_power(self):
        with pytest.raises(ValueError):
            regular_hadamard_matrix(12)

    def test_invalid_order_too_small(self):
        with pytest.raises(ValueError):
            regular_hadamard_matrix(2)

    @pytest.mark.parametrize("order", [4, 16, 64, 256])
    def test_verify_regular_function(self, order):
        H = regular_hadamard_matrix(order)
        assert verify_regular(H)

    def test_normalized_is_orthogonal(self):
        """get_regular_hadamard returns H/√n which should satisfy H @ H.T = I."""
        H_norm = get_regular_hadamard(64)
        product = H_norm @ H_norm.T
        assert torch.allclose(product, torch.eye(64), atol=1e-5)


class TestComparisonWithSylvester:
    """Compare Regular vs Sylvester Hadamard for outlier behavior."""

    def _sylvester_hadamard(self, order):
        """Construct Sylvester-type Hadamard matrix."""
        H = torch.tensor([[1.0]])
        n = 1
        while n < order:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
            n *= 2
        return H

    def test_sylvester_has_worse_discrepancy(self):
        """Sylvester-type H has max column sum = n (worst case)."""
        H_syl = self._sylvester_hadamard(16)
        col_sums = H_syl.sum(dim=0)
        # Sylvester has first column all 1s, so max col sum = n
        assert col_sums.abs().max().item() == 16.0

        H_reg = regular_hadamard_matrix(16)
        col_sums_reg = H_reg.sum(dim=0)
        # Regular has max col sum = √n
        assert col_sums_reg.abs().max().item() == pytest.approx(4.0, abs=1e-4)

    def test_regular_suppresses_row_outliers_better(self):
        """Regular Hadamard should not amplify row-wise outliers.

        Row-wise outliers means many elements in the same row are large.
        Sylvester's all-ones column sums them up, creating a huge single
        output element. Regular distributes more evenly.
        """
        torch.manual_seed(42)
        # Create activation with row-wise outliers (many large values in one row)
        x = torch.randn(1, 256)
        # Make ALL entries large (simulating a row-wise outlier pattern)
        x = x + 10.0  # Shift mean to 10 — all positive, large mean

        H_reg = get_regular_hadamard(256)
        H_syl = self._sylvester_hadamard(256) / math.sqrt(256)

        x_reg = x @ H_reg.T
        x_syl = x @ H_syl.T

        # Sylvester's first column is all-ones, so it sums all 256 values
        # (each ~10) giving ~10*√256 = 160 in one output element.
        # Regular has column sums = ±√256, so max output ≈ 10*√256/√256 = 10
        assert x_reg.abs().max() < x_syl.abs().max()
