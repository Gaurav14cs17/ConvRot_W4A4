"""Tests for ConvRot group-wise RHT and quantized linear layers."""

import torch
import torch.nn as nn
import math
import pytest
from convrot.convrot import group_rht, group_rht_weight, compute_outlier_amplitude
from convrot.conv_linear import ConvLinear8bit, ConvLinear4bit


class TestGroupRHT:
    """Test group-wise Regular Hadamard Transform."""

    def test_basic_shape(self):
        x = torch.randn(4, 256)
        y = group_rht(x, group_size=64)
        assert y.shape == x.shape

    def test_batch_shape(self):
        x = torch.randn(2, 8, 1024)
        y = group_rht(x, group_size=256)
        assert y.shape == x.shape

    def test_orthogonality_preserves_norm(self):
        """RHT should preserve the L2 norm of each row."""
        x = torch.randn(16, 256)
        y = group_rht(x, group_size=256)
        x_norms = x.norm(dim=-1)
        y_norms = y.norm(dim=-1)
        assert torch.allclose(x_norms, y_norms, atol=1e-4)

    def test_linear_equivalence(self):
        """Y = X @ W.T should equal sum_i RHT(X_i) @ RHT(W_i).T"""
        torch.manual_seed(42)
        M, K, N = 4, 256, 128
        X = torch.randn(M, K)
        W = torch.randn(N, K)

        # Original computation
        Y_orig = X @ W.T

        # With ConvRot: split into groups, rotate both, then matmul
        group_size = 64
        X_rot = group_rht(X, group_size=group_size)
        W_rot = group_rht_weight(W, group_size=group_size)
        Y_rot = X_rot @ W_rot.T

        assert torch.allclose(Y_orig, Y_rot, atol=1e-4)

    def test_reduces_outliers(self):
        """RHT should reduce outlier amplitude for typical activations."""
        torch.manual_seed(42)
        x = torch.randn(1, 1024)
        # Inject column-wise outliers
        x[0, 5] = 50.0
        x[0, 100] = -40.0

        original_amp = compute_outlier_amplitude(x)
        rotated = group_rht(x, group_size=256)
        rotated_amp = compute_outlier_amplitude(rotated)

        assert rotated_amp < original_amp

    def test_indivisible_raises(self):
        x = torch.randn(4, 100)
        with pytest.raises(ValueError, match="divisible"):
            group_rht(x, group_size=64)

    def test_different_group_sizes(self):
        x = torch.randn(4, 1024)
        for gs in [4, 16, 64, 256]:
            y = group_rht(x, group_size=gs)
            assert y.shape == x.shape


class TestConvLinear8bit:
    """Test W8A8 quantized linear layer."""

    def test_from_linear(self):
        linear = nn.Linear(256, 128, bias=False)
        conv8 = ConvLinear8bit.from_linear(linear, group_size=64)
        assert conv8.in_features == 256
        assert conv8.out_features == 128

    def test_forward_shape(self):
        linear = nn.Linear(256, 128, bias=False)
        conv8 = ConvLinear8bit.from_linear(linear, group_size=64)
        x = torch.randn(4, 256)
        y = conv8(x)
        assert y.shape == (4, 128)

    def test_forward_with_bias(self):
        linear = nn.Linear(256, 128, bias=True)
        conv8 = ConvLinear8bit.from_linear(linear, group_size=64)
        x = torch.randn(4, 256)
        y = conv8(x)
        assert y.shape == (4, 128)

    def test_numerical_accuracy(self):
        """W8A8 output should be close to float reference."""
        torch.manual_seed(42)
        linear = nn.Linear(256, 128, bias=False)
        conv8 = ConvLinear8bit.from_linear(linear, group_size=64)

        x = torch.randn(4, 256)
        y_ref = linear(x)
        y_q = conv8(x)

        # INT8 quantization error should be small
        relative_error = (y_ref - y_q).norm() / y_ref.norm()
        assert relative_error < 0.1  # Within 10% relative error

    def test_batch_input(self):
        linear = nn.Linear(256, 128, bias=False)
        conv8 = ConvLinear8bit.from_linear(linear, group_size=64)
        x = torch.randn(2, 8, 256)
        y = conv8(x)
        assert y.shape == (2, 8, 128)

    def test_memory_reduction(self):
        """INT8 weights should use less memory than FP32."""
        linear = nn.Linear(1024, 512, bias=False)
        conv8 = ConvLinear8bit.from_linear(linear, group_size=256)

        fp32_bytes = 1024 * 512 * 4  # float32
        int8_bytes = conv8.weight_q.nelement() * conv8.weight_q.element_size()
        assert int8_bytes < fp32_bytes


class TestConvLinear4bit:
    """Test W4A4 quantized linear layer."""

    def test_from_linear(self):
        linear = nn.Linear(256, 128, bias=False)
        conv4 = ConvLinear4bit.from_linear(linear, group_size=64)
        assert conv4.in_features == 256
        assert conv4.out_features == 128

    def test_forward_shape(self):
        linear = nn.Linear(256, 128, bias=False)
        conv4 = ConvLinear4bit.from_linear(linear, group_size=64)
        x = torch.randn(4, 256)
        y = conv4(x)
        assert y.shape == (4, 128)

    def test_forward_with_bias(self):
        linear = nn.Linear(256, 128, bias=True)
        conv4 = ConvLinear4bit.from_linear(linear, group_size=64)
        x = torch.randn(4, 256)
        y = conv4(x)
        assert y.shape == (4, 128)

    def test_weight_range_is_4bit(self):
        """Unpacked quantized weights should be in [-8, 7] range."""
        linear = nn.Linear(256, 128, bias=False)
        conv4 = ConvLinear4bit.from_linear(linear, group_size=64)
        wq = conv4.weight_q
        assert wq.min() >= -8
        assert wq.max() <= 7

    def test_numerical_accuracy(self):
        """W4A4 has more error than W8A8 but should still be reasonable."""
        torch.manual_seed(42)
        linear = nn.Linear(256, 128, bias=False)
        conv4 = ConvLinear4bit.from_linear(linear, group_size=64)

        x = torch.randn(4, 256)
        y_ref = linear(x)
        y_q = conv4(x)

        relative_error = (y_ref - y_q).norm() / y_ref.norm()
        assert relative_error < 0.3  # W4A4 allows more error

    def test_memory_reduction(self):
        """Packed INT4 weights should be ~4x smaller than BF16 / ~8x vs FP32."""
        linear = nn.Linear(1024, 512, bias=False)
        conv4 = ConvLinear4bit.from_linear(linear, group_size=256)

        fp32_bytes = 1024 * 512 * 4
        bf16_bytes = 1024 * 512 * 2
        packed_bytes = conv4.weight_packed.nelement() * conv4.weight_packed.element_size()
        assert packed_bytes == 1024 * 512 // 2  # 2 int4 values per byte
        assert packed_bytes < fp32_bytes
        assert packed_bytes == bf16_bytes // 4

    def test_pack_roundtrip(self):
        from convrot.conv_linear import pack_int4, unpack_int4
        w = torch.randint(-8, 8, (4, 16), dtype=torch.int8)
        packed = pack_int4(w)
        assert packed.dtype == torch.uint8
        assert packed.shape == (4, 8)
        assert torch.equal(unpack_int4(packed, 16), w)
