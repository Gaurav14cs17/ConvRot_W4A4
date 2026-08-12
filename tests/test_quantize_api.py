"""Tests for the model-level quantization API."""

import torch
import torch.nn as nn
import pytest
from convrot.quantize import convrot_quantize_, ConvRotConfig, get_model_size_mb
from convrot.conv_linear import ConvLinear8bit, ConvLinear4bit


class SimpleModel(nn.Module):
    """Simple model for testing quantization."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(256, 512, bias=False)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(512, 256, bias=False)
        self.layer3 = nn.Linear(256, 64, bias=True)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)


class TestConvRotConfig:
    def test_valid_config(self):
        config = ConvRotConfig(precision="w8a8", group_size=256)
        assert config.precision == "w8a8"
        assert config.group_size == 256

    def test_invalid_precision(self):
        with pytest.raises(ValueError, match="precision"):
            ConvRotConfig(precision="w2a2")

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="power of 4"):
            ConvRotConfig(group_size=128)  # Power of 2 but not 4

    def test_valid_group_sizes(self):
        for gs in [4, 16, 64, 256, 1024]:
            config = ConvRotConfig(group_size=gs)
            assert config.group_size == gs


class TestQuantizeAPI:
    def test_w8a8_quantization(self):
        model = SimpleModel()
        config = ConvRotConfig(precision="w8a8", group_size=64)
        convrot_quantize_(model, config)

        assert isinstance(model.layer1, ConvLinear8bit)
        assert isinstance(model.layer2, ConvLinear8bit)
        assert isinstance(model.layer3, ConvLinear8bit)

    def test_w4a4_quantization(self):
        model = SimpleModel()
        config = ConvRotConfig(precision="w4a4", group_size=64)
        convrot_quantize_(model, config)

        assert isinstance(model.layer1, ConvLinear4bit)
        assert isinstance(model.layer2, ConvLinear4bit)
        assert isinstance(model.layer3, ConvLinear4bit)

    def test_mixed_precision(self):
        model = SimpleModel()
        config = ConvRotConfig(
            precision="w4a4",
            group_size=64,
            mixed_precision_layers=["layer2"],
        )
        convrot_quantize_(model, config)

        assert isinstance(model.layer1, ConvLinear4bit)
        assert isinstance(model.layer2, ConvLinear8bit)  # Mixed precision
        assert isinstance(model.layer3, ConvLinear4bit)

    def test_skip_layers(self):
        model = SimpleModel()
        config = ConvRotConfig(
            precision="w8a8",
            group_size=64,
            skip_layers=["layer3"],
        )
        convrot_quantize_(model, config)

        assert isinstance(model.layer1, ConvLinear8bit)
        assert isinstance(model.layer2, ConvLinear8bit)
        assert isinstance(model.layer3, nn.Linear)  # Skipped

    def test_filter_fn(self):
        model = SimpleModel()
        config = ConvRotConfig(
            precision="w8a8",
            group_size=64,
            filter_fn=lambda m, fqn: m.in_features >= 256,
        )
        convrot_quantize_(model, config)

        assert isinstance(model.layer1, ConvLinear8bit)
        assert isinstance(model.layer2, ConvLinear8bit)
        assert isinstance(model.layer3, ConvLinear8bit)

    def test_quantized_model_runs(self):
        model = SimpleModel()
        config = ConvRotConfig(precision="w8a8", group_size=64)
        convrot_quantize_(model, config)

        x = torch.randn(4, 256)
        y = model(x)
        assert y.shape == (4, 64)

    def test_memory_reduction(self):
        model = SimpleModel()
        size_before = get_model_size_mb(model)

        config = ConvRotConfig(precision="w8a8", group_size=64)
        convrot_quantize_(model, config)
        size_after = get_model_size_mb(model)

        # INT8 should be roughly 4x smaller than FP32 weights
        assert size_after < size_before

    def test_indivisible_layers_skipped(self):
        """Layers whose in_features aren't divisible by group_size should be skipped."""
        model = nn.Sequential(
            nn.Linear(100, 256, bias=False),  # 100 not divisible by 64
            nn.Linear(256, 64, bias=False),   # 256 divisible by 64
        )
        config = ConvRotConfig(precision="w8a8", group_size=64)
        convrot_quantize_(model, config)

        assert isinstance(model[0], nn.Linear)      # Skipped
        assert isinstance(model[1], ConvLinear8bit)  # Quantized
