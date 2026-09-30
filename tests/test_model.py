"""
tests/test_model.py
====================
Pytest suite for src/training/model.py (LaneSegNet).

Coverage (Objective 6):
1. Forward pass produces correct output shape [B, 4, H, W]
2. Parameter count in expected range (3M-5M)
3. Handles batch sizes 1, 4, 8
4. Works on CPU and GPU (if CUDA available)
5. Encoder loads pretrained weights successfully
6. Attention modules (SEBlock & SpatialAttention) produce valid outputs
7. Auxiliary head can be toggled/disabled at inference
8. Predict helper returns correct class index tensor [B, H, W]
"""

import pytest
import torch
import torch.nn as nn

from src.training.model import (
    FPN_CHANNELS,
    NUM_CLASSES,
    FPNDecoder,
    LaneSegNet,
    LateralBlock,
    MobileNetV3Encoder,
    SEBlock,
    SpatialAttention,
)


@pytest.fixture(scope="module")
def model_scratch() -> LaneSegNet:
    """LaneSegNet initialized from scratch."""
    return LaneSegNet(pretrained=False).eval()


# ═══════════════════════════════════════════════════════════════════════════════
# Attention Modules Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAttentionModules:

    def test_se_block_output_shape(self):
        se = SEBlock(channels=64)
        x = torch.randn(2, 64, 45, 80)
        out = se(x)
        assert out.shape == x.shape

    def test_se_block_modulates_features(self):
        se = SEBlock(channels=32)
        x = torch.randn(1, 32, 10, 10)
        out = se(x)
        assert not torch.allclose(out, x), "SE block should modulate features"

    def test_spatial_attention_output_shape(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 64, 30, 40)
        out = sa(x)
        assert out.shape == x.shape

    def test_spatial_attention_modulates_features(self):
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(1, 16, 20, 20)
        out = sa(x)
        assert not torch.allclose(out, x), "Spatial attention should modulate features"


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMobileNetV3Encoder:

    @pytest.fixture()
    def encoder(self):
        return MobileNetV3Encoder(pretrained=False).eval()

    def test_returns_four_feature_maps(self, encoder):
        x = torch.randn(1, 3, 360, 640)
        feats = encoder(x)
        assert len(feats) == 4, f"Expected 4 feature maps, got {len(feats)}"

    def test_feature_map_channels(self, encoder):
        x = torch.randn(1, 3, 360, 640)
        feats = encoder(x)
        expected_channels = (24, 40, 48, 96)
        for i, (feat, exp_ch) in enumerate(zip(feats, expected_channels)):
            assert feat.shape[1] == exp_ch, f"Feature map {i}: expected {exp_ch}, got {feat.shape[1]}"


# ═══════════════════════════════════════════════════════════════════════════════
# FPNDecoder Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFPNDecoder:

    @pytest.fixture()
    def decoder(self):
        return FPNDecoder(fpn_ch=FPN_CHANNELS).eval()

    def _mock_feats(self, b=1, h=45, w=80):
        return [
            torch.randn(b, 24, h, w),
            torch.randn(b, 40, h // 2, w // 2),
            torch.randn(b, 48, h // 4, w // 4),
            torch.randn(b, 96, h // 8, w // 8),
        ]

    def test_output_shape(self, decoder):
        feats = self._mock_feats(b=2)
        out = decoder(feats)
        assert out.shape == (2, FPN_CHANNELS, 45, 80)


# ═══════════════════════════════════════════════════════════════════════════════
# LaneSegNet End-to-End Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneSegNet:

    @pytest.mark.parametrize("B,H,W", [
        (1, 360, 640),
        (2, 360, 640),
        (1, 224, 224),
    ])
    def test_forward_shape(self, model_scratch, B, H, W):
        x = torch.randn(B, 3, H, W)
        with torch.no_grad():
            out = model_scratch(x)
        assert out.shape == (B, NUM_CLASSES, H, W)

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_handles_batch_sizes(self, model_scratch, batch_size):
        x = torch.randn(batch_size, 3, 180, 320)
        with torch.no_grad():
            out = model_scratch(x)
        assert out.shape == (batch_size, NUM_CLASSES, 180, 320)

    def test_param_count_in_expected_range(self, model_scratch):
        """LaneSegNet should have between 3M and 5M parameters."""
        count = model_scratch.param_count()
        assert 3_000_000 <= count <= 5_000_000, f"Expected 3M-5M parameters, got {count:,}"

    def test_auxiliary_head_toggle(self, model_scratch):
        """Verify aux head can be toggled on for training and off for inference."""
        x = torch.randn(2, 3, 180, 320)

        # Inference: return_aux=False -> returns single tensor
        out_inf = model_scratch(x, return_aux=False)
        assert isinstance(out_inf, torch.Tensor)
        assert out_inf.shape == (2, NUM_CLASSES, 180, 320)

        # Training: return_aux=True -> returns (logits, aux_boundary)
        out_train = model_scratch(x, return_aux=True)
        assert isinstance(out_train, tuple) and len(out_train) == 2
        logits, aux = out_train
        assert logits.shape == (2, NUM_CLASSES, 180, 320)
        assert aux.shape == (2, 1, 180, 320)

    def test_predict_helper(self, model_scratch):
        x = torch.randn(2, 3, 180, 320)
        preds = model_scratch.predict(x)
        assert preds.shape == (2, 180, 320)
        assert preds.dtype == torch.int64
        assert preds.min() >= 0 and preds.max() < NUM_CLASSES

    def test_runs_on_cpu_and_gpu(self, model_scratch):
        """Must run cleanly on CPU, and GPU if CUDA is available."""
        x_cpu = torch.randn(1, 3, 180, 320)
        out_cpu = model_scratch(x_cpu)
        assert out_cpu is not None

        if torch.cuda.is_available():
            model_gpu = model_scratch.cuda()
            x_gpu = x_cpu.cuda()
            out_gpu = model_gpu(x_gpu)
            assert out_gpu.is_cuda
            model_scratch.cpu()

    def test_pretrained_encoder_instantiation(self):
        """LaneSegNet instantiates with pretrained=True without error."""
        model_pre = LaneSegNet(pretrained=True).eval()
        x = torch.randn(1, 3, 180, 320)
        with torch.no_grad():
            out = model_pre(x)
        assert out.shape == (1, NUM_CLASSES, 180, 320)
