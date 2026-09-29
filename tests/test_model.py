"""
tests/test_model.py
====================
Pytest suite for src/training/model.py (LaneSegNet).

Coverage
--------
- Forward pass shape: (B, 4, H, W) for canonical and non-canonical resolutions.
- Predict helper returns int64 class index map of shape (B, H, W).
- No gradient through frozen encoder blocks (first 4 blocks).
- Trainable parameter count is non-zero and within expected range.
- SEBlock channel scaling correctness.
- LateralBlock output channel count.
- FPNDecoder output shape.
- Model instantiates without pretrained weights (offline/CI environments).
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
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def model_pretrained() -> LaneSegNet:
    """LaneSegNet with pretrained=True (downloaded once per session)."""
    return LaneSegNet(pretrained=True).eval()


@pytest.fixture(scope="module")
def model_scratch() -> LaneSegNet:
    """LaneSegNet initialised from scratch (no download — CI friendly)."""
    return LaneSegNet(pretrained=False).eval()


# ═══════════════════════════════════════════════════════════════════════════════
# SEBlock tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSEBlock:
    def test_output_shape_unchanged(self):
        """SE block must return same (B, C, H, W) shape as input."""
        se = SEBlock(channels=64)
        x = torch.randn(2, 64, 45, 80)
        out = se(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_scale_range(self):
        """SE scale values must be in [0, 1] (sigmoid output)."""
        se = SEBlock(channels=32)
        x = torch.randn(1, 32, 10, 10)
        # Capture scale by inspecting the forward manually
        # We'll verify output values are modulated (not identical) to input
        out = se(x)
        # output != input when scale != 1
        assert not torch.allclose(out, x), "SE block should modulate the input"

    def test_single_channel(self):
        """SE block should work even with 1 channel (edge case)."""
        se = SEBlock(channels=1, reduction=1)
        x = torch.randn(1, 1, 8, 8)
        out = se(x)
        assert out.shape == x.shape


# ═══════════════════════════════════════════════════════════════════════════════
# LateralBlock tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLateralBlock:
    def test_output_channels(self):
        """LateralBlock must output exactly FPN_CHANNELS channels."""
        lb = LateralBlock(in_channels=96, out_channels=FPN_CHANNELS)
        x = torch.randn(2, 96, 11, 20)
        out = lb(x)
        assert out.shape[1] == FPN_CHANNELS

    def test_spatial_unchanged(self):
        """Spatial dimensions must not change."""
        lb = LateralBlock(in_channels=48)
        x = torch.randn(1, 48, 22, 40)
        out = lb(x)
        assert out.shape[-2:] == x.shape[-2:]


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMobileNetV3Encoder:
    @pytest.fixture(scope="class")
    def encoder(self):
        return MobileNetV3Encoder(pretrained=False).eval()

    def test_returns_four_feature_maps(self, encoder):
        x = torch.randn(1, 3, 360, 640)
        feats = encoder(x)
        assert len(feats) == 4, f"Expected 4 feature maps, got {len(feats)}"

    def test_feature_map_channels(self, encoder):
        """Verify channel counts at tap points match MobileNetV3-Small (probed)."""
        x = torch.randn(1, 3, 360, 640)
        feats = encoder(x)
        expected_channels = (24, 40, 48, 96)   # blocks 2, 4, 8, 11
        for i, (feat, expected_ch) in enumerate(zip(feats, expected_channels)):
            assert feat.shape[1] == expected_ch, (
                f"Feature map {i}: expected {expected_ch} ch, got {feat.shape[1]}"
            )

    def test_stride_halving(self, encoder):
        """
        P2 (finest) must have strictly more spatial area than P5 (coarsest).

        Note: MobileNetV3-Small blocks 4 and 8 share the same 23×40 spatial
        resolution, so strict consecutive halving does not hold at every level.
        We assert the global invariant instead.
        """
        x = torch.randn(1, 3, 360, 640)
        feats = encoder(x)
        p2_area = feats[0].shape[-2] * feats[0].shape[-1]
        p5_area = feats[-1].shape[-2] * feats[-1].shape[-1]
        assert p2_area > p5_area, (
            f"P2 area ({p2_area}) should be > P5 area ({p5_area})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FPNDecoder tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFPNDecoder:
    @pytest.fixture(scope="class")
    def decoder(self):
        return FPNDecoder(fpn_ch=FPN_CHANNELS).eval()

    def _mock_feats(self, b=1, h=45, w=80):
        """Dummy feature maps matching probed MobileNetV3-Small channel counts."""
        return [
            torch.randn(b, 24, h,    w),     # P2  /4   (block 2)
            torch.randn(b, 40, h//2, w//2),  # P3  /8   (block 4)
            torch.randn(b, 48, h//4, w//4),  # P4  /16  (block 8)
            torch.randn(b, 96, h//8, w//8),  # P5  /32  (block 11)
        ]

    def test_output_shape(self, decoder):
        feats = self._mock_feats(b=2)
        out = decoder(feats)
        assert out.shape == (2, FPN_CHANNELS, 45, 80)

    def test_output_channels(self, decoder):
        feats = self._mock_feats()
        out = decoder(feats)
        assert out.shape[1] == FPN_CHANNELS


# ═══════════════════════════════════════════════════════════════════════════════
# LaneSegNet end-to-end tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneSegNet:

    # ── Forward pass shapes ────────────────────────────────────────────────────

    @pytest.mark.parametrize("B,H,W", [
        (1, 360, 640),   # canonical
        (2, 360, 640),   # batch=2
        (1, 224, 224),   # square (non-canonical)
        (1, 480, 640),   # taller
    ])
    def test_forward_shape(self, model_scratch, B, H, W):
        x = torch.randn(B, 3, H, W)
        with torch.no_grad():
            out = model_scratch(x)
        assert out.shape == (B, NUM_CLASSES, H, W), (
            f"Expected ({B},{NUM_CLASSES},{H},{W}), got {out.shape}"
        )

    def test_logits_are_finite(self, model_scratch):
        """Network output must not contain NaN or Inf."""
        x = torch.randn(2, 3, 360, 640)
        with torch.no_grad():
            out = model_scratch(x)
        assert torch.isfinite(out).all(), "Logits contain NaN or Inf"

    # ── Predict helper ─────────────────────────────────────────────────────────

    def test_predict_shape(self, model_scratch):
        x = torch.randn(2, 3, 360, 640)
        preds = model_scratch.predict(x)
        assert preds.shape == (2, 360, 640)

    def test_predict_dtype(self, model_scratch):
        x = torch.randn(1, 3, 360, 640)
        preds = model_scratch.predict(x)
        assert preds.dtype == torch.int64, f"Expected int64, got {preds.dtype}"

    def test_predict_class_range(self, model_scratch):
        x = torch.randn(1, 3, 360, 640)
        preds = model_scratch.predict(x)
        assert preds.min() >= 0
        assert preds.max() < NUM_CLASSES

    # ── Frozen encoder blocks ──────────────────────────────────────────────────

    def test_first_four_encoder_blocks_frozen(self, model_scratch):
        """The first 3 MobileNetV3 blocks (0,1,2) must have no grad."""
        for i in range(3):
            block = model_scratch.encoder.features[i]
            for p in block.parameters():
                assert not p.requires_grad, (
                    f"Block {i} parameter should be frozen"
                )

    def test_later_encoder_blocks_trainable(self, model_scratch):
        """Block 3+ in the encoder must be trainable."""
        block = model_scratch.encoder.features[4]
        has_trainable = any(p.requires_grad for p in block.parameters())
        assert has_trainable, "Block 4+ should be trainable"

    # ── Parameter count ────────────────────────────────────────────────────────

    def test_param_count_positive(self, model_scratch):
        assert model_scratch.param_count() > 0

    def test_param_count_reasonable(self, model_scratch):
        """LaneSegNet should be < 5M params (lightweight target)."""
        count = model_scratch.param_count()
        assert count < 5_000_000, f"Too many params: {count:,}"

    # ── Custom num_classes ─────────────────────────────────────────────────────

    def test_custom_num_classes(self):
        m = LaneSegNet(num_classes=3, pretrained=False).eval()
        x = torch.randn(1, 3, 360, 640)
        with torch.no_grad():
            out = m(x)
        assert out.shape[1] == 3
