"""
tests/test_losses.py
=====================
Pytest suite for src/training/losses.py.

Coverage
--------
DiceLoss:
  - Output is a scalar.
  - Loss is in [0, 1] for valid inputs.
  - Perfect prediction → loss ~0.
  - All-wrong prediction → loss ~1.
  - class_weights tensor is applied (weighted != unweighted on imbalanced targets).

compute_class_weights:
  - Returns tensor of correct length.
  - All weights positive.
  - Rare classes get higher weight than frequent classes.
  - inverse_freq, inverse_sqrt_freq, median_freq strategies all work.
  - Zero total count → uniform weights returned.

CombinedLaneLoss:
  - Returns three-tuple (total, ce, dice).
  - Shapes are all scalar.
  - total ≈ 0.5*ce + 0.5*dice (within numerical tolerance).
  - Loss decreases on a step toward correct prediction.
  - update_class_weights works without raising.
  - ce_weight + dice_weight must equal 1 (assertion check).
"""

import pytest
import torch
import torch.nn.functional as F

from src.training.losses import (
    DEFAULT_CE_WEIGHT,
    DEFAULT_DICE_WEIGHT,
    NUM_CLASSES,
    CombinedLaneLoss,
    DiceLoss,
    compute_class_weights,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _perfect_logits(targets: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """
    Construct logits that are extremely confident on the correct class.
    Shape: (B, C, H, W).
    """
    B, H, W = targets.shape
    logits = torch.full((B, num_classes, H, W), -10.0)
    for c in range(num_classes):
        mask = (targets == c).unsqueeze(1)
        logits[:, c:c+1, :, :] = torch.where(mask, torch.tensor(10.0), logits[:, c:c+1, :, :])
    return logits


def _uniform_logits(B: int = 2, C: int = NUM_CLASSES, H: int = 8, W: int = 8) -> torch.Tensor:
    return torch.zeros(B, C, H, W)   # equal logits → uniform prob


def _uniform_targets(B: int = 2, H: int = 8, W: int = 8) -> torch.Tensor:
    return torch.zeros(B, H, W, dtype=torch.long)   # all background


# ═══════════════════════════════════════════════════════════════════════════════
# DiceLoss tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiceLoss:

    @pytest.fixture()
    def dice(self):
        return DiceLoss(num_classes=NUM_CLASSES)

    @pytest.fixture()
    def random_targets(self):
        return torch.randint(0, NUM_CLASSES, (2, 8, 8))

    # ── Shape ──────────────────────────────────────────────────────────────────

    def test_output_is_scalar(self, dice, random_targets):
        logits = torch.randn(2, NUM_CLASSES, 8, 8)
        loss = dice(logits, random_targets)
        assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"

    # ── Range ──────────────────────────────────────────────────────────────────

    def test_loss_in_unit_range(self, dice):
        torch.manual_seed(42)
        for _ in range(5):
            logits  = torch.randn(2, NUM_CLASSES, 16, 16)
            targets = torch.randint(0, NUM_CLASSES, (2, 16, 16))
            loss = dice(logits, targets)
            assert 0.0 <= loss.item() <= 1.05, (
                f"Dice loss out of [0,1]: {loss.item():.4f}"
            )

    # ── Perfect prediction ─────────────────────────────────────────────────────

    def test_perfect_prediction_near_zero(self, dice):
        targets = torch.randint(0, NUM_CLASSES, (2, 8, 8))
        logits  = _perfect_logits(targets)
        loss = dice(logits, targets)
        assert loss.item() < 0.05, (
            f"Perfect prediction should give Dice~0, got {loss.item():.4f}"
        )

    # ── All-wrong prediction ───────────────────────────────────────────────────

    def test_all_wrong_prediction_near_one(self):
        """When prediction is always maximally wrong, Dice loss should be near 1."""
        # Single class target: all background (0)
        targets = torch.zeros(2, 8, 8, dtype=torch.long)
        # Predict class 1 everywhere with high confidence
        logits = torch.full((2, NUM_CLASSES, 8, 8), -10.0)
        logits[:, 1, :, :] = 10.0
        dice = DiceLoss(num_classes=NUM_CLASSES)
        loss = dice(logits, targets)
        # Class 0 Dice should be near 0, so loss (1-Dice) near 1
        # With macro averaging, classes 1-3 get Dice=0 too → average loss high
        assert loss.item() > 0.5, (
            f"All-wrong prediction should give high Dice loss, got {loss.item():.4f}"
        )

    # ── Class weights ──────────────────────────────────────────────────────────

    def test_class_weights_affect_loss(self):
        """Weighted and unweighted Dice should differ on imbalanced targets."""
        targets = torch.zeros(2, 16, 16, dtype=torch.long)  # all background
        targets[:, 8:, :] = 2   # half the pixels are left_lane (class 2)
        logits  = torch.randn(2, NUM_CLASSES, 16, 16)

        dice_uniform  = DiceLoss(num_classes=NUM_CLASSES)
        # Weight class 2 heavily
        weights = torch.tensor([1.0, 1.0, 5.0, 1.0])
        dice_weighted = DiceLoss(num_classes=NUM_CLASSES, class_weights=weights)

        loss_u = dice_uniform(logits, targets)
        loss_w = dice_weighted(logits, targets)
        assert not torch.isclose(loss_u, loss_w, atol=1e-4), (
            "Weighted and unweighted Dice should differ on imbalanced target"
        )

    def test_dtype_float32(self, dice, random_targets):
        logits = torch.randn(2, NUM_CLASSES, 8, 8)
        loss = dice(logits, random_targets)
        assert loss.dtype == torch.float32


# ═══════════════════════════════════════════════════════════════════════════════
# compute_class_weights tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeClassWeights:

    @pytest.fixture()
    def imbalanced_counts(self):
        """Simulate realistic class imbalance: background 50×, lanes 1× each."""
        return {0: 50_000, 1: 15_000, 2: 1_000, 3: 1_000}

    # ── Basic properties ───────────────────────────────────────────────────────

    def test_output_length(self, imbalanced_counts):
        w = compute_class_weights(imbalanced_counts, num_classes=4)
        assert w.shape == (4,)

    def test_all_weights_positive(self, imbalanced_counts):
        w = compute_class_weights(imbalanced_counts)
        assert (w > 0).all()

    def test_output_dtype(self, imbalanced_counts):
        w = compute_class_weights(imbalanced_counts)
        assert w.dtype == torch.float32

    # ── Rare classes get higher weight ────────────────────────────────────────

    def test_lane_classes_heavier_than_background(self, imbalanced_counts):
        w = compute_class_weights(imbalanced_counts, strategy="inverse_freq")
        # Classes 2 and 3 (rare lanes) should outweigh class 0 (background)
        assert w[2] > w[0], f"w[2]={w[2]:.3f} should > w[0]={w[0]:.3f}"
        assert w[3] > w[0], f"w[3]={w[3]:.3f} should > w[0]={w[0]:.3f}"

    # ── Strategies ────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("strategy", ["inverse_freq", "inverse_sqrt_freq", "median_freq"])
    def test_all_strategies_produce_positive_weights(self, imbalanced_counts, strategy):
        w = compute_class_weights(imbalanced_counts, strategy=strategy)
        assert (w > 0).all(), f"Strategy '{strategy}' produced non-positive weights"

    def test_invalid_strategy_raises(self, imbalanced_counts):
        with pytest.raises(ValueError, match="Unknown strategy"):
            compute_class_weights(imbalanced_counts, strategy="bad_strategy")

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_zero_total_gives_uniform_weights(self):
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        w = compute_class_weights(counts)
        assert torch.allclose(w, torch.ones(4)), "Zero total should give uniform weights"


# ═══════════════════════════════════════════════════════════════════════════════
# CombinedLaneLoss tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCombinedLaneLoss:

    @pytest.fixture()
    def criterion(self):
        return CombinedLaneLoss()

    @pytest.fixture()
    def batch(self):
        logits  = torch.randn(2, NUM_CLASSES, 16, 16)
        targets = torch.randint(0, NUM_CLASSES, (2, 16, 16))
        return logits, targets

    # ── Return type and shapes ─────────────────────────────────────────────────

    def test_returns_three_tuple(self, criterion, batch):
        out = criterion(*batch)
        assert len(out) == 3, f"Expected 3-tuple, got {len(out)} items"

    def test_all_three_are_scalars(self, criterion, batch):
        total, ce, dice = criterion(*batch)
        for name, val in [("total", total), ("ce", ce), ("dice", dice)]:
            assert val.shape == torch.Size([]), f"{name} is not a scalar"

    def test_all_losses_finite(self, criterion, batch):
        total, ce, dice = criterion(*batch)
        for name, val in [("total", total), ("ce", ce), ("dice", dice)]:
            assert torch.isfinite(val), f"{name} is not finite: {val}"

    def test_total_is_weighted_sum(self, criterion, batch):
        logits, targets = batch
        out = criterion(logits, targets)
        total, ce_d, dice_d = out

        # Total = ce_w * ce + dice_w * dice + boundary_w * boundary
        expected = (
            criterion.ce_weight * ce_d.item()
            + criterion.dice_weight * dice_d.item()
            + criterion.boundary_weight * out.boundary.item()
        )
        assert abs(total.item() - expected) < 1e-4, (
            f"total={total.item():.5f} ≠ {expected:.5f}"
        )


    # ── Gradient flow ─────────────────────────────────────────────────────────

    def test_gradient_flows_to_logits(self, criterion):
        logits  = torch.randn(2, NUM_CLASSES, 16, 16, requires_grad=True)
        targets = torch.randint(0, NUM_CLASSES, (2, 16, 16))
        total, _, _ = criterion(logits, targets)
        total.backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape

    # ── Loss decreases toward correct prediction ───────────────────────────────

    def test_loss_lower_on_correct_logits(self, criterion):
        targets = torch.randint(0, NUM_CLASSES, (2, 16, 16))
        random_logits  = torch.randn(2, NUM_CLASSES, 16, 16)
        correct_logits = _perfect_logits(targets)

        loss_random,  _, _ = criterion(random_logits,  targets)
        loss_correct, _, _ = criterion(correct_logits, targets)

        assert loss_correct.item() < loss_random.item(), (
            f"Correct logits (loss={loss_correct.item():.4f}) should beat "
            f"random (loss={loss_random.item():.4f})"
        )

    # ── Focal loss variant on lane classes ────────────────────────────────────

    def test_focal_loss_reduces_easy_example_weights(self):
        """Focal modulation (1 - p_t)^2 should downweight confident correct lane pixels."""
        from src.training.losses import FocalCrossEntropyLoss
        focal_loss = FocalCrossEntropyLoss(gamma=2.0)

        # Target is lane class 2
        targets = torch.full((1, 8, 8), 2, dtype=torch.long)

        # Easy (high confidence p=0.99) vs Hard (uncertain p=0.40)
        easy_logits = torch.full((1, NUM_CLASSES, 8, 8), -5.0)
        easy_logits[:, 2, :, :] = 10.0   # p_t ~ 0.9999

        hard_logits = torch.zeros((1, NUM_CLASSES, 8, 8))  # uniform p_t = 0.25

        loss_easy = focal_loss(easy_logits, targets)
        loss_hard = focal_loss(hard_logits, targets)

        assert loss_easy.item() < 0.01
        assert loss_hard.item() > 0.5
        assert loss_hard.item() > (loss_easy.item() * 50)

    # ── Boundary loss penalizes misalignment ───────────────────────────────────

    def test_boundary_loss_penalizes_misalignment(self):
        """Boundary loss must be lower for well-aligned lane masks than misaligned masks."""
        from src.training.losses import BoundaryLoss
        b_loss = BoundaryLoss()

        targets = torch.zeros((1, 32, 32), dtype=torch.long)
        targets[:, :, 14:18] = 2  # Left lane in center

        # Aligned logits
        aligned_logits = torch.full((1, NUM_CLASSES, 32, 32), -5.0)
        aligned_logits[:, 2, :, 14:18] = 5.0

        # Misaligned logits (shifted laterally by 6 pixels)
        shifted_logits = torch.full((1, NUM_CLASSES, 32, 32), -5.0)
        shifted_logits[:, 2, :, 20:24] = 5.0

        loss_aligned = b_loss(aligned_logits, targets)
        loss_shifted = b_loss(shifted_logits, targets)

        assert loss_aligned.item() < loss_shifted.item()

    # ── Auxiliary boundary loss ───────────────────────────────────────────────

    def test_auxiliary_loss_supervision(self, criterion):
        logits = torch.randn(2, NUM_CLASSES, 16, 16)
        targets = torch.randint(0, NUM_CLASSES, (2, 16, 16))
        aux_logits = torch.randn(2, 1, 16, 16)

        out = criterion(logits, targets, aux_logits=aux_logits)
        assert out.aux is not None
        assert torch.isfinite(out.aux)
        assert out.total.item() > 0

    # ── Class weights ─────────────────────────────────────────────────────────

    def test_update_class_weights_no_error(self, criterion):
        weights = torch.tensor([1.0, 2.0, 5.0, 5.0])
        criterion.update_class_weights(weights)  # Should not raise

    def test_class_weighted_criterion_instantiation(self):
        weights = compute_class_weights(
            {0: 50_000, 1: 15_000, 2: 1_000, 3: 1_000}
        )
        criterion = CombinedLaneLoss(class_weights=weights)
        logits  = torch.randn(2, NUM_CLASSES, 16, 16)
        targets = torch.randint(0, NUM_CLASSES, (2, 16, 16))
        total, ce, dice = criterion(logits, targets)
        assert torch.isfinite(total)

    # ── Assertion on invalid weights ──────────────────────────────────────────

    def test_invalid_weight_split_raises(self):
        with pytest.raises(AssertionError):
            CombinedLaneLoss(ce_weight=0.6, dice_weight=0.6, boundary_weight=0.2)  # sums to 1.4

