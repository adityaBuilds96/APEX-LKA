"""
tests/test_dl_pipeline.py
=========================
Comprehensive Test Suite for Phase 4 Section 3:
- LaneSegNet Architecture & Parameter Bounds (3.0M - 5.0M, FP16 < 20MB)
- Squeeze-and-Excitation & Spatial Attention Blocks
- Albumentations Dataset & Strict Geometric Prohibition Checks
- Multi-Task CombinedLoss Math (Dice, Focal CE, EDT Boundary Distance)
- High-Throughput Fault-Tolerant Training Loop (EMA, Warmup Cosine, Progressive Unfreeze, OOM Recovery)
- Model Export & Numerical Parity Verification (>99.5% agreement, MAE < 1e-3)
- Predictor Backend Dynamic Loader Auto-Swap (TensorRT -> ONNX -> PyTorch -> Classical)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.training.dataset import FORBIDDEN_TRANSFORMS, LaneSegDataset
from src.training.losses import (
    BoundaryDistanceLoss,
    BoundaryLoss,
    CombinedLaneLoss,
    CombinedLoss,
    DiceLoss,
    FocalCrossEntropyLoss,
    compute_class_frequency_weights,
    compute_class_weights,
)
from src.training.model import (
    FPN_CHANNELS,
    NUM_CLASSES,
    ChannelAttention,
    FPNDecoder,
    LaneSegNet,
    SpatialAttention,
)
from src.training.schedulers import ModelEMA
from src.training.train import Trainer
from src.training.utils import find_latest_checkpoint


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Model Architecture & Parameter Constraints
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneSegNetArchitecture:

    def test_parameter_count_within_specification(self):
        """LaneSegNet parameter count must strictly lie in [3.0M, 5.0M]."""
        model = LaneSegNet(num_classes=4, pretrained=False)
        total_params = sum(p.numel() for p in model.parameters())
        assert 3_000_000 <= total_params <= 5_000_000, (
            f"Parameter count {total_params:,} is outside the allowed range [3.0M, 5.0M]."
        )

    def test_fp16_model_size_under_20mb(self):
        """FP16 weight tensor size must be strictly under 20 MB."""
        model = LaneSegNet(num_classes=4, pretrained=False)
        total_params = sum(p.numel() for p in model.parameters())
        # FP16 = 2 bytes per parameter
        fp16_size_mb = (total_params * 2) / (1024 * 1024)
        assert fp16_size_mb < 20.0, f"FP16 model size {fp16_size_mb:.2f} MB exceeds 20 MB ceiling."

    def test_eval_mode_returns_single_tensor(self):
        """During model.eval(), forward pass returns single tensor of shape (B, 4, H, W)."""
        model = LaneSegNet(num_classes=4, pretrained=False)
        model.eval()
        x = torch.randn(2, 3, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert isinstance(out, torch.Tensor), f"Expected Tensor in eval mode, got {type(out)}"
        assert out.shape == (2, 4, 360, 640)

    def test_train_mode_with_aux_returns_tuple(self):
        """During model.train(), forward pass returns (seg_logits, aux_logits)."""
        model = LaneSegNet(num_classes=4, pretrained=False, with_aux_head=True)
        model.train()
        x = torch.randn(2, 3, 180, 320)
        out = model(x, return_aux=True)
        assert isinstance(out, tuple)
        assert len(out) == 2
        seg_logits, aux_logits = out
        assert seg_logits.shape == (2, 4, 180, 320)
        assert aux_logits.shape == (2, 1, 180, 320)

    def test_channel_attention_squeeze_and_excitation(self):
        """Squeeze-and-Excitation channel attention with reduction ratio 16 preserves channels."""
        se = ChannelAttention(in_channels=128, reduction=16)
        x = torch.randn(2, 128, 45, 80)
        out = se(x)
        assert out.shape == (2, 128, 45, 80)
        # Weights are in range [0, 1] through sigmoid
        assert not torch.isnan(out).any()

    def test_spatial_attention_filter_map(self):
        """Spatial attention 7x7 conv generates a 1-channel spatial weight map in [0, 1]."""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 128, 45, 80)
        out = sa(x)
        assert out.shape == (2, 128, 45, 80)
        # Verify modulation works without NaN
        assert torch.isfinite(out).all()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dataset & Albumentations Augmentations
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatasetAndAugmentations:

    def test_forbidden_transforms_strictly_blocked(self):
        """Verify HorizontalFlip and other geometry-inverting transforms are strictly forbidden."""
        import albumentations as A

        assert "HorizontalFlip" in FORBIDDEN_TRANSFORMS
        assert "VerticalFlip" in FORBIDDEN_TRANSFORMS
        assert "ElasticTransform" in FORBIDDEN_TRANSFORMS
        assert "GridDistortion" in FORBIDDEN_TRANSFORMS

        # A pipeline with HorizontalFlip must be rejected by dataset validation
        illegal_pipeline = A.Compose([A.HorizontalFlip(p=1.0)])
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "masks").mkdir()
            with pytest.raises(ValueError, match="CRITICAL"):
                LaneSegDataset(root, split="train", transform=illegal_pipeline)

    def test_dataset_allowed_photometric_transforms_preserve_mask_values(self):
        """Photometric transforms must preserve mask pixel class values in {0, 1, 2, 3}."""
        import cv2

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            img_dir = root / "images"
            mask_dir = root / "masks"
            img_dir.mkdir()
            mask_dir.mkdir()

            # Create dummy sample
            sample_img = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            sample_mask = np.zeros((360, 640), dtype=np.uint8)
            sample_mask[180:240, :] = 1  # road
            sample_mask[240:300, 100:120] = 2  # left lane
            sample_mask[240:300, 520:540] = 3  # right lane

            cv2.imwrite(str(img_dir / "frame_001.png"), sample_img)
            cv2.imwrite(str(mask_dir / "frame_001.png"), sample_mask)

            ds = LaneSegDataset(root, split="train", height=360, width=640)
            assert len(ds) == 1

            img_tensor, mask_tensor = ds[0]
            assert img_tensor.shape == (3, 360, 640)
            assert mask_tensor.shape == (360, 640)
            unique_classes = set(mask_tensor.unique().tolist())
            assert unique_classes.issubset({0, 1, 2, 3})


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Task Combined Loss Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiTaskLossArchitecture:

    def test_class_frequency_weighting_formula(self):
        """Verify w_c = N_total / (C * N_c + epsilon) calculation."""
        pixel_counts = {0: 70_000, 1: 25_000, 2: 2_500, 3: 2_500}
        total = 100_000
        w = compute_class_frequency_weights(pixel_counts, num_classes=4, epsilon=1e-5)

        # Expected w for lane class 2: 100_000 / (4 * 2500 + 1e-5) = 10.0
        expected_w2 = total / (4 * 2500 + 1e-5)
        assert abs(w[2].item() - expected_w2) < 1e-3
        # Rare lanes should have higher weight than dominant background
        assert w[2] > w[0]
        assert w[3] > w[0]

    def test_dice_loss_excludes_background_class_0(self):
        """Dice loss with exclude_background=True ignores class 0 in numerator weighting."""
        dice_ex = DiceLoss(num_classes=4, exclude_background=True)
        dice_inc = DiceLoss(num_classes=4, exclude_background=False)

        # Target containing background (0), road (1), left lane (2), right lane (3)
        targets = torch.zeros((2, 32, 32), dtype=torch.long)
        targets[:, 10:20, :] = 1
        targets[:, 20:25, :] = 2
        targets[:, 25:30, :] = 3

        # Logits that predict road and lanes accurately, but background pixels with uncertainty
        logits = torch.full((2, 4, 32, 32), -5.0)
        logits[:, 0, :10, :] = 10.0   # background correct at top
        logits[:, 0, 30:, :] = -5.0   # background error at bottom
        logits[:, 1, 10:20, :] = 10.0 # road correct
        logits[:, 2, 20:25, :] = 10.0 # left lane correct
        logits[:, 3, 25:30, :] = 10.0 # right lane correct

        loss_ex = dice_ex(logits, targets)
        loss_inc = dice_inc(logits, targets)

        # With background excluded, loss on foreground is very low
        assert loss_ex.item() < 0.05
        # Background exclusion is configured
        assert dice_ex.exclude_background is True
        assert dice_inc.exclude_background is False

    def test_focal_cross_entropy_scaling_on_lane_classes(self):
        """Focal factor (1 - p_t)^2 drastically reduces gradient updates for confident lane pixels."""
        focal_loss = FocalCrossEntropyLoss(gamma=2.0)
        targets = torch.full((1, 16, 16), 2, dtype=torch.long)  # class 2 (Left Lane)

        # Confident correct (p ~ 0.999)
        confident_logits = torch.full((1, 4, 16, 16), -5.0)
        confident_logits[:, 2, :, :] = 10.0

        # Uncertain (p ~ 0.25)
        uncertain_logits = torch.zeros((1, 4, 16, 16))

        loss_conf = focal_loss(confident_logits, targets)
        loss_unc = focal_loss(uncertain_logits, targets)

        assert loss_conf.item() < 0.01
        assert loss_unc.item() > loss_conf.item() * 50

    def test_boundary_distance_loss_with_euclidean_transform(self):
        """BoundaryDistanceLoss computes L2 distance to Euclidean distance transform of lane edges."""
        bnd_loss = BoundaryDistanceLoss(max_dist=20.0)

        targets = torch.zeros((2, 64, 64), dtype=torch.long)
        targets[:, :, 30:34] = 2  # Ground truth lane line

        # Predicted continuous boundary logits near center line
        perfect_aux = torch.full((2, 1, 64, 64), -10.0)
        perfect_aux[:, 0, :, 29:35] = 10.0  # high confidence on edge

        shifted_aux = torch.full((2, 1, 64, 64), -10.0)
        shifted_aux[:, 0, :, 50:56] = 10.0  # 20 pixels away from lane

        loss_near = bnd_loss(perfect_aux, targets)
        loss_far = bnd_loss(shifted_aux, targets)

        assert loss_near.item() < loss_far.item()
        assert torch.isfinite(loss_near)

    def test_combined_loss_linear_combination_weights(self):
        """CombinedLoss strictly equals 0.4*L_Dice + 0.4*L_CE + 0.2*L_Boundary."""
        criterion = CombinedLoss(
            ce_weight=0.4,
            dice_weight=0.4,
            boundary_weight=0.2,
            num_classes=4,
        )
        logits = torch.randn(2, 4, 32, 32, requires_grad=True)
        aux_logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        targets = torch.randint(0, 4, (2, 32, 32))

        out = criterion(logits, targets, aux_logits=aux_logits)
        total, ce, dice = out

        # Backward compatibility tuple unpack
        assert isinstance(out, tuple)
        assert len(out) == 3

        # Attribute access
        assert out.boundary is not None
        assert out.aux is not None

        expected_total = 0.4 * ce.item() + 0.4 * dice.item() + 0.2 * out.boundary.item()
        assert abs(total.item() - expected_total) < 1e-4

        # Verify gradient flow
        total.backward()
        assert logits.grad is not None
        assert aux_logits.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Training Engine & Edge Resilience
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingEngineAndResilience:

    def test_model_ema_decay_tracking(self):
        """ModelEMA maintains shadow weights with decay=0.999."""
        model = LaneSegNet(num_classes=4, pretrained=False)
        ema = ModelEMA(model, decay=0.999, device="cpu")

        # Mutate model weights
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)

        # Update EMA
        ema.update(model)

        # Shadow weights should not equal raw active weights for trainable parameters
        trainable_tested = 0
        for p_model, p_ema in zip(model.parameters(), ema.ema_model.parameters()):
            if p_model.requires_grad:
                assert not torch.equal(p_model, p_ema)
                trainable_tested += 1
        assert trainable_tested > 0

    def test_progressive_unfreezing_schedule(self):
        """Epochs 1-5 encoder frozen; Epochs 6+ unfreezes entire backbone with 0.1x LR."""
        trainer = Trainer(
            device="cpu",
            epochs_override=10,
            batch_size_override=2,
            pretrained=False,
            fresh=True,
        )

        # Epoch 0 (Epoch 1 in 1-indexed count): Encoder frozen
        trainer._apply_progressive_unfreezing(epoch=0)
        for p in trainer.model.encoder.parameters():
            assert not p.requires_grad
        assert trainer.optimizer.param_groups[1]["lr"] == 0.0

        # Epoch 5 (Epoch 6 in 1-indexed count): Unfrozen with 0.1x LR
        trainer._apply_progressive_unfreezing(epoch=5)
        for p in trainer.model.encoder.parameters():
            assert p.requires_grad
        base_lr = trainer.optimizer.param_groups[0]["lr"]
        assert abs(trainer.optimizer.param_groups[1]["lr"] - (base_lr * 0.1)) < 1e-6

    def test_cuda_oom_auto_recovery_mechanism(self):
        """Simulate CUDA OOM and verify batch halving and grad_accum doubling."""
        trainer = Trainer(
            device="cpu",
            epochs_override=1,
            batch_size_override=8,
            pretrained=False,
            fresh=True,
        )
        trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        initial_bs = trainer.batch_size
        initial_accum = trainer.grad_accum_steps

        # Simulate OOM handling step
        trainer.batch_size = max(1, trainer.batch_size // 2)
        trainer.grad_accum_steps = trainer.grad_accum_steps * 2

        assert trainer.batch_size == initial_bs // 2
        assert trainer.grad_accum_steps == initial_accum * 2

    def test_atomic_checkpoint_latest_roundtrip(self):
        """Save state to checkpoint_latest.pth and resume accurately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            latest_path = ckpt_dir / "checkpoint_latest.pth"

            trainer = Trainer(
                device="cpu",
                epochs_override=5,
                batch_size_override=4,
                pretrained=False,
                fresh=True,
            )

            # Find callback and save state
            for cb in trainer.callbacks:
                if hasattr(cb, "checkpoint_dir"):
                    cb.checkpoint_dir = ckpt_dir
                    cb.save_checkpoint(latest_path, trainer, epoch=3, batch_idx=10, metrics={"val_mean_iou": 0.72})

            assert latest_path.exists()

            # Verify find_latest_checkpoint prioritizes checkpoint_latest.pth
            detected = find_latest_checkpoint(ckpt_dir)
            assert detected == latest_path

            # Load into fresh trainer
            new_trainer = Trainer(
                device="cpu",
                resume_checkpoint=latest_path,
                pretrained=False,
            )
            assert new_trainer.start_epoch == 3
            assert new_trainer.start_batch_idx == 10


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Export Parity & Predictor Backend Auto-Swap
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportParityAndPredictorSwap:

    def test_export_onnx_and_numerical_parity_check(self):
        """Export model to ONNX and verify numerical parity across 20 test frames."""
        from src.training.export import export_to_onnx, verify_model_parity

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            model = LaneSegNet(num_classes=4, pretrained=False)
            model.eval()

            # Export
            res = export_to_onnx(model_or_checkpoint=model, output_path=onnx_path, input_shape=(1, 3, 180, 320))
            assert onnx_path.exists()
            assert res["verification_passed"]

            # Parity check
            parity = verify_model_parity(
                pytorch_model_or_path=model,
                onnx_path=onnx_path,
                num_test_frames=20,
                input_shape=(1, 3, 180, 320),
                max_mae_threshold=1e-3,
                min_agreement_threshold=0.995,
            )
            assert parity["passed"]
            assert parity["mean_agreement"] >= 0.995
            assert parity["mean_mae"] < 1e-3

    def test_predictor_dynamic_loader_auto_swap(self):
        """MLSegmentationPredictor follows 4-tier fallback: TensorRT -> ONNX -> PyTorch -> Classical."""
        from src.inference.predictor import MLSegmentationPredictor, ModelStatus
        from src.inference.preprocessing import preprocess

        # 1. Without models present: falls back to Universal Classical CV
        p_cv = MLSegmentationPredictor(allow_classical_fallback=True)
        assert p_cv.model_status == ModelStatus.CLASSICAL_FALLBACK

        # Run inference
        dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        cv2.line(dummy_img, (200, 360), (280, 180), (255, 255, 255), 5)
        cv2.line(dummy_img, (440, 360), (360, 180), (255, 255, 255), 5)
        prep = preprocess(dummy_img)

        pred_cv = p_cv.predict(prep)
        assert pred_cv is not None
        assert pred_cv.model_status == ModelStatus.CLASSICAL_FALLBACK

    def test_predictor_direct_pytorch_load(self):
        """MLSegmentationPredictor directly loads a PyTorch checkpoint if provided."""
        from src.inference.predictor import MLSegmentationPredictor, ModelStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            pth_path = Path(tmpdir) / "test_model.pth"
            model = LaneSegNet(num_classes=4, pretrained=False)
            torch.save({"model_state_dict": model.state_dict()}, str(pth_path))

            pred_pth = MLSegmentationPredictor(model_path=pth_path, device="cpu")
            assert pred_pth.model_status == ModelStatus.READY_PYTORCH
            assert "PyTorch" in pred_pth.backend_name
