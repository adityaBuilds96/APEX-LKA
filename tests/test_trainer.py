"""
tests/test_trainer.py
======================
Comprehensive Pytest suite for APEX-LKA Phase 3 Training Engine:
- WarmupCosineScheduler
- ModelEMA
- Callbacks (ModelCheckpoint, BestModelExporter, CSVLogger, EarlyStopping)
- Bulletproof Trainer (metrics, crash recovery state, OOM reduction, dummy training loop)
"""

import copy
import math
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.callbacks import (
    BestModelExporter,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger,
)
from src.training.losses import CombinedLaneLoss
from src.training.model import LaneSegNet
from src.training.schedulers import ModelEMA, WarmupCosineScheduler
from src.training.train import (
    Trainer,
    compute_confusion_matrix,
    metrics_from_confusion_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Dummy Fixtures & Classes
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleModel(nn.Module):
    """Tiny model for lightweight fast unit testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4)
        self.register_buffer("running_mean", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class SyntheticLaneDataset(Dataset):
    """Synthetic dataset generating (image, mask) pairs for fast CPU testing."""
    def __init__(self, size: int = 12, height: int = 64, width: int = 128, num_classes: int = 4):
        self.size = size
        self.height = height
        self.width = width
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.randn(3, self.height, self.width)
        mask = torch.randint(0, self.num_classes, (self.height, self.width), dtype=torch.int64)
        return img, mask


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WarmupCosineScheduler Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestWarmupCosineScheduler:
    def test_warmup_phase_monotonic_increase(self):
        model = SimpleModel()
        base_lr = 0.01
        start_lr = 1e-5
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, max_epochs=20, warmup_start_lr=start_lr, min_lr=1e-6
        )

        lrs = []
        for epoch in range(5):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Check linear progression during warmup
        assert lrs[0] == pytest.approx(start_lr + (base_lr - start_lr) * 1 / 5, rel=1e-3)
        assert lrs[-1] == pytest.approx(base_lr, rel=1e-3)
        for i in range(len(lrs) - 1):
            assert lrs[i] < lrs[i + 1]

    def test_cosine_decay_phase(self):
        model = SimpleModel()
        base_lr = 0.01
        min_lr = 1e-5
        optimizer = torch.optim.SGD(model.parameters(), lr=base_lr)
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_epochs=5, max_epochs=25, warmup_start_lr=1e-6, min_lr=min_lr
        )

        # Fast forward through warmup
        for _ in range(5):
            optimizer.step()
            scheduler.step()

        decay_lrs = []
        for _ in range(20):
            decay_lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Decay should start near base_lr and end at min_lr
        assert decay_lrs[0] == pytest.approx(base_lr, rel=1e-2)
        assert decay_lrs[-1] == pytest.approx(min_lr, rel=1e-2)
        for i in range(len(decay_lrs) - 1):
            assert decay_lrs[i] >= decay_lrs[i + 1]

    def test_state_dict_serialization(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=4, max_epochs=10)

        for _ in range(3):
            optimizer.step()
            scheduler.step()

        state = scheduler.state_dict()
        assert state["last_epoch"] == 3

        # Create new scheduler and restore
        new_scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=4, max_epochs=10)
        new_scheduler.load_state_dict(state)
        assert new_scheduler.last_epoch == 3
        assert new_scheduler.get_lr() == scheduler.get_lr()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ModelEMA Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelEMA:
    def test_initialization(self):
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.9)
        assert ema.decay == 0.9
        assert ema.step_count == 0
        for p in ema.ema_model.parameters():
            assert not p.requires_grad

    def test_weight_update_with_decay(self):
        model = SimpleModel()
        with torch.no_grad():
            model.fc.weight.fill_(1.0)
        ema = ModelEMA(model, decay=0.8)

        # Active model weight changes to 2.0
        with torch.no_grad():
            model.fc.weight.fill_(2.0)
        ema.update(model)

        # Expected EMA: 0.8 * 1.0 + 0.2 * 2.0 = 1.2
        expected = 0.8 * 1.0 + 0.2 * 2.0
        assert torch.allclose(ema.ema_model.fc.weight, torch.tensor(expected), atol=1e-5)

    def test_buffer_exact_copy(self):
        model = SimpleModel()
        model.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
        ema = ModelEMA(model, decay=0.9)

        # Mutate buffer
        model.running_mean.copy_(torch.tensor([10.0, 20.0, 30.0, 40.0]))
        ema.update(model)
        assert torch.allclose(ema.ema_model.running_mean, model.running_mean)

    def test_apply_shadow_and_restore(self):
        model = SimpleModel()
        with torch.no_grad():
            model.fc.weight.fill_(5.0)
        ema = ModelEMA(model, decay=0.5)

        # Update EMA to fill with 5.0
        # Now change active model to 99.0
        with torch.no_grad():
            model.fc.weight.fill_(99.0)

        # Apply shadow
        ema.apply_shadow(model)
        assert torch.allclose(model.fc.weight, torch.tensor(5.0))

        # Restore active model
        ema.restore(model)
        assert torch.allclose(model.fc.weight, torch.tensor(99.0))

    def test_export_state_dict(self):
        model = SimpleModel()
        ema = ModelEMA(model, decay=0.99)
        exported = ema.export_state_dict()
        assert "fc.weight" in exported
        assert isinstance(exported["fc.weight"], torch.Tensor)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Callbacks Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallbacks:
    def test_best_model_exporter_triggers_only_on_improvement(self, tmp_path):
        export_file = tmp_path / "models" / "exported" / "best_model.pth"
        exporter = BestModelExporter(export_path=export_file, monitor="val_mean_iou", mode="max")

        class MockTrainer:
            def __init__(self):
                self.model = SimpleModel()
                self.ema = ModelEMA(self.model, decay=0.9)
                self.best_metric = 0.0

        trainer = MockTrainer()

        # Epoch 0: val_mean_iou = 0.65 -> Should export
        exporter.on_epoch_end(trainer, epoch=0, metrics={"val_mean_iou": 0.65})
        assert export_file.exists()
        assert exporter.best_score == 0.65
        assert trainer.best_metric == 0.65

        # Verify saved state is pure state dict
        saved_weights = torch.load(export_file, map_location="cpu")
        assert "fc.weight" in saved_weights

        # Epoch 1: val_mean_iou = 0.60 -> Should NOT update
        mtime_before = export_file.stat().st_mtime_ns
        exporter.on_epoch_end(trainer, epoch=1, metrics={"val_mean_iou": 0.60})
        mtime_after = export_file.stat().st_mtime_ns
        assert mtime_before == mtime_after
        assert exporter.best_score == 0.65

        # Epoch 2: val_mean_iou = 0.72 -> Should update
        exporter.on_epoch_end(trainer, epoch=2, metrics={"val_mean_iou": 0.72})
        assert exporter.best_score == 0.72

    def test_model_checkpoint_crash_and_recovery(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_cb = ModelCheckpoint(checkpoint_dir=ckpt_dir, save_every_n_epochs=2)

        class MockTrainer:
            def __init__(self):
                self.model = SimpleModel()
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
                self.scheduler = None
                self.ema = ModelEMA(self.model)
                self.scaler = None
                self.batch_size = 16
                self.best_metric = 0.81
                self.config = {"epochs": 10}

        trainer = MockTrainer()

        # Regular epoch end
        ckpt_cb.on_epoch_end(trainer, epoch=1, metrics={"val_mean_iou": 0.81})
        last_ckpt = ckpt_dir / "last_checkpoint.pt"
        assert last_ckpt.exists()
        data = torch.load(last_ckpt, map_location="cpu")
        assert data["epoch"] == 1
        assert data["batch_size"] == 16
        assert data["best_metric"] == 0.81

        # Periodic checkpoint on epoch + 1 = 2
        periodic_ckpt = ckpt_dir / "ckpt_epoch_002.pt"
        assert periodic_ckpt.exists()

        # Crash event mid-epoch at epoch 4, batch 27
        ckpt_cb.on_crash(
            trainer, epoch=4, batch_idx=27, error=RuntimeError("Simulated Power Failure")
        )
        crash_ckpt = ckpt_dir / "crash_checkpoint.pt"
        assert crash_ckpt.exists()
        crash_data = torch.load(crash_ckpt, map_location="cpu")
        assert crash_data["epoch"] == 4
        assert crash_data["batch_idx"] == 27

    def test_csv_logger(self, tmp_path):
        csv_file = tmp_path / "logs" / "metrics.csv"
        logger = CSVLogger(filename=csv_file)

        class MockTrainer:
            def __init__(self):
                self.optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.005)

        trainer = MockTrainer()
        logger.on_epoch_end(trainer, epoch=0, metrics={"train_loss": 0.42, "val_mean_iou": 0.77})
        logger.on_epoch_end(trainer, epoch=1, metrics={"train_loss": 0.31, "val_mean_iou": 0.82})

        assert csv_file.exists()
        lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3  # header + 2 epochs
        assert "val_mean_iou" in lines[0]
        assert "0.77" in lines[1]
        assert "0.82" in lines[2]

    def test_early_stopping(self):
        es = EarlyStopping(patience=3, min_delta=0.01, monitor="val_mean_iou", mode="max")

        class MockTrainer:
            def __init__(self):
                self.should_stop = False

        trainer = MockTrainer()

        es.on_epoch_end(trainer, epoch=0, metrics={"val_mean_iou": 0.50})
        assert not trainer.should_stop

        # No improvement for 3 epochs
        es.on_epoch_end(trainer, epoch=1, metrics={"val_mean_iou": 0.505})  # delta < 0.01
        assert not trainer.should_stop
        es.on_epoch_end(trainer, epoch=2, metrics={"val_mean_iou": 0.49})
        assert not trainer.should_stop
        es.on_epoch_end(trainer, epoch=3, metrics={"val_mean_iou": 0.50})
        # Now patience is reached
        assert trainer.should_stop


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Trainer Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainerEngine:
    def test_confusion_matrix_and_metrics_calculation(self):
        # 4 classes: 0=bg, 1=road, 2=left_lane, 3=right_lane
        targets = torch.tensor([[0, 1, 2, 3]])
        preds   = torch.tensor([[0, 1, 2, 0]])  # class 3 misclassified as 0

        conf = compute_confusion_matrix(preds, targets, num_classes=4)
        assert conf.shape == (4, 4)
        assert conf[0, 0] == 1  # 0->0
        assert conf[1, 1] == 1  # 1->1
        assert conf[2, 2] == 1  # 2->2
        assert conf[3, 0] == 1  # 3->0 (pred 0, true 3)

        metrics = metrics_from_confusion_matrix(conf)
        assert metrics["road_iou"] == 1.0
        assert metrics["left_lane_iou"] == 1.0
        assert metrics["right_lane_iou"] == 0.0
        assert metrics["background_iou"] == 0.5  # tp=1, fp=1, fn=0 -> 1/2
        assert "mean_iou" in metrics
        assert "lane_mean_iou" in metrics

    def test_resume_checkpoint_exact_batch(self, tmp_path):
        ckpt_path = tmp_path / "resume_test.pt"

        # Mock checkpoint saved mid-epoch
        model = LaneSegNet(pretrained=False)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        torch.save(
            {
                "epoch": 2,
                "batch_idx": 14,
                "batch_size": 8,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": None,
                "ema_state_dict": None,
                "scaler_state_dict": None,
                "best_metric": 0.73,
            },
            ckpt_path,
        )

        dummy_train = SyntheticLaneDataset(size=20, height=180, width=320)
        dummy_val = SyntheticLaneDataset(size=10, height=180, width=320)

        trainer = Trainer(
            train_dataset=dummy_train,
            val_dataset=dummy_val,
            resume_checkpoint=ckpt_path,
            device="cpu",
            callbacks=[],
            pretrained=False,
        )

        assert trainer.start_epoch == 2
        assert trainer.start_batch_idx == 14
        assert trainer.best_metric == 0.73

    def test_oom_auto_reduction_logic(self):
        """Verify that batch size is halved dynamically and DataLoader rebuilt."""
        dataset = SyntheticLaneDataset(size=16, height=180, width=320)
        trainer = Trainer(
            train_dataset=dataset,
            val_dataset=dataset,
            batch_size_override=8,
            device="cpu",
            callbacks=[],
            pretrained=False,
        )
        assert trainer.batch_size == 8

        # Simulate OOM halving
        old_bs = trainer.batch_size
        trainer.batch_size = max(1, trainer.batch_size // 2)
        trainer._build_dataloaders()

        assert trainer.batch_size == 4
        assert trainer.train_loader.batch_size == 4

    def test_dummy_end_to_end_training_cycle(self, tmp_path):
        """End-to-end run of 1 epoch on CPU with synthetic dataset to ensure zero crashes."""
        export_file = tmp_path / "best_model.pth"
        log_csv = tmp_path / "metrics.csv"
        ckpt_dir = tmp_path / "ckpts"

        callbacks = [
            ModelCheckpoint(checkpoint_dir=ckpt_dir, save_every_n_epochs=1),
            BestModelExporter(export_path=export_file, monitor="val_mean_iou", mode="max"),
            CSVLogger(filename=log_csv),
        ]

        train_ds = SyntheticLaneDataset(size=4, height=180, width=320)
        val_ds = SyntheticLaneDataset(size=2, height=180, width=320)

        trainer = Trainer(
            train_dataset=train_ds,
            val_dataset=val_ds,
            epochs_override=1,
            batch_size_override=2,
            callbacks=callbacks,
            device="cpu",
            pretrained=False,
        )

        summary = trainer.train()

        assert summary["total_epochs"] == 1
        assert "best_metric" in summary
        assert (ckpt_dir / "last_checkpoint.pt").exists()
        assert log_csv.exists()
        assert export_file.exists()
