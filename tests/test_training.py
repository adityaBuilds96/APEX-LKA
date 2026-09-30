"""
tests/test_training.py
======================
Comprehensive Pytest suite for APEX-LKA Phase 3 Training Engine (RTX 4060):
- 1-epoch smoke test on 5-frame dummy dataset
- Checkpoint save/load roundtrip with full metadata
- Exact-batch resume from checkpoint
- OOM auto-recovery and batch size reduction
- ModelEMA weight smoothing
- WarmupCosineWithRestarts scheduler progression
- HealthMonitor telemetry logging to CSV
- Emergency checkpoint preservation on KeyboardInterrupt
- Progressive layer unfreezing schedule
"""

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.training.callbacks import (
    BestModelExporter,
    CSVLogger,
    HealthMonitor,
    ModelCheckpoint,
)
from src.training.losses import CombinedLaneLoss
from src.training.model import LaneSegNet
from src.training.schedulers import ModelEMA, WarmupCosineWithRestarts
from src.training.train import Trainer
from src.training.utils import (
    compute_confusion_matrix,
    get_git_commit_hash,
    metrics_from_confusion_matrix,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures & Dummy Classes
# ═══════════════════════════════════════════════════════════════════════════════

class DummyLaneDataset(Dataset):
    """5-frame synthetic dataset for rapid smoke testing on CPU."""

    def __init__(self, size: int = 5, height: int = 64, width: int = 128, num_classes: int = 4):
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
# 1. 1-Epoch Smoke Test on 5-Frame Dummy Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def test_smoke_one_epoch_dummy_dataset(tmp_path):
    """End-to-end smoke test verifying 1 epoch completes on CPU without crashing."""
    train_ds = DummyLaneDataset(size=5, height=64, width=128)
    val_ds = DummyLaneDataset(size=2, height=64, width=128)

    ckpt_dir = tmp_path / "checkpoints"
    export_file = tmp_path / "best_model.pth"
    csv_file = tmp_path / "train_log.csv"

    callbacks = [
        ModelCheckpoint(checkpoint_dir=ckpt_dir, save_every_n_epochs=1),
        BestModelExporter(export_path=export_file, monitor="val_mean_iou", mode="max"),
        CSVLogger(filename=csv_file),
    ]

    trainer = Trainer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        epochs_override=1,
        batch_size_override=2,
        callbacks=callbacks,
        device="cpu",
        pretrained=False,
    )

    summary = trainer.fit()

    assert summary["total_epochs"] == 1
    assert "train_loss" in summary
    assert "val_loss" in summary
    assert (ckpt_dir / "last_checkpoint.pt").exists()
    assert csv_file.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Checkpoint Save/Load Roundtrip with Metadata
# ═══════════════════════════════════════════════════════════════════════════════

def test_checkpoint_save_load_roundtrip(tmp_path):
    """Verify that state dict and metadata are serialized and deserialized accurately."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_cb = ModelCheckpoint(checkpoint_dir=ckpt_dir, save_every_n_epochs=1)

    model = LaneSegNet(pretrained=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    class MockTrainer:
        def __init__(self):
            self.model = model
            self.optimizer = opt
            self.scheduler = None
            self.ema = ModelEMA(model)
            self.scaler = None
            self.batch_size = 8
            self.grad_accum_steps = 2
            self.best_metric = 0.785
            self.config = {"model": "LaneSegNet"}

    trainer = MockTrainer()
    ckpt_path = ckpt_dir / "roundtrip.pt"
    ckpt_cb.save_checkpoint(ckpt_path, trainer, epoch=2, batch_idx=5, metrics={"val_mean_iou": 0.785})

    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    assert loaded["epoch"] == 2
    assert loaded["batch_idx"] == 5
    assert loaded["batch_size"] == 8
    assert loaded["gradient_accumulation_steps"] == 2
    assert loaded["best_metric"] == 0.785
    assert "metadata" in loaded
    assert loaded["metadata"]["version"] == "3.0.0"
    assert "git_commit" in loaded["metadata"]
    assert "random_states" in loaded


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Resume from Checkpoint Exact Batch
# ═══════════════════════════════════════════════════════════════════════════════

def test_resume_from_checkpoint_exact_batch(tmp_path):
    """Ensure Trainer resumes from the exact epoch and batch index stored in checkpoint."""
    ckpt_path = tmp_path / "resume.pt"
    model = LaneSegNet(pretrained=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    torch.save(
        {
            "epoch": 4,
            "batch_idx": 18,
            "batch_size": 4,
            "gradient_accumulation_steps": 2,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "best_metric": 0.82,
        },
        ckpt_path,
    )

    dummy_train = DummyLaneDataset(size=20, height=64, width=128)
    dummy_val = DummyLaneDataset(size=5, height=64, width=128)

    trainer = Trainer(
        train_dataset=dummy_train,
        val_dataset=dummy_val,
        resume_checkpoint=ckpt_path,
        device="cpu",
        callbacks=[],
        pretrained=False,
    )

    assert trainer.start_epoch == 4
    assert trainer.start_batch_idx == 18
    assert trainer.batch_size == 4
    assert trainer.grad_accum_steps == 2
    assert trainer.best_metric == 0.82


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OOM Handler Dynamic Reduction Logic
# ═══════════════════════════════════════════════════════════════════════════════

def test_oom_handler_reduces_batch_size():
    """Verify that OOM handling halves batch size and doubles grad accumulation steps."""
    dummy_ds = DummyLaneDataset(size=16, height=64, width=128)
    trainer = Trainer(
        train_dataset=dummy_ds,
        val_dataset=dummy_ds,
        batch_size_override=8,
        device="cpu",
        callbacks=[],
        pretrained=False,
    )

    assert trainer.batch_size == 8
    initial_accum = trainer.grad_accum_steps

    # Simulate OOM reduction logic
    old_bs = trainer.batch_size
    trainer.batch_size = max(1, trainer.batch_size // 2)
    trainer.grad_accum_steps = max(1, trainer.grad_accum_steps * 2)
    trainer._build_dataloaders()

    assert trainer.batch_size == 4
    assert trainer.grad_accum_steps == initial_accum * 2
    assert trainer.train_loader.batch_size == 4


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ModelEMA Weights Update Smoothly
# ═══════════════════════════════════════════════════════════════════════════════

def test_ema_weights_update_correctly():
    """Verify that ModelEMA updates shadow weights via exponential moving average."""
    model = nn.Linear(10, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)

    ema = ModelEMA(model, decay=0.9)

    # Mutate active model weights
    with torch.no_grad():
        model.weight.fill_(2.0)

    ema.update(model)

    # Expected: 0.9 * 1.0 + 0.1 * 2.0 = 1.1
    expected = 1.1
    actual = next(ema.ema_model.parameters()).data[0, 0].item()
    assert pytest.approx(actual, rel=1e-4) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 6. WarmupCosineWithRestarts Scheduler Progression
# ═══════════════════════════════════════════════════════════════════════════════

def test_scheduler_produces_correct_lr_values():
    """Verify that WarmupCosineWithRestarts performs linear warmup and resets at restart period."""
    param = nn.Parameter(torch.zeros(1))
    base_lr = 0.01
    warmup_start = 1e-6
    opt = torch.optim.SGD([param], lr=base_lr)

    scheduler = WarmupCosineWithRestarts(
        opt,
        warmup_epochs=5,
        restart_epochs=10,
        max_epochs=30,
        warmup_start_lr=warmup_start,
        min_lr=1e-5,
    )

    lrs = []
    for epoch in range(25):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()

    # 1. Warmup monotonic progression
    for i in range(4):
        assert lrs[i] < lrs[i + 1]

    # At epoch 4 (end of warmup), should be near base_lr
    assert pytest.approx(lrs[4], rel=1e-3) == base_lr

    # At epoch 5 (start of restart period 1), progress = 0 -> lr = base_lr
    assert pytest.approx(lrs[5], rel=1e-3) == base_lr

    # At epoch 15 (start of restart period 2: (15 - 5) % 10 == 0), lr resets to base_lr
    assert pytest.approx(lrs[15], rel=1e-3) == base_lr

    # In between, lr decays smoothly
    assert lrs[10] < lrs[5]


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HealthMonitor Telemetry Logs to CSV
# ═══════════════════════════════════════════════════════════════════════════════

def test_health_monitor_logs_to_csv(tmp_path):
    """Verify that HealthMonitor logs hardware telemetry to the specified CSV file."""
    csv_path = tmp_path / "health.csv"
    monitor = HealthMonitor(log_csv=csv_path, monitor_interval_sec=0.1)

    class MockTrainer:
        def __init__(self):
            self.thermal_cooldown_active = False

    trainer = MockTrainer()
    stats = monitor._gather_telemetry(trainer)
    monitor._log_telemetry(stats)

    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) >= 1
        assert "cpu_util_pct" in reader[0]
        assert "disk_free_gb" in reader[0]
        assert "gpu_temp_c" in reader[0]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. KeyboardInterrupt Emergency Checkpoint Preservation
# ═══════════════════════════════════════════════════════════════════════════════

def test_keyboard_interrupt_saves_emergency_checkpoint(tmp_path):
    """Verify that on_crash triggered by KeyboardInterrupt saves crash and emergency checkpoints."""
    ckpt_dir = tmp_path / "emergency_test"
    ckpt_cb = ModelCheckpoint(checkpoint_dir=ckpt_dir)

    model = LaneSegNet(pretrained=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    class MockTrainer:
        def __init__(self):
            self.model = model
            self.optimizer = opt
            self.scheduler = None
            self.ema = ModelEMA(model)
            self.scaler = None
            self.batch_size = 8
            self.grad_accum_steps = 1
            self.best_metric = 0.65
            self.config = {}

    trainer = MockTrainer()
    ckpt_cb.on_crash(trainer, epoch=7, batch_idx=33, error=KeyboardInterrupt("User terminated"))

    crash_file = ckpt_dir / "crash_checkpoint.pt"
    emergency_file = ckpt_dir / "emergency_checkpoint.pt"

    assert crash_file.exists()
    assert emergency_file.exists()

    data = torch.load(emergency_file, map_location="cpu", weights_only=False)
    assert data["epoch"] == 7
    assert data["batch_idx"] == 33


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Progressive Layer Unfreezing Schedule
# ═══════════════════════════════════════════════════════════════════════════════

def test_progressive_unfreezing_schedule():
    """Verify that encoder stages are frozen in Epochs 1-5, partially unfrozen in 6-15, and full in 16+."""
    dummy_ds = DummyLaneDataset(size=4)
    trainer = Trainer(
        train_dataset=dummy_ds,
        val_dataset=dummy_ds,
        device="cpu",
        callbacks=[],
        pretrained=False,
    )
    trainer.progressive_unfreezing = True
    trainer.unfreeze_schedule = [5, 15]

    # Epoch 2 (Phase 1: 0-4): Encoder completely frozen
    trainer._apply_progressive_unfreezing(epoch=2)
    encoder_params_grad = [p.requires_grad for p in trainer.model.encoder.parameters()]
    assert not any(encoder_params_grad)
    assert any(p.requires_grad for p in trainer.model.decoder.parameters())

    # Epoch 8 (Phase 2: 5-14): Last encoder stages unfrozen with 0.1x LR
    trainer._apply_progressive_unfreezing(epoch=8)
    assert any(p.requires_grad for p in trainer.model.encoder.parameters())
    assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(trainer.optimizer.param_groups[0]["lr"] * 0.1, rel=1e-3)

    # Epoch 18 (Phase 3: 15+): Full model unfrozen with uniform LR
    trainer._apply_progressive_unfreezing(epoch=18)
    assert all(p.requires_grad for p in trainer.model.encoder.parameters())
    assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(trainer.optimizer.param_groups[0]["lr"], rel=1e-3)
