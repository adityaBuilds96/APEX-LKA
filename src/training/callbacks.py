"""
src/training/callbacks.py
==========================
Modular training callbacks for APEX-LKA:
- Callback: Base callback class with lifecycle hooks.
- ModelCheckpoint: Saves periodic, latest, and emergency crash checkpoints.
- BestModelExporter: Automatically exports pure EMA weights to
  `models/exported/best_model.pth` when validation mIoU improves.
- CSVLogger: Writes per-epoch training and validation metrics to CSV with immediate flush.
- TensorBoardLogger: Streams learning rate, losses, and per-class mIoU to TensorBoard.
- EarlyStopping: Halts training if validation metric plateaus for `patience` epochs.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Base Callback
# ═══════════════════════════════════════════════════════════════════════════════

class Callback:
    """Base class for all training callbacks."""

    def on_train_start(self, trainer: Any) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        pass

    def on_batch_start(self, trainer: Any, epoch: int, batch_idx: int) -> None:
        pass

    def on_batch_end(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        pass

    def on_crash(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        error: BaseException,
    ) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Model Checkpoint
# ═══════════════════════════════════════════════════════════════════════════════

class ModelCheckpoint(Callback):
    """
    Saves full training state for fault tolerance and crash recovery.

    Saves:
    - `last_checkpoint.pt`: updated at the end of every epoch.
    - `ckpt_epoch_{epoch}.pt`: saved every `save_every_n_epochs`.
    - `crash_checkpoint.pt`: saved immediately if a training crash occurs.

    Parameters
    ----------
    checkpoint_dir : str | Path
        Directory to store checkpoints.
    save_every_n_epochs : int, default 5
        Frequency of epoch-tagged checkpoints.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_every_n_epochs: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = max(1, save_every_n_epochs)

    def create_state_dict(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Bundle complete state needed to resume seamlessly."""
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "batch_size": getattr(trainer, "batch_size", 8),
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            "ema_state_dict": (
                trainer.ema.state_dict() if getattr(trainer, "ema", None) else None
            ),
            "scaler_state_dict": (
                trainer.scaler.state_dict() if getattr(trainer, "scaler", None) else None
            ),
            "best_metric": getattr(trainer, "best_metric", 0.0),
            "metrics": metrics or {},
            "config": getattr(trainer, "config", {}),
        }
        return state

    def save_checkpoint(
        self,
        path: Path,
        trainer: Any,
        epoch: int,
        batch_idx: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Atomic write to checkpoint path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.create_state_dict(trainer, epoch, batch_idx, metrics)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)
        log.info("Saved checkpoint -> %s (epoch %d, batch %d)", path.name, epoch, batch_idx)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        last_ckpt = self.checkpoint_dir / "last_checkpoint.pt"
        self.save_checkpoint(last_ckpt, trainer, epoch=epoch, batch_idx=0, metrics=metrics)

        if (epoch + 1) % self.save_every_n_epochs == 0:
            periodic_ckpt = self.checkpoint_dir / f"ckpt_epoch_{epoch + 1:03d}.pt"
            self.save_checkpoint(periodic_ckpt, trainer, epoch=epoch + 1, batch_idx=0, metrics=metrics)

    def on_crash(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        error: BaseException,
    ) -> None:
        crash_ckpt = self.checkpoint_dir / "crash_checkpoint.pt"
        self.save_checkpoint(crash_ckpt, trainer, epoch=epoch, batch_idx=batch_idx)
        log.warning(
            "Crash recovery saved to %s at epoch %d, batch %d (Reason: %s)",
            crash_ckpt, epoch, batch_idx, error
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Best Model Exporter
# ═══════════════════════════════════════════════════════════════════════════════

class BestModelExporter(Callback):
    """
    Exports only the Exponential Moving Average (EMA) weights to
    `models/exported/best_model.pth` whenever validation mIoU improves.

    Parameters
    ----------
    export_path : str | Path
        Target destination for inference-ready weights.
    monitor : str, default 'val_mean_iou'
        Metric name to maximize.
    mode : str, default 'max'
        Direction of optimization ('max' or 'min').
    """

    def __init__(
        self,
        export_path: str | Path = "models/exported/best_model.pth",
        monitor: str = "val_mean_iou",
        mode: str = "max",
    ) -> None:
        self.export_path = Path(export_path)
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode.lower()
        self.best_score: float = -float("inf") if self.mode == "max" else float("inf")
        self.best_epoch: int = -1

    def is_better(self, score: float) -> bool:
        if self.mode == "max":
            return score > self.best_score
        return score < self.best_score

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self.monitor not in metrics:
            return

        current_score = metrics[self.monitor]
        if self.is_better(current_score):
            prev = self.best_score
            self.best_score = current_score
            self.best_epoch = epoch
            trainer.best_metric = current_score

            # Extract pure weights: prefer EMA if available, else active model
            if hasattr(trainer, "ema") and trainer.ema is not None:
                weights_to_export = trainer.ema.export_state_dict()
                src_name = "EMA"
            else:
                weights_to_export = trainer.model.state_dict()
                src_name = "ActiveModel"

            # Atomic save
            tmp_path = self.export_path.with_suffix(self.export_path.suffix + ".tmp")
            torch.save(weights_to_export, tmp_path)
            tmp_path.replace(self.export_path)

            log.info(
                "BestModelExporter: [%s] Improved %s: %.4f -> %.4f (Epoch %d). Exported to %s",
                src_name, self.monitor, prev, current_score, epoch + 1, self.export_path
            )


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Logger
# ═══════════════════════════════════════════════════════════════════════════════

class CSVLogger(Callback):
    """
    Appends training and validation metrics to a CSV file after every epoch,
    flushing immediately to guarantee data persistence.

    Parameters
    ----------
    filename : str | Path
        Path to output CSV file.
    """

    def __init__(self, filename: str | Path = "logs/training/training_metrics.csv") -> None:
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None
        self._fieldnames: Optional[List[str]] = None

    def on_train_start(self, trainer: Any) -> None:
        # We write/append when first epoch completes
        pass

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        row = {"epoch": epoch + 1}
        # Include current learning rate if available
        if hasattr(trainer, "optimizer") and trainer.optimizer.param_groups:
            row["lr"] = trainer.optimizer.param_groups[0]["lr"]
        row.update(metrics)

        file_exists = self.filename.exists()
        with open(self.filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists or self.filename.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

    def on_train_end(self, trainer: Any) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# TensorBoard Logger
# ═══════════════════════════════════════════════════════════════════════════════

class TensorBoardLogger(Callback):
    """
    Streams scalars (loss, lr, mIoU, per-class metrics) to TensorBoard.

    Parameters
    ----------
    log_dir : str | Path
        Directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str | Path = "logs/training/tensorboard") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            log.warning("TensorBoard not available (pip install tensorboard). Skipping TensorBoard logging.")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self.writer is None:
            return

        step = epoch + 1
        if hasattr(trainer, "optimizer") and trainer.optimizer.param_groups:
            self.writer.add_scalar("train/learning_rate", trainer.optimizer.param_groups[0]["lr"], step)

        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                if key.startswith("train_"):
                    tag = f"train/{key[6:]}"
                elif key.startswith("val_"):
                    tag = f"val/{key[4:]}"
                else:
                    tag = f"metrics/{key}"
                self.writer.add_scalar(tag, val, step)

        self.writer.flush()

    def on_train_end(self, trainer: Any) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping(Callback):
    """
    Stops training when monitored metric stops improving.

    Parameters
    ----------
    patience : int, default 10
        Number of epochs with no improvement after which training stops.
    min_delta : float, default 1e-4
        Minimum change in monitored metric to qualify as an improvement.
    monitor : str, default 'val_mean_iou'
        Metric to track.
    mode : str, default 'max'
        'max' for metrics to maximize, 'min' for losses to minimize.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        monitor: str = "val_mean_iou",
        mode: str = "max",
    ) -> None:
        self.patience = max(1, patience)
        self.min_delta = float(min_delta)
        self.monitor = monitor
        self.mode = mode.lower()
        self.wait_count: int = 0
        self.best_score: float = -float("inf") if self.mode == "max" else float("inf")

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]
        improved = (
            (current - self.best_score) > self.min_delta
            if self.mode == "max"
            else (self.best_score - current) > self.min_delta
        )

        if improved:
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                log.info(
                    "EarlyStopping triggered: no improvement in %s for %d consecutive epochs.",
                    self.monitor, self.patience
                )
                trainer.should_stop = True
