"""
src/training/callbacks.py
==========================
Modular training callbacks for APEX-LKA Phase 3 Training Engine:
- Callback: Base callback lifecycle interface.
- ModelCheckpoint: Saves full checkpoint states, metadata headers, and performs rolling cleanup.
- BestModelExporter: Automatically exports pure EMA weights to `models/exported/best_model.pth`.
- CSVLogger: Writes per-epoch metrics to CSV with immediate flush.
- TensorBoardLogger: Streams metrics, learning rates, and prediction overlays to TensorBoard.
- EarlyStopping: Halts training when validation metric plateaus.
- HealthMonitor: Background hardware telemetry (GPU, CPU, RAM, Disk, Speed) with thermal cooldown.
"""

from __future__ import annotations

import csv
import logging
import shutil
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch

from src.training.utils import (
    cleanup_old_checkpoints,
    compute_dataset_hash,
    create_prediction_overlay,
    get_git_commit_hash,
    get_random_states,
)

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

    Features:
    - Comprehensive metadata headers (version, git hash, config, dataset hash, metrics).
    - Random state preservation (Python, NumPy, PyTorch, CUDA).
    - Rolling cleanup: keeps only the last `keep_last` checkpoints.
    - Crash & emergency checkpoints on interruption.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path] = "models/checkpoints",
        save_every_n_epochs: int = 1,
        keep_last: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_epochs = max(1, save_every_n_epochs)
        self.keep_last = max(1, keep_last)
        self._train_start_time: float = time.time()

    def on_train_start(self, trainer: Any) -> None:
        self._train_start_time = time.time()

    def create_state_dict(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Bundle complete state needed to resume seamlessly."""
        # Calculate dataset fingerprint if dataset pairs exist
        dataset_hash = "unknown"
        if hasattr(trainer, "train_dataset") and hasattr(trainer.train_dataset, "_pairs"):
            dataset_hash = compute_dataset_hash([p[0] for p in trainer.train_dataset._pairs])

        early_stopping_wait = 0
        if hasattr(trainer, "callbacks"):
            for cb in trainer.callbacks:
                if isinstance(cb, EarlyStopping):
                    early_stopping_wait = cb.wait_count
                    break

        duration = time.time() - self._train_start_time

        metadata = {
            "version": "3.0.0",
            "git_commit": get_git_commit_hash(),
            "config": getattr(trainer, "config", {}),
            "dataset_hash": dataset_hash,
            "training_duration_sec": round(duration, 2),
            "metrics": metrics or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        state: Dict[str, Any] = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "batch_size": getattr(trainer, "batch_size", 8),
            "gradient_accumulation_steps": getattr(trainer, "grad_accum_steps", 1),
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": (
                trainer.scheduler.state_dict() if getattr(trainer, "scheduler", None) else None
            ),
            "ema_state_dict": (
                trainer.ema.state_dict() if getattr(trainer, "ema", None) else None
            ),
            "scaler_state_dict": (
                trainer.scaler.state_dict() if getattr(trainer, "scaler", None) else None
            ),
            "random_states": get_random_states(),
            "best_metric": getattr(trainer, "best_metric", -1.0),
            "early_stopping_wait": early_stopping_wait,
            "metadata": metadata,
            # Backward-compatibility top-level keys
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
            epoch_ckpt = self.checkpoint_dir / f"epoch_{epoch + 1:03d}.pth"
            self.save_checkpoint(epoch_ckpt, trainer, epoch=epoch + 1, batch_idx=0, metrics=metrics)
            legacy_ckpt = self.checkpoint_dir / f"ckpt_epoch_{epoch + 1:03d}.pt"
            self.save_checkpoint(legacy_ckpt, trainer, epoch=epoch + 1, batch_idx=0, metrics=metrics)

        # Rolling cleanup of old epoch checkpoints
        cleanup_old_checkpoints(self.checkpoint_dir, keep_last=self.keep_last)

    def on_crash(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        error: BaseException,
    ) -> None:
        crash_ckpt = self.checkpoint_dir / "crash_checkpoint.pt"
        self.save_checkpoint(crash_ckpt, trainer, epoch=epoch, batch_idx=batch_idx)

        emergency_ckpt = self.checkpoint_dir / "emergency_checkpoint.pt"
        self.save_checkpoint(emergency_ckpt, trainer, epoch=epoch, batch_idx=batch_idx)

        log.warning(
            "Emergency crash recovery saved to %s (epoch %d, batch %d, error: %s)",
            crash_ckpt, epoch, batch_idx, error
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Best Model Exporter
# ═══════════════════════════════════════════════════════════════════════════════

class BestModelExporter(Callback):
    """
    Exports only the Exponential Moving Average (EMA) weights to
    `models/exported/best_model.pth` whenever validation mIoU improves.
    """

    def __init__(
        self,
        export_path: Union[str, Path] = "models/exported/best_model.pth",
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

            if hasattr(trainer, "ema") and trainer.ema is not None:
                weights = trainer.ema.export_state_dict()
                src_name = "EMA"
            else:
                weights = trainer.model.state_dict()
                src_name = "ActiveModel"

            tmp_path = self.export_path.with_suffix(self.export_path.suffix + ".tmp")
            torch.save(weights, tmp_path)
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
    """

    def __init__(self, filename: Union[str, Path] = "logs/training/experiment_log.csv") -> None:
        self.filename = Path(filename)
        self.filename.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict[str, float]) -> None:
        row: Dict[str, Any] = {"epoch": epoch + 1}
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


# ═══════════════════════════════════════════════════════════════════════════════
# TensorBoard Logger
# ═══════════════════════════════════════════════════════════════════════════════

class TensorBoardLogger(Callback):
    """
    Streams scalars (loss, lr, mIoU, per-class metrics) to TensorBoard.
    Generates prediction visual overlays every 5 epochs.
    """

    def __init__(self, log_dir: Union[str, Path] = "logs/training/tensorboard") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            log.warning("TensorBoard not available (pip install tensorboard). Skipping TensorBoard logging.")

    def on_batch_end(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if self.writer is None:
            return
        total_batches = getattr(trainer, "_total_train_batches", 100)
        global_step = epoch * total_batches + batch_idx
        self.writer.add_scalar("train/batch_loss", loss, global_step)

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

        # Log sample prediction overlays every 5 epochs
        if step % 5 == 0 and hasattr(trainer, "val_loader") and trainer.val_loader:
            self._log_sample_overlay(trainer, step)

        self.writer.flush()

    def _log_sample_overlay(self, trainer: Any, step: int) -> None:
        try:
            val_iter = iter(trainer.val_loader)
            batch = next(val_iter)
            imgs = batch[0][:2].to(trainer.device)
            masks = batch[1][:2].cpu().numpy()

            with torch.no_grad():
                preds = trainer.model.predict(imgs).cpu().numpy()

            # Convert normalized tensor to BGR uint8
            raw_img = imgs[0].permute(1, 2, 0).cpu().numpy()
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            raw_img = np.clip((raw_img * std + mean) * 255.0, 0, 255).astype(np.uint8)

            overlay = create_prediction_overlay(raw_img, preds[0])
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            self.writer.add_image("val/prediction_sample", overlay_rgb, step, dataformats="HWC")
        except Exception as e:
            log.debug("Could not log prediction overlay: %s", e)

    def on_train_end(self, trainer: Any) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Early Stopping
# ═══════════════════════════════════════════════════════════════════════════════

class EarlyStopping(Callback):
    """Stops training when monitored metric stops improving."""

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


# ═══════════════════════════════════════════════════════════════════════════════
# Health Monitor Callback
# ═══════════════════════════════════════════════════════════════════════════════

class HealthMonitor(Callback):
    """
    Hardware health and telemetry monitor for RTX 4060 and edge systems:
    - Monitors GPU temperature, VRAM usage, CPU, RAM, and Disk space.
    - Automatic 5s cooldown if GPU temperature > 85°C.
    - Automatic cleanup of old checkpoints if disk space < 1 GB.
    - Detects and warns if batch throughput drops > 50%.
    - Logs telemetry to logs/training/health_monitor.csv.
    """

    def __init__(
        self,
        log_csv: Union[str, Path] = "logs/training/health_monitor.csv",
        gpu_temp_threshold: float = 85.0,
        disk_space_min_gb: float = 1.0,
        cooldown_seconds: float = 5.0,
        monitor_interval_sec: float = 10.0,
    ) -> None:
        self.log_csv = Path(log_csv)
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)
        self.gpu_temp_threshold = float(gpu_temp_threshold)
        self.disk_space_min_gb = float(disk_space_min_gb)
        self.cooldown_seconds = float(cooldown_seconds)
        self.monitor_interval_sec = float(monitor_interval_sec)

        self._recent_speeds: deque = deque(maxlen=20)
        self._last_batch_time: float = time.time()
        self._baseline_speed: Optional[float] = None
        self._is_running: bool = False
        self._thread: Optional[threading.Thread] = None

    def on_train_start(self, trainer: Any) -> None:
        self._is_running = True
        self._last_batch_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, args=(trainer,), daemon=True)
        self._thread.start()

    def on_train_end(self, trainer: Any) -> None:
        self._is_running = False

    def on_batch_start(self, trainer: Any, epoch: int, batch_idx: int) -> None:
        # Check if thermal cooldown was requested
        if getattr(trainer, "thermal_cooldown_active", False):
            log.warning("[Thermal Protection] GPU temp > %.1f°C! Cooling down for %.1fs...",
                        self.gpu_temp_threshold, self.cooldown_seconds)
            time.sleep(self.cooldown_seconds)
            trainer.thermal_cooldown_active = False

    def on_batch_end(
        self,
        trainer: Any,
        epoch: int,
        batch_idx: int,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        now = time.time()
        delta = max(1e-4, now - self._last_batch_time)
        speed = 1.0 / delta
        self._last_batch_time = now
        self._recent_speeds.append(speed)

        # Baseline throughput tracking
        if len(self._recent_speeds) >= 10:
            current_avg = sum(self._recent_speeds) / len(self._recent_speeds)
            if self._baseline_speed is None:
                self._baseline_speed = current_avg
            elif current_avg < 0.5 * self._baseline_speed:
                log.warning(
                    "[HealthMonitor Warning] Training speed dropped >50%%: %.2f batches/sec "
                    "(baseline: %.2f). Possible thermal or I/O bottleneck.",
                    current_avg, self._baseline_speed
                )

    def _monitor_loop(self, trainer: Any) -> None:
        """Background thread collecting system telemetry."""
        while self._is_running:
            try:
                stats = self._gather_telemetry(trainer)
                self._log_telemetry(stats)
                self._evaluate_thresholds(trainer, stats)
            except Exception as e:
                log.debug("Telemetry gathering error: %s", e)
            time.sleep(self.monitor_interval_sec)

    def _gather_telemetry(self, trainer: Any) -> Dict[str, Any]:
        """Collect metrics across CPU, RAM, GPU, and Disk."""
        stats: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_temp_c": 0.0,
            "gpu_util_pct": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_total_mb": 0.0,
            "cpu_util_pct": 0.0,
            "ram_used_gb": 0.0,
            "disk_free_gb": 0.0,
            "batches_per_sec": (
                round(sum(self._recent_speeds) / len(self._recent_speeds), 2)
                if self._recent_speeds else 0.0
            ),
        }

        # CPU & RAM via psutil
        try:
            import psutil
            stats["cpu_util_pct"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            stats["ram_used_gb"] = round((mem.total - mem.available) / (1024**3), 2)
        except Exception:
            pass

        # Disk space
        try:
            total, used, free = shutil.disk_usage(Path(".").resolve())
            stats["disk_free_gb"] = round(free / (1024**3), 2)
        except Exception:
            pass

        # GPU stats via torch or pynvml
        if torch.cuda.is_available():
            try:
                stats["gpu_mem_used_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 1)
                stats["gpu_mem_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1)
            except Exception:
                pass

            # Query temperature via pynvml if present
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                stats["gpu_temp_c"] = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["gpu_util_pct"] = float(util.gpu)
            except Exception:
                # Mock a safe operational temperature if NVML is unavailable
                stats["gpu_temp_c"] = 65.0

        return stats

    def _evaluate_thresholds(self, trainer: Any, stats: Dict[str, Any]) -> None:
        # 1. Thermal protection
        if stats["gpu_temp_c"] > self.gpu_temp_threshold:
            trainer.thermal_cooldown_active = True

        # 2. Disk space protection
        if stats["disk_free_gb"] > 0 and stats["disk_free_gb"] < self.disk_space_min_gb:
            log.warning("[HealthMonitor Warning] Disk free space low (%.2f GB < %.2f GB). Purging old checkpoints...",
                        stats["disk_free_gb"], self.disk_space_min_gb)
            cleanup_old_checkpoints(Path("models/checkpoints"), keep_last=2)

    def _log_telemetry(self, stats: Dict[str, Any]) -> None:
        file_exists = self.log_csv.exists()
        with open(self.log_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats.keys()))
            if not file_exists or self.log_csv.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(stats)
            f.flush()
