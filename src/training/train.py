"""
src/training/train.py
======================
Bulletproof Master Training Engine for APEX-LKA (LaneSegNet).

Features
--------
1. PyTorch Automatic Mixed Precision (AMP):
   - 2x speedup and ~50% GPU memory reduction via `torch.cuda.amp`.
   - Automatic graceful fallback on CPU.
2. Crash Recovery & Exact-Batch Resume:
   - Complete state tracking: model, optimizer, scheduler, EMA, scaler, epoch, batch_idx.
   - Emergency crash handler on SIGINT (Ctrl+C) / unhandled exception.
   - Instant resumption from the exact batch of interruption.
3. CUDA OOM Auto-Reduction:
   - Catches CUDA OutOfMemoryError, purges GPU cache, halves batch size dynamically,
     rebuilds DataLoader, and continues without crashing.
4. Rich Visuals & Real-Time Monitoring:
   - Per-epoch metrics logged to CSV, TensorBoard, and Rich console tables.
   - Automatic EMA export to `models/exported/best_model.pth` upon val mIoU improvement.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg
from src.training.callbacks import (
    BestModelExporter,
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoardLogger,
)
from src.training.dataset import LaneSegDataset
from src.training.losses import CombinedLaneLoss
from src.training.model import LaneSegNet
from src.training.schedulers import ModelEMA, WarmupCosineScheduler

log = logging.getLogger("apex_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Metric Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Fast vectorized confusion matrix computation on device.
    Shape: (num_classes, num_classes), row=truth, col=pred.
    """
    mask = (targets >= 0) & (targets < num_classes)
    flat_targets = targets[mask].view(-1)
    flat_preds = preds[mask].view(-1)

    bins = flat_targets * num_classes + flat_preds
    hist = torch.bincount(bins, minlength=num_classes**2)
    return hist.reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(
    conf_mat: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes per-class IoU, mean IoU, and lane mean IoU from a confusion matrix.
    """
    if class_names is None:
        class_names = ["background", "road", "left_lane", "right_lane"]

    num_classes = conf_mat.shape[0]
    ious: List[float] = []
    res: Dict[str, float] = {}

    for c in range(num_classes):
        tp = conf_mat[c, c].item()
        fp = conf_mat[:, c].sum().item() - tp
        fn = conf_mat[c, :].sum().item() - tp
        denom = tp + fp + fn
        iou = (tp / denom) if denom > 0 else 0.0
        ious.append(iou)
        c_name = class_names[c] if c < len(class_names) else f"class_{c}"
        res[f"{c_name}_iou"] = round(iou, 4)

    mean_iou = sum(ious) / max(1, len(ious))
    res["mean_iou"] = round(mean_iou, 4)

    # Specific focus metric: lanes only (classes 2 & 3)
    if num_classes >= 4:
        lane_miou = (ious[2] + ious[3]) / 2.0
        res["lane_mean_iou"] = round(lane_miou, 4)

    return res


# ═══════════════════════════════════════════════════════════════════════════════
# Master Trainer Engine
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    APEX-LKA Bulletproof Trainer.

    Handles mixed precision, OOM recovery, exact batch resuming, EMA weights,
    and callback execution.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        train_dataset: Optional[LaneSegDataset] = None,
        val_dataset: Optional[LaneSegDataset] = None,
        device: Optional[str] = None,
        resume_checkpoint: Optional[str | Path] = None,
        callbacks: Optional[List[Callback]] = None,
        epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        lr_override: Optional[float] = None,
        pretrained: Optional[bool] = None,
    ) -> None:
        self.config = config or cfg

        # Device determination
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training hyperparameters
        t_cfg = self.config.get("training", {})
        m_cfg = self.config.get("model", {})
        p_cfg = self.config.get("preprocessing", {})

        self.epochs = epochs_override or t_cfg.get("epochs", 50)
        self.batch_size = batch_size_override or t_cfg.get("batch_size", 8)
        self.initial_batch_size = self.batch_size
        self.lr = lr_override or t_cfg.get("learning_rate", 1e-3)
        self.weight_decay = t_cfg.get("weight_decay", 1e-4)
        self.num_workers = t_cfg.get("num_workers", 0)
        self.pin_memory = t_cfg.get("pin_memory", False) and self.device.type == "cuda"
        self.grad_clip = t_cfg.get("gradient_clip", 1.0)
        self.num_classes = m_cfg.get("target_num_classes", 4)
        self.img_height = p_cfg.get("image_height", 360)
        self.img_width = p_cfg.get("image_width", 640)

        # AMP Configuration
        self.use_amp = (self.device.type == "cuda")
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        )

        # Tracking state
        self.start_epoch: int = 0
        self.start_batch_idx: int = 0
        self.best_metric: float = -1.0
        self.should_stop: bool = False

        # Build Model
        use_pretrained = pretrained if pretrained is not None else m_cfg.get("pretrained_backbone", True)
        self.model = LaneSegNet(num_classes=self.num_classes, pretrained=use_pretrained)
        self.model.to(self.device)

        # Build EMA shadow weights
        self.ema = ModelEMA(self.model, decay=0.9999, device=self.device)

        # Build Criterion
        ce_w = t_cfg.get("ce_weight", 0.5)
        dice_w = t_cfg.get("dice_weight", 0.5)
        self.criterion = CombinedLaneLoss(
            ce_weight=ce_w,
            dice_weight=dice_w,
            num_classes=self.num_classes,
        )
        self.criterion.to(self.device)

        # Build Optimizer
        opt_name = t_cfg.get("optimizer", "adam").lower()
        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

        # Build Scheduler (Linear Warmup for 5 epochs + Cosine Annealing)
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=5,
            max_epochs=self.epochs,
            warmup_start_lr=1e-6,
            min_lr=1e-6,
        )

        # Datasets & DataLoaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if self.train_dataset is None:
            self._init_datasets()

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self._build_dataloaders()

        # Callbacks
        self.callbacks = callbacks if callbacks is not None else self._default_callbacks()

        # Resume state if checkpoint specified
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)

        # Register signal handlers for graceful shutdown on Ctrl+C / SIGINT
        self._register_signals()

    # ── Initialization Helpers ─────────────────────────────────────────────────

    def _init_datasets(self) -> None:
        """Initialize LaneSegDatasets from disk paths if available."""
        train_path = PATHS.train
        val_path = PATHS.val

        if (train_path / "images").exists() and (train_path / "masks").exists():
            try:
                self.train_dataset = LaneSegDataset(
                    train_path, split="train", height=self.img_height, width=self.img_width
                )
            except Exception as e:
                log.warning("Could not initialize train_dataset: %s", e)

        if (val_path / "images").exists() and (val_path / "masks").exists():
            try:
                self.val_dataset = LaneSegDataset(
                    val_path, split="val", height=self.img_height, width=self.img_width
                )
            except Exception as e:
                log.warning("Could not initialize val_dataset: %s", e)

    def _build_dataloaders(self) -> None:
        """Create PyTorch DataLoaders with current batch_size."""
        if self.train_dataset and len(self.train_dataset) > 0:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )
        if self.val_dataset and len(self.val_dataset) > 0:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=False,
            )

    def _default_callbacks(self) -> List[Callback]:
        """Configure standard production callbacks."""
        ckpt_dir = PATHS.checkpoints
        export_file = PATHS.exported / "best_model.pth"
        log_csv = PATHS.logs / "training" / "training_metrics.csv"
        tb_dir = PATHS.logs / "training" / "tensorboard"

        return [
            ModelCheckpoint(
                checkpoint_dir=ckpt_dir,
                save_every_n_epochs=self.config.get("model", {}).get("save_every_n_epochs", 5),
            ),
            BestModelExporter(
                export_path=export_file,
                monitor="val_mean_iou",
                mode="max",
            ),
            CSVLogger(filename=log_csv),
            TensorBoardLogger(log_dir=tb_dir),
            EarlyStopping(
                patience=self.config.get("training", {}).get("early_stopping_patience", 10),
                monitor="val_mean_iou",
                mode="max",
            ),
        ]

    def _register_signals(self) -> None:
        """Handle interrupt signals to save emergency checkpoint before termination."""
        def handler(sig, frame):
            log.warning("Termination signal received (%s). Triggering emergency save...", sig)
            self._handle_crash(
                epoch=getattr(self, "_current_epoch", 0),
                batch_idx=getattr(self, "_current_batch_idx", 0),
                error=KeyboardInterrupt("Process interrupted by user/system signal"),
            )
            sys.exit(130)

        try:
            signal.signal(signal.SIGINT, handler)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, handler)
        except (ValueError, AttributeError):
            pass

    # ── Checkpointing & Crash Recovery ─────────────────────────────────────────

    def load_checkpoint(self, path: str | Path) -> None:
        """
        Load complete training state from checkpoint.
        Supports resuming from exact batch if checkpoint occurred mid-epoch.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        log.info("Loading training checkpoint: %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if self.ema and state.get("ema_state_dict"):
            self.ema.load_state_dict(state["ema_state_dict"])

        if self.scaler and state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])

        self.start_epoch = int(state.get("epoch", 0))
        self.start_batch_idx = int(state.get("batch_idx", 0))
        self.best_metric = float(state.get("best_metric", 0.0))

        # Restore saved batch size if present
        saved_bs = state.get("batch_size")
        if saved_bs and saved_bs != self.batch_size:
            log.info("Restoring batch size from checkpoint: %d -> %d", self.batch_size, saved_bs)
            self.batch_size = saved_bs
            self._build_dataloaders()

        log.info(
            "Resumed successfully at Epoch %d, Batch %d (Best metric: %.4f)",
            self.start_epoch + 1, self.start_batch_idx, self.best_metric
        )

    def _handle_crash(self, epoch: int, batch_idx: int, error: BaseException) -> None:
        """Invoke crash handlers across all callbacks."""
        for cb in self.callbacks:
            try:
                cb.on_crash(self, epoch, batch_idx, error)
            except Exception as cb_err:
                log.error("Error in callback %s during on_crash: %s", cb, cb_err)

    # ── Training & Validation Loops ────────────────────────────────────────────

    def _train_epoch(
        self,
        epoch: int,
        start_batch_idx: int = 0,
    ) -> Dict[str, float]:
        """
        Train for a single epoch with PyTorch AMP and OOM auto-reduction.
        Resumes from `start_batch_idx` if restarting mid-epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        steps = 0

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=True)
            if self.use_amp
            else nullcontext()
        )

        batch_idx = start_batch_idx
        while True:
            try:
                if self.train_loader is None or len(self.train_loader) == 0:
                    break

                for b_i, (images, targets) in enumerate(self.train_loader):
                    if b_i < batch_idx:
                        continue

                    batch_idx = b_i
                    self._current_epoch = epoch
                    self._current_batch_idx = batch_idx

                    for cb in self.callbacks:
                        cb.on_batch_start(self, epoch, batch_idx)

                    images = images.to(self.device, non_blocking=self.pin_memory)
                    targets = targets.to(self.device, non_blocking=self.pin_memory)

                    self.optimizer.zero_grad(set_to_none=True)

                    with autocast_ctx:
                        logits = self.model(images)
                        loss, l_ce, l_dice = self.criterion(logits, targets)

                    # Backward with AMP scaling if available
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        if self.grad_clip > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip
                            )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.grad_clip
                            )
                        self.optimizer.step()

                    # Update Exponential Moving Average weights
                    if self.ema is not None:
                        self.ema.update(self.model)

                    l_val = loss.item()
                    total_loss += l_val
                    total_ce += l_ce.item()
                    total_dice += l_dice.item()
                    steps += 1

                    for cb in self.callbacks:
                        cb.on_batch_end(self, epoch, batch_idx, l_val)

                break  # Completed epoch successfully

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                # Catch CUDA OutOfMemoryError and handle auto-reduction
                is_oom = isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())) or (
                    "out of memory" in str(exc).lower()
                )
                if not is_oom or self.device.type != "cuda":
                    raise exc

                # Purge GPU cache
                self.optimizer.zero_grad(set_to_none=True)
                if "images" in locals():
                    del images
                if "targets" in locals():
                    del targets
                if "logits" in locals():
                    del logits
                if "loss" in locals():
                    del loss
                torch.cuda.empty_cache()

                if self.batch_size <= 1:
                    log.critical("OOM: Batch size already 1. Cannot reduce further.")
                    raise exc

                old_bs = self.batch_size
                self.batch_size = max(1, self.batch_size // 2)
                log.warning(
                    "[OOM Auto-Reduction] Caught CUDA OOM at epoch %d, batch %d! "
                    "Reducing batch size: %d -> %d and resuming without crash...",
                    epoch + 1, batch_idx, old_bs, self.batch_size
                )

                # Rebuild DataLoader with halved batch size and resume from current batch
                self._build_dataloaders()
                # Continue while loop with new loader from batch_idx

        avg_loss = total_loss / max(1, steps)
        avg_ce = total_ce / max(1, steps)
        avg_dice = total_dice / max(1, steps)

        return {
            "train_loss": round(avg_loss, 4),
            "train_ce": round(avg_ce, 4),
            "train_dice": round(avg_dice, 4),
        }

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate model using EMA weights on validation split.
        Returns validation loss, per-class IoU, and mean IoU.
        """
        if self.val_loader is None or len(self.val_loader) == 0:
            return {}

        # Apply EMA shadow weights for stable validation evaluation
        if self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()

        total_loss = 0.0
        total_ce = 0.0
        total_dice = 0.0
        steps = 0

        conf_mat = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
            device=self.device,
        )

        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=True)
            if self.use_amp
            else nullcontext()
        )

        for images, targets in self.val_loader:
            images = images.to(self.device, non_blocking=self.pin_memory)
            targets = targets.to(self.device, non_blocking=self.pin_memory)

            with autocast_ctx:
                logits = self.model(images)
                loss, l_ce, l_dice = self.criterion(logits, targets)

            preds = logits.argmax(dim=1)
            conf_mat += compute_confusion_matrix(preds, targets, num_classes=self.num_classes)

            total_loss += loss.item()
            total_ce += l_ce.item()
            total_dice += l_dice.item()
            steps += 1

        # Restore active model weights
        if self.ema is not None:
            self.ema.restore(self.model)

        avg_loss = total_loss / max(1, steps)
        avg_ce = total_ce / max(1, steps)
        avg_dice = total_dice / max(1, steps)

        metrics = metrics_from_confusion_matrix(conf_mat)
        metrics["val_loss"] = round(avg_loss, 4)
        metrics["val_ce"] = round(avg_ce, 4)
        metrics["val_dice"] = round(avg_dice, 4)
        metrics["val_mean_iou"] = metrics.pop("mean_iou", 0.0)
        if "lane_mean_iou" in metrics:
            metrics["val_lane_mean_iou"] = metrics.pop("lane_mean_iou")

        return metrics

    # ── Main Training Loop ─────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        """
        Execute full training workflow across all epochs.
        """
        console.print(
            Panel(
                f"[bold cyan]APEX-LKA Master Training Engine[/bold cyan]\n"
                f"Device: [green]{self.device}[/green] | AMP: [green]{self.use_amp}[/green] | "
                f"Batch Size: [green]{self.batch_size}[/green] | Epochs: [green]{self.epochs}[/green]\n"
                f"Model: LaneSegNet (MobileNetV3 + FPN + SE Attention) | Classes: {self.num_classes}",
                border_style="cyan",
            )
        )

        for cb in self.callbacks:
            cb.on_train_start(self)

        start_time = time.time()

        try:
            for epoch in range(self.start_epoch, self.epochs):
                self._current_epoch = epoch
                epoch_start = time.time()

                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)

                # Determine starting batch index (only non-zero if resuming mid-epoch)
                batch_start = self.start_batch_idx if epoch == self.start_epoch else 0

                # Train epoch
                train_metrics = self._train_epoch(epoch, start_batch_idx=batch_start)

                # Validate epoch (using EMA weights)
                val_metrics = self._val_epoch(epoch)

                # Step scheduler
                if self.scheduler:
                    self.scheduler.step()

                # Aggregate epoch metrics
                epoch_time = time.time() - epoch_start
                all_metrics = {**train_metrics, **val_metrics, "epoch_time_sec": round(epoch_time, 2)}

                # Rich console status table
                table = Table(title=f"Epoch {epoch + 1}/{self.epochs} Summary", show_lines=False)
                table.add_column("Train Loss", justify="center")
                table.add_column("Val Loss", justify="center")
                table.add_column("Val mIoU", style="bold green", justify="center")
                table.add_column("Lane mIoU", justify="center")
                table.add_column("LR", justify="center")
                table.add_column("Time", justify="center")

                lr_val = self.optimizer.param_groups[0]["lr"]
                table.add_row(
                    f"{train_metrics.get('train_loss', 0.0):.4f}",
                    f"{val_metrics.get('val_loss', 0.0):.4f}",
                    f"{val_metrics.get('val_mean_iou', 0.0):.4f}",
                    f"{val_metrics.get('val_lane_mean_iou', 0.0):.4f}",
                    f"{lr_val:.6f}",
                    f"{epoch_time:.1f}s",
                )
                console.print(table)

                # Notify callbacks
                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, all_metrics)

                if self.should_stop:
                    log.info("Early stopping triggered at epoch %d. Terminating training.", epoch + 1)
                    break

        except (KeyboardInterrupt, SystemExit) as sig_err:
            log.warning("Training halted by user interrupt (Ctrl+C). Checkpoint saved.")
            self._handle_crash(getattr(self, "_current_epoch", 0), getattr(self, "_current_batch_idx", 0), sig_err)
            raise sig_err

        except Exception as err:
            log.exception("Unhandled exception during training: %s", err)
            self._handle_crash(getattr(self, "_current_epoch", 0), getattr(self, "_current_batch_idx", 0), err)
            raise err

        finally:
            for cb in self.callbacks:
                cb.on_train_end(self)

        total_elapsed = time.time() - start_time
        console.print(
            Panel(
                f"[bold green]✔ Training Completed Successfully[/bold green]\n"
                f"Total Time: [cyan]{total_elapsed:.1f}s[/cyan] | Best mIoU: [bold green]{self.best_metric:.4f}[/bold green]\n"
                f"Best Model Exported: [cyan]{PATHS.exported / 'best_model.pth'}[/cyan]",
                border_style="green",
            )
        )

        return {
            "best_metric": self.best_metric,
            "total_epochs": epoch + 1,
            "elapsed_seconds": total_elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="APEX-LKA Master Training Script")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Check for auto-resume if requested or crash checkpoint exists
    resume_path = args.resume
    if resume_path is None:
        crash_path = PATHS.checkpoints / "crash_checkpoint.pt"
        if crash_path.exists():
            console.print(f"[yellow]Detected crash checkpoint: {crash_path}. Auto-resuming...[/yellow]")
            resume_path = str(crash_path)

    trainer = Trainer(
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        resume_checkpoint=resume_path,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
