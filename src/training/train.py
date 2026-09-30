"""
src/training/train.py
======================
Bulletproof Master Training Engine for APEX-LKA (LaneSegNet).
Optimized for NVIDIA RTX 4060 8GB and Jetson Orin Nano Deployment Targets.

Key Capabilities:
1. PyTorch Automatic Mixed Precision (AMP):
   - Fast fp16 training via `torch.cuda.amp`, auto-disabled on CPU.
   - Gradient accumulation to maintain large effective batch sizes.
2. Progressive Layer Unfreezing:
   - Epochs 1-5: Decoder-only training (encoder frozen).
   - Epochs 6-15: Last 2 encoder stages unfrozen with 0.1x LR.
   - Epochs 16+: Full model trained with uniform LR.
3. Crash Recovery & Exact-Batch Resume:
   - Captures model weights (raw + EMA), optimizer moments, scheduler state,
     AMP scaler, random states (Python, NumPy, PyTorch, CUDA), epoch, and batch index.
   - Auto-detects latest checkpoint on startup (override with --fresh or --resume).
   - Emergency SIGINT (Ctrl+C) / KeyboardInterrupt checkpointing.
4. Dynamic CUDA OOM Auto-Reduction:
   - Halves batch size, increases gradient accumulation, empties cache,
     rebuilds DataLoader, and retries the batch without user intervention.
5. Multi-Destination Metrics & Logging:
   - TensorBoard, CSV (experiment_log.csv), live JSON (live_metrics.json),
     Health Monitor CSV, and Rich console progress.
"""

from __future__ import annotations

import argparse
import copy
import logging
import signal
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    HealthMonitor,
    ModelCheckpoint,
    TensorBoardLogger,
)
from src.training.dataset import LaneSegDataset
from src.training.losses import CombinedLaneLoss
from src.training.model import LaneSegNet
from src.training.schedulers import ModelEMA, WarmupCosineWithRestarts, WarmupCosineScheduler
from src.training.utils import (
    compute_confusion_matrix,
    find_latest_checkpoint,
    metrics_from_confusion_matrix,
    set_random_states,
    update_live_metrics_json,
)

log = logging.getLogger("apex_trainer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Master Trainer Engine
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Production-grade training engine for LaneSegNet on RTX 4060 8GB.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        device: Optional[str] = None,
        resume_checkpoint: Optional[Union[str, Path]] = None,
        callbacks: Optional[List[Callback]] = None,
        epochs_override: Optional[int] = None,
        batch_size_override: Optional[int] = None,
        lr_override: Optional[float] = None,
        pretrained: Optional[bool] = None,
        fresh: bool = False,
    ) -> None:
        self.config = config or cfg

        # Device determination
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configuration sections
        t_cfg = self.config.get("training", {})
        m_cfg = self.config.get("model", {})
        p_cfg = self.config.get("preprocessing", {})
        adv_cfg = self.config.get("training_advanced", {})
        rob_cfg = self.config.get("robustness", {})
        rtx_cfg = self.config.get("rtx4060_optimized", {})

        # Hyperparameters
        self.epochs = epochs_override or t_cfg.get("epochs", 50)
        self.batch_size = (
            batch_size_override
            or rtx_cfg.get("default_batch_size")
            or t_cfg.get("batch_size", 8)
        )
        self.initial_batch_size = self.batch_size
        self.lr = lr_override or t_cfg.get("learning_rate", 1e-3)
        self.weight_decay = t_cfg.get("weight_decay", 1e-4)
        self.num_workers = rtx_cfg.get("default_num_workers", t_cfg.get("num_workers", 0))
        # Windows multiprocessing safety
        if sys.platform.startswith("win"):
            self.num_workers = 0

        self.pin_memory = (
            rtx_cfg.get("default_pin_memory", False) and self.device.type == "cuda"
        )
        self.grad_clip = t_cfg.get("gradient_clip", 1.0)
        self.grad_accum_steps = adv_cfg.get("gradient_accumulation_steps", 1)
        self.num_classes = m_cfg.get("target_num_classes", 4)
        self.img_height = p_cfg.get("image_height", 360)
        self.img_width = p_cfg.get("image_width", 640)

        # Progressive unfreezing schedule
        self.progressive_unfreezing = adv_cfg.get("progressive_unfreezing", True)
        self.unfreeze_schedule = adv_cfg.get("unfreeze_schedule", [5, 15])

        # Automatic Mixed Precision (AMP)
        mixed_precision_req = adv_cfg.get("mixed_precision", True)
        self.use_amp = (self.device.type == "cuda") and mixed_precision_req
        self.scaler: Optional[torch.cuda.amp.GradScaler] = (
            torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        )

        # Thermal protection flag
        self.thermal_cooldown_active: bool = False

        # State tracking
        self.start_epoch: int = 0
        self.start_batch_idx: int = 0
        self.best_metric: float = -1.0
        self.should_stop: bool = False
        self._current_epoch: int = 0
        self._current_batch_idx: int = 0
        self._total_train_batches: int = 1

        # 1. Build Model
        use_pretrained = (
            pretrained if pretrained is not None else m_cfg.get("pretrained_backbone", True)
        )
        self.model = LaneSegNet(num_classes=self.num_classes, pretrained=use_pretrained)
        self.model.to(self.device)

        # 2. Build Model EMA
        ema_decay = adv_cfg.get("ema_decay", 0.999)
        self.ema = ModelEMA(self.model, decay=ema_decay, device=self.device)

        # 3. Build Criterion
        ce_w = t_cfg.get("ce_weight", 0.4)
        dice_w = t_cfg.get("dice_weight", 0.4)
        boundary_w = t_cfg.get("boundary_weight", None)
        label_smooth = adv_cfg.get("label_smoothing", 0.05)
        self.criterion = CombinedLaneLoss(
            ce_weight=ce_w,
            dice_weight=dice_w,
            boundary_weight=boundary_w,
            num_classes=self.num_classes,
            label_smoothing=label_smooth,
        )
        self.criterion.to(self.device)

        # 4. Build Optimizer
        self.optimizer = self._build_optimizer(t_cfg.get("optimizer", "adam").lower())

        # 5. Build Scheduler
        warmup_ep = adv_cfg.get("warmup_epochs", 5)
        restart_ep = adv_cfg.get("cosine_restart_epochs", 15)
        self.scheduler = WarmupCosineWithRestarts(
            self.optimizer,
            warmup_epochs=warmup_ep,
            restart_epochs=restart_ep,
            max_epochs=self.epochs,
            warmup_start_lr=1e-6,
            min_lr=1e-6,
        )

        # 6. Datasets & Loaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if self.train_dataset is None:
            self._init_datasets()

        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self._build_dataloaders()

        # 7. Callbacks
        self.callbacks = callbacks if callbacks is not None else self._default_callbacks()

        # 8. Checkpoint resume detection
        target_ckpt = resume_checkpoint
        if target_ckpt is None and not fresh:
            # Auto-detect latest checkpoint in models/checkpoints
            auto_found = find_latest_checkpoint(PATHS.checkpoints)
            if auto_found:
                log.info("Auto-detected existing checkpoint for resume: %s", auto_found)
                target_ckpt = auto_found

        if target_ckpt:
            self.load_checkpoint(target_ckpt)

        # 9. Register system signals for emergency crash recovery
        self._register_signals()

    # ── Initialization Helpers ─────────────────────────────────────────────────

    def _build_optimizer(self, opt_name: str) -> torch.optim.Optimizer:
        """Construct parameter groups for progressive unfreezing and weight decay."""
        # Separate parameters:
        # Group 0: Decoder, fusion, and heads
        # Group 1: Encoder stages
        head_params = []
        encoder_params = []

        for name, param in self.model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {"params": head_params, "lr": self.lr, "weight_decay": self.weight_decay},
            {"params": encoder_params, "lr": self.lr, "weight_decay": self.weight_decay},
        ]

        if opt_name == "adamw":
            return torch.optim.AdamW(param_groups)
        elif opt_name == "sgd":
            return torch.optim.SGD(param_groups, momentum=0.9)
        return torch.optim.Adam(param_groups)

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
            self._total_train_batches = len(self.train_loader)

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
        log_csv = PATHS.logs / "training" / "experiment_log.csv"
        health_csv = PATHS.logs / "training" / "health_monitor.csv"
        tb_dir = PATHS.logs / "training" / "tensorboard"

        return [
            ModelCheckpoint(
                checkpoint_dir=ckpt_dir,
                save_every_n_epochs=1,
                keep_last=5,
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
            HealthMonitor(
                log_csv=health_csv,
                gpu_temp_threshold=self.config.get("robustness", {}).get("gpu_temp_threshold_celsius", 85.0),
                disk_space_min_gb=self.config.get("robustness", {}).get("disk_space_min_gb", 1.0),
                cooldown_seconds=self.config.get("robustness", {}).get("thermal_cooldown_sec", 5.0),
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

    # ── Progressive Unfreezing ─────────────────────────────────────────────────

    def _apply_progressive_unfreezing(self, epoch: int) -> None:
        """
        Dynamically configures encoder layer unfreezing based on epoch schedule:
        - Epochs 1-5 (epoch 0..4): Decoder trains only (encoder frozen).
        - Epochs 6-15 (epoch 5..14): Last 2 encoder stages unfrozen with 0.1x LR.
        - Epochs 16+ (epoch 15+): Full model trains with uniform LR.
        """
        if not self.progressive_unfreezing or not hasattr(self.model, "encoder"):
            return

        e_thresh1 = self.unfreeze_schedule[0] if len(self.unfreeze_schedule) > 0 else 5
        e_thresh2 = self.unfreeze_schedule[1] if len(self.unfreeze_schedule) > 1 else 15

        encoder = self.model.encoder
        features = getattr(encoder, "features", None)
        current_base_lr = self.optimizer.param_groups[0]["lr"]

        if epoch < e_thresh1:
            # Phase 1: Freeze all encoder weights
            for p in encoder.parameters():
                p.requires_grad = False
            if len(self.optimizer.param_groups) > 1:
                self.optimizer.param_groups[1]["lr"] = 0.0

        elif epoch < e_thresh2:
            # Phase 2: Unfreeze last 2 encoder stages with 0.1x LR
            if features is not None:
                for idx, block in enumerate(features):
                    req = (idx >= 8)  # Last stages C3 (48ch) and C4 (96ch)
                    for p in block.parameters():
                        p.requires_grad = req
            else:
                for p in encoder.parameters():
                    p.requires_grad = True

            if len(self.optimizer.param_groups) > 1:
                self.optimizer.param_groups[1]["lr"] = current_base_lr * 0.1

        else:
            # Phase 3: Full model unfreezing with uniform LR
            for p in encoder.parameters():
                p.requires_grad = True
            if len(self.optimizer.param_groups) > 1:
                self.optimizer.param_groups[1]["lr"] = current_base_lr

    # ── Checkpointing & Crash Recovery ─────────────────────────────────────────

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """
        Load complete training state from checkpoint.
        Supports exact-batch resume with random state restoration.
        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        log.info("Loading training checkpoint: %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=self.device)

        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            saved_groups = state["optimizer_state_dict"].get("param_groups", [])
            if len(saved_groups) != len(self.optimizer.param_groups):
                # Adapt optimizer to match saved parameter groups structure
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(state["optimizer_state_dict"])

        if self.scheduler and state.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(state["scheduler_state_dict"])

        if self.ema and state.get("ema_state_dict"):
            self.ema.load_state_dict(state["ema_state_dict"])

        if self.scaler and state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])

        if "random_states" in state and isinstance(state["random_states"], dict):
            set_random_states(state["random_states"])

        self.start_epoch = int(state.get("epoch", 0))
        self.start_batch_idx = int(state.get("batch_idx", 0))
        self.best_metric = float(state.get("best_metric", -1.0))

        # Restore saved batch size & grad_accum_steps if modified by OOM handler
        saved_bs = state.get("batch_size")
        if saved_bs and saved_bs != self.batch_size:
            log.info("Restoring batch size from checkpoint: %d -> %d", self.batch_size, saved_bs)
            self.batch_size = saved_bs
            self._build_dataloaders()

        if "gradient_accumulation_steps" in state:
            self.grad_accum_steps = int(state["gradient_accumulation_steps"])

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
        Train for a single epoch with PyTorch AMP, gradient accumulation, and OOM auto-recovery.
        """
        self.model.train()
        self._apply_progressive_unfreezing(epoch)

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

                accum_loss = 0.0
                self.optimizer.zero_grad(set_to_none=True)

                for b_i, batch_data in enumerate(self.train_loader):
                    if b_i < batch_idx:
                        continue

                    batch_idx = b_i
                    self._current_epoch = epoch
                    self._current_batch_idx = batch_idx

                    for cb in self.callbacks:
                        cb.on_batch_start(self, epoch, batch_idx)

                    images, targets = batch_data[0], batch_data[1]
                    images = images.to(self.device, non_blocking=self.pin_memory)
                    targets = targets.to(self.device, non_blocking=self.pin_memory)

                    with autocast_ctx:
                        logits = self.model(images)
                        loss, l_ce, l_dice = self.criterion(logits, targets)
                        loss_normalized = loss / max(1, self.grad_accum_steps)

                    # Backward with AMP scaling
                    if self.scaler is not None:
                        self.scaler.scale(loss_normalized).backward()
                    else:
                        loss_normalized.backward()

                    accum_loss += loss.item()

                    # Gradient Accumulation Step
                    if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                        if self.scaler is not None:
                            if self.grad_clip > 0:
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), self.grad_clip
                                )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            if self.grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.parameters(), self.grad_clip
                                )
                            self.optimizer.step()

                        self.optimizer.zero_grad(set_to_none=True)

                        # Update EMA shadow weights
                        if self.ema is not None:
                            self.ema.update(self.model)

                    l_val = loss.item()
                    total_loss += l_val
                    total_ce += l_ce.item() if hasattr(l_ce, "item") else float(l_ce)
                    total_dice += l_dice.item() if hasattr(l_dice, "item") else float(l_dice)
                    steps += 1

                    for cb in self.callbacks:
                        cb.on_batch_end(self, epoch, batch_idx, l_val)

                break  # Completed epoch successfully

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
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
                self.grad_accum_steps = max(1, self.grad_accum_steps * 2)

                log.warning(
                    "[OOM Auto-Reduction] Caught CUDA OOM at epoch %d, batch %d! "
                    "Reducing batch size: %d -> %d, increasing grad_accum_steps: -> %d. Resuming...",
                    epoch + 1, batch_idx, old_bs, self.batch_size, self.grad_accum_steps
                )

                # Rebuild DataLoader with halved batch size and resume from current batch
                self._build_dataloaders()

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

        for batch_data in self.val_loader:
            images, targets = batch_data[0], batch_data[1]
            images = images.to(self.device, non_blocking=self.pin_memory)
            targets = targets.to(self.device, non_blocking=self.pin_memory)

            logits = self.model(images)
            loss, l_ce, l_dice = self.criterion(logits, targets)

            total_loss += loss.item()
            total_ce += l_ce.item() if hasattr(l_ce, "item") else float(l_ce)
            total_dice += l_dice.item() if hasattr(l_dice, "item") else float(l_dice)
            steps += 1

            preds = torch.argmax(logits, dim=1)
            conf_mat += compute_confusion_matrix(preds, targets, num_classes=self.num_classes)

        # Restore active weights
        if self.ema is not None:
            self.ema.restore(self.model)

        metrics = metrics_from_confusion_matrix(conf_mat)
        metrics["val_loss"] = round(total_loss / max(1, steps), 4)
        metrics["val_ce"] = round(total_ce / max(1, steps), 4)
        metrics["val_dice"] = round(total_dice / max(1, steps), 4)
        metrics["val_mean_iou"] = metrics.get("mean_iou", 0.0)

        return metrics

    # ── Main Training Loop ─────────────────────────────────────────────────────

    def fit(self) -> Dict[str, float]:
        """
        Execute full training loop across all epochs with fault tolerance.
        """
        console.print(
            Panel(
                f"[bold cyan]APEX-LKA Phase 3 Training Engine[/bold cyan]\n"
                f"Model: LaneSegNet (Params: {sum(p.numel() for p in self.model.parameters()):,})\n"
                f"Device: {self.device} (AMP: {self.use_amp})\n"
                f"Target Epochs: {self.epochs} (Start: {self.start_epoch + 1})\n"
                f"Batch Size: {self.batch_size} (Grad Accum: {self.grad_accum_steps})\n"
                f"Base Learning Rate: {self.lr}",
                border_style="cyan",
            )
        )

        for cb in self.callbacks:
            cb.on_train_start(self)

        last_metrics: Dict[str, float] = {}

        try:
            for epoch in range(self.start_epoch, self.epochs):
                if self.should_stop:
                    log.info("Early stopping condition triggered. Ending training.")
                    break

                for cb in self.callbacks:
                    cb.on_epoch_start(self, epoch)

                epoch_start_time = time.time()

                # Resume batch_idx on the first resumed epoch only
                batch_start = self.start_batch_idx if epoch == self.start_epoch else 0
                train_metrics = self._train_epoch(epoch, start_batch_idx=batch_start)

                val_metrics = self._val_epoch(epoch)

                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

                # Consolidate epoch metrics
                metrics = {**train_metrics, **val_metrics}
                epoch_time = time.time() - epoch_start_time
                metrics["epoch_time_sec"] = round(epoch_time, 2)
                last_metrics = metrics

                current_lr = (
                    self.optimizer.param_groups[0]["lr"]
                    if self.optimizer.param_groups
                    else self.lr
                )

                # Update live JSON for dashboard
                live_json = PATHS.logs / "training" / "live_metrics.json"
                batches_in_epoch = len(self.train_loader) if self.train_loader else 1
                fps = batches_in_epoch / max(0.1, epoch_time)
                update_live_metrics_json(
                    filepath=live_json,
                    epoch=epoch,
                    total_epochs=self.epochs,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    lr=current_lr,
                    speed_fps=fps,
                    status="TRAINING",
                )

                # Rich console summary table
                self._print_epoch_table(epoch, train_metrics, val_metrics, epoch_time, current_lr)

                for cb in self.callbacks:
                    cb.on_epoch_end(self, epoch, metrics)

        except KeyboardInterrupt as ki:
            log.warning("Training interrupted by user (KeyboardInterrupt). Saving emergency state...")
            self._handle_crash(
                epoch=getattr(self, "_current_epoch", 0),
                batch_idx=getattr(self, "_current_batch_idx", 0),
                error=ki,
            )
            # Update live json to INTERRUPTED
            update_live_metrics_json(
                filepath=PATHS.logs / "training" / "live_metrics.json",
                epoch=getattr(self, "_current_epoch", 0),
                total_epochs=self.epochs,
                train_metrics=last_metrics,
                val_metrics={},
                lr=0.0,
                speed_fps=0.0,
                status="INTERRUPTED",
            )
            raise ki

        except BaseException as exc:
            log.error("Fatal error during training loop: %s", exc)
            self._handle_crash(
                epoch=getattr(self, "_current_epoch", 0),
                batch_idx=getattr(self, "_current_batch_idx", 0),
                error=exc,
            )
            raise exc

        for cb in self.callbacks:
            cb.on_train_end(self)

        update_live_metrics_json(
            filepath=PATHS.logs / "training" / "live_metrics.json",
            epoch=self.epochs - 1,
            total_epochs=self.epochs,
            train_metrics=last_metrics,
            val_metrics={},
            lr=0.0,
            speed_fps=0.0,
            status="FINISHED",
        )

        console.print("[bold green]Training finished successfully![/bold green]")
        summary = {
            "total_epochs": self.epochs,
            "best_metric": self.best_metric,
            **last_metrics,
        }
        return summary

    def train(self) -> Dict[str, Any]:
        """Backward-compatible alias for fit() returning complete summary dict."""
        return self.fit()


    def _print_epoch_table(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float,
        lr: float,
    ) -> None:
        """Render beautiful Rich table summary for the epoch."""
        tbl = Table(title=f"Epoch {epoch + 1}/{self.epochs} Summary ({epoch_time:.1f}s, LR: {lr:.2e})", show_lines=True)
        tbl.add_column("Split", style="cyan")
        tbl.add_column("Loss", justify="right")
        tbl.add_column("mIoU", justify="right")
        tbl.add_column("Lane mIoU", justify="right")
        tbl.add_column("Dice", justify="right")

        tbl.add_row(
            "Train",
            f"{train_metrics.get('train_loss', 0.0):.4f}",
            "-",
            "-",
            f"{train_metrics.get('train_dice', 0.0):.4f}",
        )
        if val_metrics:
            tbl.add_row(
                "[bold green]Validation[/bold green]",
                f"{val_metrics.get('val_loss', 0.0):.4f}",
                f"[bold cyan]{val_metrics.get('mean_iou', 0.0):.4f}[/bold cyan]",
                f"[bold yellow]{val_metrics.get('lane_mean_iou', 0.0):.4f}[/bold yellow]",
                f"{val_metrics.get('val_dice', 0.0):.4f}",
            )
        console.print(tbl)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="APEX-LKA Master Training Engine")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per GPU step")
    parser.add_argument("--lr", type=float, default=None, help="Target base learning rate")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--fresh", action="store_true", help="Start training from scratch (ignore existing checkpoints)")
    parser.add_argument("--config", type=str, default=None, help="Custom configuration YAML path")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Target device")
    args = parser.parse_args()

    custom_cfg = None
    if args.config:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            custom_cfg = yaml.safe_load(f)

    trainer = Trainer(
        config=custom_cfg,
        device=args.device,
        resume_checkpoint=args.resume,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        lr_override=args.lr,
        fresh=args.fresh,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
