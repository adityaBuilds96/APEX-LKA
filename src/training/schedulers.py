"""
src/training/schedulers.py
===========================
Learning rate schedulers and weight averaging utilities for APEX-LKA training.

Includes:
1. WarmupCosineScheduler: Linear warmup for the initial N epochs followed by
   Cosine Annealing down to min_lr.
2. ModelEMA: Exponential Moving Average of model parameters and buffers for
   stabilized, robust inference.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:  # PyTorch < 2.0 fallback
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Warmup + Cosine Annealing Learning Rate Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler(LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    Phase 1 (Warmup: epoch 0 to warmup_epochs - 1):
        Linear ramp from `warmup_start_lr` to `base_lr`.
        lr(epoch) = warmup_start_lr + (base_lr - warmup_start_lr) * (epoch + 1) / warmup_epochs

    Phase 2 (Cosine Decay: epoch >= warmup_epochs):
        Cosine decay from `base_lr` down to `min_lr` across remaining epochs.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimizer.
    warmup_epochs : int, default 5
        Number of epochs for linear warmup.
    max_epochs : int, default 50
        Total number of training epochs.
    warmup_start_lr : float, default 1e-6
        Initial learning rate at the start of warmup.
    min_lr : float, default 1e-6
        Minimum target learning rate at the end of training.
    last_epoch : int, default -1
        The index of the last epoch (used for resuming).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.max_epochs = max(1, int(max_epochs))
        self.warmup_start_lr = float(warmup_start_lr)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        """Compute current learning rate for each param group."""
        epoch = self.last_epoch

        # If warmup is configured and we are in the warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            alpha = float(epoch + 1) / float(self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        # Post-warmup cosine annealing: reaches min_lr at epoch == max_epochs - 1
        cosine_steps = max(1, self.max_epochs - 1 - self.warmup_epochs)
        current_step = max(0, epoch - self.warmup_epochs)
        progress = float(current_step) / float(cosine_steps)
        progress = min(1.0, max(0.0, progress))

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]

    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary including warmup and cosine bounds."""
        state = super().state_dict()
        state["warmup_epochs"] = self.warmup_epochs
        state["max_epochs"] = self.max_epochs
        state["warmup_start_lr"] = self.warmup_start_lr
        state["min_lr"] = self.min_lr
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        state = dict(state_dict)
        self.warmup_epochs = state.pop("warmup_epochs", self.warmup_epochs)
        self.max_epochs = state.pop("max_epochs", self.max_epochs)
        self.warmup_start_lr = state.pop("warmup_start_lr", self.warmup_start_lr)
        self.min_lr = state.pop("min_lr", self.min_lr)
        super().load_state_dict(state)


class WarmupCosineWithRestarts(LRScheduler):
    """
    Learning rate scheduler with linear warmup followed by Cosine Annealing with Warm Restarts.

    Phase 1 (Warmup: epoch 0 to warmup_epochs - 1):
        Linear ramp from warmup_start_lr (default 1e-6) to target base_lr.

    Phase 2 (Cosine Annealing with Restarts: epoch >= warmup_epochs):
        Cosine decay restarting every `restart_epochs` (default 15).
        At the start of each restart period, LR resets to base_lr and decays smoothly to min_lr.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int = 5,
        restart_epochs: int = 15,
        max_epochs: int = 50,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.restart_epochs = max(1, int(restart_epochs))
        self.max_epochs = max(1, int(max_epochs))
        self.warmup_start_lr = float(warmup_start_lr)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch

        # Warmup phase
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            alpha = float(epoch + 1) / float(self.warmup_epochs)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        # Post-warmup with periodic warm restarts
        current_step = max(0, epoch - self.warmup_epochs)
        progress = float(current_step % self.restart_epochs) / float(self.restart_epochs)
        progress = min(1.0, max(0.0, progress))

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]

    def state_dict(self) -> Dict[str, Any]:
        state = super().state_dict()
        state["warmup_epochs"] = self.warmup_epochs
        state["restart_epochs"] = self.restart_epochs
        state["max_epochs"] = self.max_epochs
        state["warmup_start_lr"] = self.warmup_start_lr
        state["min_lr"] = self.min_lr
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state = dict(state_dict)
        self.warmup_epochs = state.pop("warmup_epochs", self.warmup_epochs)
        self.restart_epochs = state.pop("restart_epochs", self.restart_epochs)
        self.max_epochs = state.pop("max_epochs", self.max_epochs)
        self.warmup_start_lr = state.pop("warmup_start_lr", self.warmup_start_lr)
        self.min_lr = state.pop("min_lr", self.min_lr)
        super().load_state_dict(state)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Model Exponential Moving Average (EMA)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelEMA:
    """
    Maintains Exponential Moving Average (EMA) of model parameters and buffers.

    Ensures smooth and noise-resilient inference weights:
        theta_ema = decay * theta_ema + (1 - decay) * theta_model

    Parameters
    ----------
    model : nn.Module
        The active training model to track.
    decay : float, default 0.999
        EMA smoothing factor.
    device : Optional[torch.device | str], default None
        Device on which the EMA shadow model resides (None = match model device).
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"EMA decay must be in [0.0, 1.0], got {decay}")

        self.decay = decay
        self.device = torch.device(device) if device is not None else None
        self.step_count: int = 0
        self._backup: Optional[Dict[str, torch.Tensor]] = None

        # Build deepcopy for shadow model
        self.ema_model: nn.Module = copy.deepcopy(model)
        if self.device is not None:
            self.ema_model.to(self.device)

        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update the EMA shadow weights using current active model weights.

        Parameters
        ----------
        model : nn.Module
            The active model after optimizer step.
        """
        self.step_count += 1
        decay = self.decay

        target_device = next(self.ema_model.parameters()).device

        # In-place linear interpolation for parameters
        for p_ema, p_model in zip(self.ema_model.parameters(), model.parameters()):
            p_m = p_model.data.to(target_device)
            if p_model.requires_grad:
                p_ema.data.lerp_(p_m, 1.0 - decay)
            else:
                p_ema.data.copy_(p_m)

        # Exact copy for buffers (BatchNorm running_mean, running_var, etc.)
        for b_ema, b_model in zip(self.ema_model.buffers(), model.buffers()):
            b_ema.data.copy_(b_model.data.to(target_device))

    def apply_shadow(self, model: nn.Module) -> None:
        """
        Back up current model parameters and load EMA shadow weights into `model`.
        Typically used right before running validation or inference.
        """
        self._backup = {
            name: param.data.clone() for name, param in model.named_parameters()
        }
        for name, param in model.named_parameters():
            if name in self.ema_model.state_dict():
                param.data.copy_(self.ema_model.state_dict()[name].data.to(param.device))

    def restore(self, model: nn.Module) -> None:
        """
        Restore backed-up parameters to `model` after validation or export.
        """
        if self._backup is None:
            return
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name].to(param.device))
        self._backup = None

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get pure state dict of EMA model for production export.
        """
        return copy.deepcopy(self.ema_model.state_dict())

    def state_dict(self) -> Dict[str, object]:
        """
        Full state dict for checkpointing (includes decay, step count, shadow weights).
        """
        return {
            "decay": self.decay,
            "step_count": self.step_count,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore EMA state from checkpoint."""
        self.decay = float(state.get("decay", self.decay))
        self.step_count = int(state.get("step_count", 0))
        if "ema_model" in state:
            self.ema_model.load_state_dict(state["ema_model"])
        elif "model_state_dict" in state:
            self.ema_model.load_state_dict(state["model_state_dict"])
