"""
src/training/losses.py
=======================
Combined loss function for LaneSegNet 4-class lane segmentation.

Loss Formula
------------
    L_total = w_ce × L_CE  +  w_dice × L_Dice

where:
    L_CE   = class-frequency-weighted Cross-Entropy
    L_Dice = macro-averaged soft Dice Loss (per class, then mean)
    w_ce   = w_dice = 0.5   (configurable)

Motivation
----------
• Cross-Entropy (CE): Excellent gradient signal for general region
  classification — penalises wrong predictions uniformly across the image.

• Dice Loss: Addresses extreme class imbalance. Lane markings typically
  represent < 2% of image pixels. CE alone will learn to predict background
  and still achieve 98% pixel accuracy. Dice is directly proportional to the
  IoU overlap on each thin lane class, forcing the network to attend to them.

• Class-frequency weighting on CE: Class weights are inversely proportional
  to class frequency (inverse-median-frequency balancing). A wrong prediction
  on a single lane pixel is penalised ~30–50× more than a background pixel.

Classes
-------
0  background
1  road
2  left_lane    ← rare, must not be missed
3  right_lane   ← rare, must not be missed
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES: int = 4
DICE_SMOOTH:  float = 1e-6   # Laplace smoothing to prevent ÷0
DEFAULT_CE_WEIGHT:   float = 0.5
DEFAULT_DICE_WEIGHT: float = 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Class-frequency weight helper
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(
    pixel_counts: dict[int, int],
    num_classes:  int = NUM_CLASSES,
    device:       torch.device | str = "cpu",
    strategy:     str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute per-class CE weights from pixel count statistics.

    Parameters
    ----------
    pixel_counts : dict mapping class_id → total pixel count across dataset.
    num_classes  : number of classes (must equal the model output channels).
    device       : target device for the returned tensor.
    strategy     : ``"inverse_freq"``     — w_c = total / (n_classes × count_c)
                   ``"inverse_sqrt_freq"``— w_c = 1 / sqrt(freq_c)  (softer)
                   ``"median_freq"``      — w_c = median_freq / freq_c

    Returns
    -------
    weights : (num_classes,) float32 tensor, normalised so mean == 1.
    """
    total = sum(pixel_counts.values())
    if total == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=device)

    counts = torch.tensor(
        [pixel_counts.get(c, 1) for c in range(num_classes)],
        dtype=torch.float64,
    )
    freqs = counts / total

    if strategy == "inverse_freq":
        raw = 1.0 / (freqs * num_classes)
    elif strategy == "inverse_sqrt_freq":
        raw = 1.0 / freqs.sqrt()
    elif strategy == "median_freq":
        median = freqs.median()
        raw = median / freqs
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose 'inverse_freq', 'inverse_sqrt_freq', or 'median_freq'."
        )

    # Normalise so that mean weight == 1  (keeps loss magnitude stable)
    raw = raw / raw.mean()
    return raw.float().to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Soft Dice Loss
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Macro-averaged soft Dice Loss across all classes.

    Operates on logits (applies softmax internally) and supports
    optional per-class weighting to further penalise lane classes.

    Parameters
    ----------
    num_classes    : number of segmentation classes
    smooth         : Laplace smoothing constant
    class_weights  : optional (num_classes,) tensor; classes with higher
                     weight contribute more to the total loss.
    ignore_index   : class index to ignore in the mean (default -1 = none).
    """

    def __init__(
        self,
        num_classes:   int = NUM_CLASSES,
        smooth:        float = DICE_SMOOTH,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index:  int = -1,
    ) -> None:
        super().__init__()
        self.num_classes   = num_classes
        self.smooth        = smooth
        self.ignore_index  = ignore_index

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer(
                "class_weights",
                torch.ones(num_classes, dtype=torch.float32),
            )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)  float32  raw logits
        targets : (B, H, W)     int64    class indices

        Returns
        -------
        scalar Dice loss
        """
        B, C, H, W = logits.shape

        # Probabilities
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot encode targets: (B, C, H, W)
        targets_one_hot = F.one_hot(targets.clamp(0, C - 1), C)   # (B,H,W,C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # Mask for ignore_index
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs          = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

        dice_per_class = torch.zeros(C, device=logits.device)
        for c in range(C):
            if c == self.ignore_index:
                continue
            p = probs[:, c].reshape(-1)         # (B×H×W,)
            t = targets_one_hot[:, c].reshape(-1)

            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (
                p.sum() + t.sum() + self.smooth
            )
            dice_per_class[c] = 1.0 - dice     # loss = 1 - Dice score

        # Weighted mean over classes
        weights = self.class_weights.to(logits.device)
        loss = (dice_per_class * weights).sum() / weights.sum()
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Combined CE + Dice loss
# ═══════════════════════════════════════════════════════════════════════════════

class CombinedLaneLoss(nn.Module):
    """
    Primary training loss for LaneSegNet.

        L = ce_weight × L_CE  +  dice_weight × L_Dice

    Both sub-losses use the same class_weights tensor to ensure consistent
    emphasis on rare lane classes.

    Parameters
    ----------
    class_weights  : (num_classes,) float32 tensor, or None for uniform.
    ce_weight      : scalar weight on Cross-Entropy component (default 0.5).
    dice_weight    : scalar weight on Dice component          (default 0.5).
    ignore_index   : class index to exclude from both losses  (default -1 = none).
    num_classes    : number of segmentation classes           (default 4).
    label_smoothing: CE label smoothing factor [0, 1)         (default 0.05).

    Example
    -------
    >>> import torch
    >>> criterion = CombinedLaneLoss()
    >>> logits  = torch.randn(2, 4, 360, 640)
    >>> targets = torch.randint(0, 4, (2, 360, 640))
    >>> loss = criterion(logits, targets)
    >>> loss.shape
    torch.Size([])
    """

    def __init__(
        self,
        class_weights:   Optional[torch.Tensor] = None,
        ce_weight:       float = DEFAULT_CE_WEIGHT,
        dice_weight:     float = DEFAULT_DICE_WEIGHT,
        ignore_index:    int   = -1,
        num_classes:     int   = NUM_CLASSES,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        assert abs(ce_weight + dice_weight - 1.0) < 1e-6, (
            "ce_weight + dice_weight must equal 1.0"
        )

        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes

        # ── Cross-Entropy ─────────────────────────────────────────────────────
        ce_ignore  = ignore_index if ignore_index >= 0 else -100   # PyTorch convention
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ce_ignore,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

        # ── Dice ──────────────────────────────────────────────────────────────
        self.dice = DiceLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)  float32  raw model output
        targets : (B, H, W)     int64    ground-truth class indices

        Returns
        -------
        total_loss : scalar — the combined loss used for backprop
        ce_loss    : scalar — Cross-Entropy component (detached from graph)
        dice_loss  : scalar — Dice component          (detached from graph)
        """
        l_ce   = self.ce(logits, targets)
        l_dice = self.dice(logits, targets)

        total = self.ce_weight * l_ce + self.dice_weight * l_dice
        return total, l_ce.detach(), l_dice.detach()

    # ── Convenience: update class weights at runtime ───────────────────────────
    def update_class_weights(self, weights: torch.Tensor) -> None:
        """
        Replace the class weights on both CE and Dice heads.

        Useful when weights are computed from the actual dataset after init.
        """
        device = next(self.parameters(), torch.tensor(0.0)).device
        w = weights.float().to(device)
        self.ce.weight = w
        self.dice.class_weights = w
