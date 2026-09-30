"""
src/training/losses.py
=======================
Combined Multi-Task Loss Function for Universal Lane & Path Segmentation.

Loss Formulation
----------------
    L_total = 0.4 * L_CE + 0.4 * L_Dice + 0.2 * L_Boundary (+ aux_weight * L_Aux)

Key Components:
1. Focal Cross-Entropy Loss:
   - Evaluates per-pixel classification with label smoothing (0.05).
   - Focal loss modulation (1 - p_t)^gamma with gamma=2.0 on rare lane classes (2, 3)
     to suppress background over-dominance.
2. Soft Macro-Averaged Dice Loss:
   - Direct overlap optimization for thin continuous lane markings.
3. Boundary Loss:
   - Spatial morphological edge extraction targeting lane contour transitions.
   - Heavily penalizes boundary misalignment and lateral bleed.
4. Auxiliary Boundary Distance Loss:
   - Supervises the auxiliary 1-channel boundary head during training.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES: int = 4
DICE_SMOOTH: float = 1e-6
DEFAULT_CE_WEIGHT: float = 0.4
DEFAULT_DICE_WEIGHT: float = 0.4
DEFAULT_BOUNDARY_WEIGHT: float = 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Class Weights Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_class_weights(
    pixel_counts: dict[int, int],
    num_classes: int = NUM_CLASSES,
    device: Union[torch.device, str] = "cpu",
    strategy: str = "inverse_freq",
) -> torch.Tensor:
    """
    Compute per-class weights from pixel count statistics.
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
            f"Unknown strategy '{strategy}'. Choose 'inverse_freq', 'inverse_sqrt_freq', or 'median_freq'."
        )

    raw = raw / raw.mean()
    return raw.float().to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Focal Cross-Entropy Loss with Label Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class FocalCrossEntropyLoss(nn.Module):
    """
    Cross-Entropy with label smoothing and focal modulation on rare lane classes (2, 3).
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.ignore_index = ignore_index

        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)
        targets : (B, H, W)
        """
        B, C, H, W = logits.shape
        # Base smoothed cross entropy per pixel
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        # One-hot target with label smoothing
        targets_clamped = targets.clamp(0, C - 1)
        one_hot = F.one_hot(targets_clamped, C).permute(0, 3, 1, 2).float()
        if self.label_smoothing > 0.0:
            one_hot = one_hot * (1.0 - self.label_smoothing) + (self.label_smoothing / C)

        # Cross entropy per pixel: -sum(one_hot * log_p, dim=1)
        ce_per_pixel = -torch.sum(one_hot * log_p, dim=1)   # (B, H, W)

        # Focal modulation for lane classes (2 and 3)
        # p_t is the predicted probability of the true target class
        target_p = torch.gather(p, dim=1, index=targets_clamped.unsqueeze(1)).squeeze(1)
        focal_weight = torch.ones_like(ce_per_pixel)

        # Apply (1 - p_t)^gamma for lane classes
        is_lane = (targets == 2) | (targets == 3)
        focal_weight = torch.where(
            is_lane,
            torch.pow(1.0 - target_p + 1e-6, self.gamma),
            focal_weight,
        )

        loss = ce_per_pixel * focal_weight

        # Apply class weights if provided
        if self.weight is not None:
            w_per_pixel = self.weight[targets_clamped]
            loss = loss * w_per_pixel

        if self.ignore_index >= 0:
            valid = (targets != self.ignore_index).float()
            loss = (loss * valid).sum() / valid.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Soft Dice Loss
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Macro-averaged soft Dice Loss across all classes.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        smooth: float = DICE_SMOOTH,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.register_buffer("class_weights", torch.ones(num_classes, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()

        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid_mask
            targets_one_hot = targets_one_hot * valid_mask

        dice_per_class = torch.zeros(C, device=logits.device)
        for c in range(C):
            if c == self.ignore_index:
                continue
            p = probs[:, c].reshape(-1)
            t = targets_one_hot[:, c].reshape(-1)
            intersection = (p * t).sum()
            dice = (2.0 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)
            dice_per_class[c] = 1.0 - dice

        weights = self.class_weights.to(logits.device)
        return (dice_per_class * weights).sum() / weights.sum()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Boundary Loss
# ═══════════════════════════════════════════════════════════════════════════════

class BoundaryLoss(nn.Module):
    """
    Boundary loss that extracts spatial contours of lane markings and penalizes edge displacement.
    """

    def __init__(self, smooth: float = DICE_SMOOTH) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        # Extract lane probabilities (classes 2 & 3)
        lane_probs = probs[:, 2:4]  # (B, 2, H, W)

        targets_one_hot = F.one_hot(targets.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()
        lane_targets = targets_one_hot[:, 2:4]

        # Morphological boundary extraction via max_pool - min_pool
        p_max = F.max_pool2d(lane_probs, kernel_size=3, stride=1, padding=1)
        p_min = -F.max_pool2d(-lane_probs, kernel_size=3, stride=1, padding=1)
        p_boundary = p_max - p_min

        t_max = F.max_pool2d(lane_targets, kernel_size=3, stride=1, padding=1)
        t_min = -F.max_pool2d(-lane_targets, kernel_size=3, stride=1, padding=1)
        t_boundary = t_max - t_min

        # Soft Dice on extracted boundaries
        inter = (p_boundary * t_boundary).sum()
        union = p_boundary.sum() + t_boundary.sum()
        bnd_loss = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
        return torch.clamp(bnd_loss, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Combined Loss
# ═══════════════════════════════════════════════════════════════════════════════

class LossOutput(tuple):
    """
    3-tuple (total, ce, dice) with attribute access to all components (.total, .ce, .dice, .boundary, .aux).
    Ensures 100% backward compatibility with `total, ce, dice = criterion(...)`.
    """
    def __new__(cls, total: torch.Tensor, ce: torch.Tensor, dice: torch.Tensor, boundary: Optional[torch.Tensor] = None, aux: Optional[torch.Tensor] = None):
        obj = super().__new__(cls, (total, ce, dice))
        obj.total = total
        obj.ce = ce
        obj.dice = dice
        obj.boundary = boundary
        obj.aux = aux
        return obj


class CombinedLaneLoss(nn.Module):
    """
    Master combined loss function for LaneSegNet:
        L_total = 0.4 * L_CE + 0.4 * L_Dice + 0.2 * L_Boundary (+ aux_weight * L_Aux)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = DEFAULT_CE_WEIGHT,
        dice_weight: float = DEFAULT_DICE_WEIGHT,
        boundary_weight: Optional[float] = None,
        aux_weight: float = 0.2,
        ignore_index: int = -1,
        num_classes: int = NUM_CLASSES,
        label_smoothing: float = 0.05,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        # If boundary_weight is omitted, adapt to legacy configs where ce_weight + dice_weight == 1.0
        if boundary_weight is None:
            if abs(ce_weight + dice_weight - 1.0) < 1e-4:
                boundary_weight = 0.0
            else:
                boundary_weight = DEFAULT_BOUNDARY_WEIGHT

        # Validate primary weights sum
        weight_sum = ce_weight + dice_weight + boundary_weight
        assert abs(weight_sum - 1.0) < 1e-4, f"ce_weight + dice_weight + boundary_weight must equal 1.0 (got {weight_sum})"

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.aux_weight = aux_weight
        self.num_classes = num_classes

        ce_ignore = ignore_index if ignore_index >= 0 else -100
        self.ce = FocalCrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
            gamma=gamma,
            ignore_index=ce_ignore,
        )

        self.dice = DiceLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            ignore_index=ignore_index,
        )

        self.boundary = BoundaryLoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_logits: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        """
        Compute combined loss.
        """
        l_ce = self.ce(logits, targets)
        l_dice = self.dice(logits, targets)
        l_boundary = self.boundary(logits, targets)

        total = (
            self.ce_weight * l_ce
            + self.dice_weight * l_dice
            + self.boundary_weight * l_boundary
        )

        l_aux = None
        if aux_logits is not None:
            # Build target boundary for auxiliary head
            targets_one_hot = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes).permute(0, 3, 1, 2).float()
            lane_t = targets_one_hot[:, 2:4]
            t_max = F.max_pool2d(lane_t, 3, 1, 1)
            t_min = -F.max_pool2d(-lane_t, 3, 1, 1)
            target_bnd = torch.clamp((t_max - t_min).sum(dim=1, keepdim=True), 0.0, 1.0)

            l_aux = F.binary_cross_entropy_with_logits(aux_logits, target_bnd)
            total = total + self.aux_weight * l_aux

        return LossOutput(total, l_ce.detach(), l_dice.detach(), boundary=l_boundary.detach(), aux=l_aux.detach() if l_aux is not None else None)

    def update_class_weights(self, weights: torch.Tensor) -> None:
        """Update per-class weights dynamically during training."""
        device = next(self.parameters(), torch.tensor(0.0)).device
        w = weights.float().to(device)
        self.ce.weight = w
        self.dice.class_weights = w


# Convenience alias for user specification
CombinedLoss = CombinedLaneLoss
