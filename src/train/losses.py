"""
src/train/losses.py
===================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Composite Multi-Objective Loss Suite for 4-Class Semantic Lane Segmentation.
ISO 26262 ASIL-B High-Precision Boundary Delimitation & Numerical Stability.

Loss Formulations:
------------------
1. Class-Weighted Focal Cross-Entropy Loss:
   Compensates for severe spatial imbalance where Background (0) constitutes >90%
   of pixels, while thin Left (2) and Right (3) lane boundaries constitute <2%.
   Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

2. Multi-Class Soft Dice Loss:
   Maximizes overlap between predicted lane contours and ground-truth polylines.
   Dice = (2 * |P intersect G| + eps) / (|P| + |G| + eps)

3. High-Precision Boundary (Edge) Loss:
   Applies Sobel/Laplacian spatial gradient filters to penalize boundary blurring,
   producing crisp, continuous lane lines without lateral dispersion.

4. Combined Master Objective:
   L_total = w_focal * L_focal + w_dice * L_dice + w_boundary * L_boundary
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema

logger = logging.getLogger("APEX.Losses")

EPSILON: float = 1e-7


# ──────────────────────────────────────────────────────────────────────────────
# 1. Class-Weighted Focal Loss
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with inverse-frequency class weighting and focusing parameter gamma.
    Suppresses easy background gradients to prioritize hard boundary transitions.
    """

    def __init__(
        self,
        num_classes: int = 4,
        alpha: Optional[Union[List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, list):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            else:
                self.register_buffer("alpha", alpha.float())
        else:
            # Default ASIL-B class weights: heavy penalty on lane boundaries
            default_weights = [0.15, 1.0, 4.5, 4.5]
            self.register_buffer("alpha", torch.tensor(default_weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes, H, W] raw unnormalized prediction scores
            targets: [B, H, W] ground-truth class indices in {0, 1, 2, 3}
        """
        if logits.dim() != 4:
            raise ValueError(f"Expected 4D logits [B, C, H, W], got {logits.shape}")
        if targets.dim() != 3:
            raise ValueError(f"Expected 3D targets [B, H, W], got {targets.shape}")

        b, c, h, w = logits.shape
        targets = targets.long()

        # Compute log-softmax and probabilities
        log_p = F.log_softmax(logits, dim=1)  # [B, C, H, W]
        p = torch.exp(log_p)                  # [B, C, H, W]

        # Gather target class probabilities
        targets_unsqueezed = targets.unsqueeze(1) # [B, 1, H, W]
        target_log_p = log_p.gather(dim=1, index=targets_unsqueezed).squeeze(1) # [B, H, W]
        target_p = p.gather(dim=1, index=targets_unsqueezed).squeeze(1)         # [B, H, W]

        # Modulating factor (1 - p_t)^gamma
        modulating_factor = (1.0 - target_p) ** self.gamma

        # Class weights alpha_t
        if self.alpha is not None:
            alpha_broadcast = self.alpha.to(logits.device)
            target_alpha = alpha_broadcast.gather(dim=0, index=targets.view(-1)).view(b, h, w)
        else:
            target_alpha = 1.0

        focal_loss = -target_alpha * modulating_factor * target_log_p

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-Class Soft Dice Loss
# ──────────────────────────────────────────────────────────────────────────────

class MulticlassDiceLoss(nn.Module):
    """
    Differentiable multi-class Soft Dice Loss evaluating intersection-over-union
    similarity across spatial dimensions for each semantic channel.
    """

    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1.0,
        ignore_background: bool = False,
        class_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.class_weights = class_weights or [0.2, 1.0, 2.0, 2.0]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, num_classes, H, W] raw predictions
            targets: [B, H, W] class indices
        """
        probs = F.softmax(logits, dim=1) # [B, C, H, W]
        b, c, h, w = probs.shape

        # One-hot encode targets: [B, C, H, W]
        targets_one_hot = F.one_hot(targets.clamp(0, c - 1), num_classes=c).permute(0, 3, 1, 2).float()

        dice_per_class: List[torch.Tensor] = []
        start_class = 1 if self.ignore_background else 0

        for cls_idx in range(start_class, c):
            p_cls = probs[:, cls_idx, :, :].contiguous().view(b, -1)
            t_cls = targets_one_hot[:, cls_idx, :, :].contiguous().view(b, -1)

            intersection = (p_cls * t_cls).sum(dim=1)
            cardinality = p_cls.sum(dim=1) + t_cls.sum(dim=1)

            dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + EPSILON)
            weight = self.class_weights[cls_idx] if cls_idx < len(self.class_weights) else 1.0
            dice_per_class.append(weight * (1.0 - dice.mean()))

        if not dice_per_class:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return torch.stack(dice_per_class).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 3. High-Precision Spatial Boundary (Laplacian) Loss
# ──────────────────────────────────────────────────────────────────────────────

class BoundaryLoss(nn.Module):
    """
    Computes spatial gradient boundary error between predicted lane line probability
    maps and true binary contours using fixed 2D Laplacian kernels.
    """

    def __init__(self, stroke_classes: Tuple[int, ...] = (2, 3)) -> None:
        super().__init__()
        self.stroke_classes = stroke_classes

        # Fixed 3x3 Laplacian edge-detection kernel
        laplacian_filter = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0) # [1, 1, 3, 3]

        self.register_buffer("laplacian_filter", laplacian_filter)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        b, c, h, w = probs.shape

        targets_one_hot = F.one_hot(targets.clamp(0, c - 1), num_classes=c).permute(0, 3, 1, 2).float()

        total_boundary_loss = torch.tensor(0.0, device=logits.device)
        count = 0

        kernel = self.laplacian_filter.to(logits.device)

        for cls_idx in self.stroke_classes:
            if cls_idx >= c:
                continue

            pred_channel = probs[:, cls_idx:cls_idx + 1, :, :]
            true_channel = targets_one_hot[:, cls_idx:cls_idx + 1, :, :]

            # Extract spatial boundary edges via 2D convolution
            pred_edges = torch.abs(F.conv2d(pred_channel, kernel, padding=1))
            true_edges = torch.abs(F.conv2d(true_channel, kernel, padding=1))

            edge_diff = F.smooth_l1_loss(pred_edges, true_edges, reduction="mean")
            total_boundary_loss = total_boundary_loss + edge_diff
            count += 1

        if count > 0:
            return total_boundary_loss / float(count)
        return total_boundary_loss


# ──────────────────────────────────────────────────────────────────────────────
# 4. Online Hard Example Mining (OHEM) Cross Entropy Loss
# ──────────────────────────────────────────────────────────────────────────────

class OHEMCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Cross Entropy:
    Only backpropagates gradients through the top-k highest loss pixels in each batch,
    forcing the network to resolve boundary ambiguities rather than learning easy highway asphalt.
    """

    def __init__(self, keep_ratio: float = 0.35, ignore_index: int = 255) -> None:
        super().__init__()
        self.keep_ratio = keep_ratio
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        b, c, h, w = logits.shape
        loss = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        loss_flat = loss.view(-1)

        num_pixels = loss_flat.numel()
        k = max(1, int(num_pixels * self.keep_ratio))

        topk_loss, _ = torch.topk(loss_flat, k, sorted=False)
        return topk_loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Lovasz-Softmax Surrogate Jaccard Loss
# ──────────────────────────────────────────────────────────────────────────────

class LovaszSoftmaxLoss(nn.Module):
    """
    Direct submodular extension surrogate for multi-class Jaccard Index (Mean IoU).
    Optimizes the global segmentation metric rather than per-pixel cross entropy.
    """

    def __init__(self, classes: Tuple[int, ...] = (1, 2, 3), per_image: bool = False) -> None:
        super().__init__()
        self.classes = classes
        self.per_image = per_image

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        losses: List[torch.Tensor] = []

        for c in self.classes:
            target_c = (targets == c).float()
            prob_c = probs[:, c, :, :]
            errors = (target_c - prob_c).abs()

            # Flatten
            errors_flat = errors.view(-1)
            target_flat = target_c.view(-1)

            if target_flat.sum() == 0 and prob_c.sum() == 0:
                continue

            # Sort errors
            perm = torch.argsort(errors_flat, descending=True)
            errors_sorted = errors_flat[perm]
            target_sorted = target_flat[perm]

            # Compute Lovasz gradient
            intersection = target_sorted.sum() - target_sorted.cumsum(0)
            union = target_sorted.sum() + (1.0 - target_sorted).cumsum(0)
            jaccard = 1.0 - (intersection / torch.clamp(union, min=1.0))

            if len(jaccard) > 1:
                jaccard[1:] = jaccard[1:] - jaccard[:-1]

            losses.append(torch.dot(errors_sorted, jaccard))

        if not losses:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return torch.stack(losses).mean()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Master Combined Lane Segmentation Loss
# ──────────────────────────────────────────────────────────────────────────────

class CombinedLaneSegmentationLoss(nn.Module):
    """
    Master production loss combining Focal, Soft Dice, and Spatial Boundary objectives.
    Optionally couples OHEM and Lovasz-Softmax for advanced IoU optimization.
    """

    def __init__(
        self,
        num_classes: int = 4,
        weight_focal: float = 1.0,
        weight_dice: float = 1.0,
        weight_boundary: float = 0.5,
        weight_ohem: float = 0.0,
        weight_lovasz: float = 0.0,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice
        self.weight_boundary = weight_boundary
        self.weight_ohem = weight_ohem
        self.weight_lovasz = weight_lovasz

        self.focal = FocalLoss(num_classes=num_classes, alpha=class_weights, gamma=focal_gamma)
        self.dice = MulticlassDiceLoss(num_classes=num_classes, class_weights=class_weights)
        self.boundary = BoundaryLoss(stroke_classes=(LaneClassID.LEFT_BOUNDARY.value, LaneClassID.RIGHT_BOUNDARY.value))
        self.ohem = OHEMCrossEntropyLoss() if weight_ohem > 0.0 else None
        self.lovasz = LovaszSoftmaxLoss() if weight_lovasz > 0.0 else None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss and return breakdown metrics for logging.
        """
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_boundary = self.boundary(logits, targets)

        total_loss = (
            self.weight_focal * loss_focal
            + self.weight_dice * loss_dice
            + self.weight_boundary * loss_boundary
        )

        metrics = {
            "loss_total": float(total_loss.detach().item()),
            "loss_focal": float(loss_focal.detach().item()),
            "loss_dice": float(loss_dice.detach().item()),
            "loss_boundary": float(loss_boundary.detach().item()),
        }

        if self.ohem is not None:
            loss_ohem = self.ohem(logits, targets)
            total_loss = total_loss + (self.weight_ohem * loss_ohem)
            metrics["loss_ohem"] = float(loss_ohem.detach().item())

        if self.lovasz is not None:
            loss_lovasz = self.lovasz(logits, targets)
            total_loss = total_loss + (self.weight_lovasz * loss_lovasz)
            metrics["loss_lovasz"] = float(loss_lovasz.detach().item())

        return total_loss, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6. Structural Longitudinal Continuity Loss
# ──────────────────────────────────────────────────────────────────────────────

class StructuralContinuityLoss(nn.Module):
    """
    Penalizes vertical gaps and fragmentation along lane markings.
    Enforces that lane boundaries maintain smooth longitudinal spatial continuity.
    """

    def __init__(self, lane_classes: Tuple[int, ...] = (2, 3)) -> None:
        super().__init__()
        self.lane_classes = lane_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        total_loss = torch.tensor(0.0, device=logits.device)
        count = 0

        for c in self.lane_classes:
            p_c = probs[:, c, :, :]  # [B, H, W]
            # Vertical gradient along y-axis
            diff_y = torch.abs(p_c[:, 1:, :] - p_c[:, :-1, :])
            # We want small second-order differences along vertical column slices
            total_loss = total_loss + diff_y.mean()
            count += 1

        if count > 0:
            return total_loss / float(count)
        return total_loss


# ──────────────────────────────────────────────────────────────────────────────
# 7. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing Losses ISO 26262 ASIL-B self-check...")
    criterion = CombinedLaneSegmentationLoss(num_classes=4, weight_ohem=0.2, weight_lovasz=0.2)

    # Synthetic batch: B=2, C=4, H=360, W=640
    dummy_logits = torch.randn(2, 4, 360, 640, requires_grad=True)
    dummy_targets = torch.randint(0, 4, (2, 360, 640), dtype=torch.long)

    loss, breakdown = criterion(dummy_logits, dummy_targets)
    assert not torch.isnan(loss), "Loss produced NaN."
    assert not torch.isinf(loss), "Loss produced Inf."
    assert loss.item() > 0.0

    # Backpropagation verification
    loss.backward()
    assert dummy_logits.grad is not None, "Gradient backpropagation failed."

    # Test individual OHEM
    ohem = OHEMCrossEntropyLoss()
    l_ohem = ohem(dummy_logits, dummy_targets)
    assert l_ohem.item() > 0.0

    # Test individual Lovasz
    lov = LovaszSoftmaxLoss()
    l_lov = lov(dummy_logits, dummy_targets)
    assert not torch.isnan(l_lov)

    # Test Structural Continuity
    cont = StructuralContinuityLoss()
    l_cont = cont(dummy_logits, dummy_targets)
    assert l_cont.item() >= 0.0

    print(f"Loss evaluation passed. Total Loss: {breakdown['loss_total']:.4f}")


