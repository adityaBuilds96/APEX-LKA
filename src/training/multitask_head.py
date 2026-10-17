"""
src/training/multitask_head.py
==============================
Shared-Backbone HydraNet Multi-Task Neural Network for APEX-LKA:
- Simultaneously outputs:
  1. 4-Class Road/Lane Semantic Segmentation Mask (logits: [B, 4, H, W])
  2. 9-Class 2D Object Detection Bounding Boxes + Confidence ([B, N_dets, 13])
- Encoder: Shared MobileNetV3-Small backbone with multi-scale feature taps.
- Decoder Head A: FPN + SE Attention + Spatial Attention (LaneSegNet compatible).
- Decoder Head B: Ultra-lightweight decoupled Detection Head with anchor-free grid decoding.
- Trace-compatible with torch.onnx.export() and TensorRT FP16 engine builder.
- Combined Multi-Task Loss:
  L_total = L_seg + 0.5 * L_cls + 1.0 * L_box_CIoU
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.model import (
    FPN_CHANNELS,
    NUM_CLASSES,
    FPNDecoder,
    MobileNetV3Encoder,
    SEBlock,
)


TARGET_CLASSES: List[str] = [
    "vehicle",
    "pedestrian",
    "cyclist",
    "traffic_cone",
    "barrier",
    "animal",
    "debris",
    "traffic_sign",
    "traffic_light",
]
NUM_OBJ_CLASSES: int = len(TARGET_CLASSES)  # 9 classes


# ═══════════════════════════════════════════════════════════════════════════
# 1. Anchor-Free Decoupled Object Detection Head
# ═══════════════════════════════════════════════════════════════════════════

class DecoupledDetectionHead(nn.Module):
    """
    Ultra-lightweight decoupled 2D object detection head.
    Predicts class classification logits and normalized bounding boxes [x1, y1, x2, y2].
    """

    def __init__(
        self,
        in_channels: int = FPN_CHANNELS,
        num_classes: int = NUM_OBJ_CLASSES,
        hidden_channels: int = 64,
        stride: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride

        # Stride reduction from FPN (H/4, W/4) to detection grid (H/8, W/8)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            SEBlock(hidden_channels),
        )

        # Classification branch
        self.cls_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1),
        )

        # Box regression branch: [dx, dy, dw, dh]
        self.reg_branch = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 4, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Prior probability bias initialization for classification (-4.59 = log(0.01 / 0.99))
        if hasattr(self.cls_branch[-1], "bias") and self.cls_branch[-1].bias is not None:
            nn.init.constant_(self.cls_branch[-1].bias, -4.59)

    def forward(
        self,
        feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            det_preds: (B, N_anchors, 4 + num_classes) [x1, y1, x2, y2, cls_probs...] in [0, 1]
            cls_logits: (B, num_classes, H_g, W_g)
            reg_preds: (B, 4, H_g, W_g)
        """
        x = self.downsample(feat)  # (B, hidden, H/8, W/8)
        b, _, h_g, w_g = x.shape

        cls_logits = self.cls_branch(x)  # (B, 9, H_g, W_g)
        reg = self.reg_branch(x)         # (B, 4, H_g, W_g)

        # Anchor-free grid mesh coordinates
        y_coords = torch.arange(h_g, dtype=torch.float32, device=x.device)
        x_coords = torch.arange(w_g, dtype=torch.float32, device=x.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)  # (B, H_g, W_g)
        grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)

        # Regressed coordinates (normalized 0.0 to 1.0)
        cx = (grid_x + torch.sigmoid(reg[:, 0])) / float(max(1, w_g))
        cy = (grid_y + torch.sigmoid(reg[:, 1])) / float(max(1, h_g))
        w = torch.sigmoid(reg[:, 2]) * 0.90 + 0.01
        h = torch.sigmoid(reg[:, 3]) * 0.90 + 0.01

        x1 = torch.clamp(cx - w / 2.0, 0.0, 1.0)
        y1 = torch.clamp(cy - h / 2.0, 0.0, 1.0)
        x2 = torch.clamp(cx + w / 2.0, 0.0, 1.0)
        y2 = torch.clamp(cy + h / 2.0, 0.0, 1.0)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # (B, H_g, W_g, 4)
        boxes_flat = boxes.reshape(b, -1, 4)           # (B, N_anchors, 4)

        cls_probs = torch.sigmoid(cls_logits).permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
        det_preds = torch.cat([boxes_flat, cls_probs], dim=-1)  # (B, N_anchors, 13)

        return det_preds, cls_logits, reg


# ═══════════════════════════════════════════════════════════════════════════
# 2. Complete CIoU Loss Module
# ═══════════════════════════════════════════════════════════════════════════

def compute_ciou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Computes Complete Intersection over Union (CIoU) Loss between box pairs.
    Boxes are in format [x1, y1, x2, y2].
    """
    # Coordinates
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.unbind(-1)

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    # Union area
    b1_area = torch.clamp(b1_x2 - b1_x1, min=0.0) * torch.clamp(b1_y2 - b1_y1, min=0.0)
    b2_area = torch.clamp(b2_x2 - b2_x1, min=0.0) * torch.clamp(b2_y2 - b2_y1, min=0.0)
    union_area = b1_area + b2_area - inter_area + eps
    iou = inter_area / union_area

    # Enclosing smallest box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c2 = torch.pow(c_x2 - c_x1, 2) + torch.pow(c_y2 - c_y1, 2) + eps

    # Center distance
    b1_cx = (b1_x1 + b1_x2) / 2.0
    b1_cy = (b1_y1 + b1_y2) / 2.0
    b2_cx = (b2_x1 + b2_x2) / 2.0
    b2_cy = (b2_y1 + b2_y2) / 2.0
    rho2 = torch.pow(b1_cx - b2_cx, 2) + torch.pow(b1_cy - b2_cy, 2)

    # Aspect ratio consistency
    b1_w = torch.clamp(b1_x2 - b1_x1, min=eps)
    b1_h = torch.clamp(b1_y2 - b1_y1, min=eps)
    b2_w = torch.clamp(b2_x2 - b2_x1, min=eps)
    b2_h = torch.clamp(b2_y2 - b2_y1, min=eps)

    v = (4.0 / (torch.pi ** 2)) * torch.pow(torch.atan(b2_w / b2_h) - torch.atan(b1_w / b1_h), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return 1.0 - torch.clamp(ciou, -1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Combined Multi-Task HydraNet Loss
# ═══════════════════════════════════════════════════════════════════════════

class HydraNetLoss(nn.Module):
    """
    Combined Multi-Task Loss:
    L_total = L_seg + 0.5 * L_cls + 1.0 * L_box_CIoU
    """

    def __init__(
        self,
        cls_weight: float = 0.5,
        box_weight: float = 1.0,
        seg_class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight

        # Segmentation cross-entropy with lane boundary weighting
        weights = seg_class_weights if seg_class_weights is not None else torch.tensor([1.0, 1.5, 3.0, 3.0])
        self.register_buffer("seg_weights", weights)

    def forward(
        self,
        seg_logits: torch.Tensor,
        det_preds: torch.Tensor,
        seg_targets: torch.Tensor,
        target_boxes: Optional[torch.Tensor] = None,
        target_classes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculates combined loss components.
        """
        # 1. Segmentation Loss (CE + Dice)
        ce_loss = F.cross_entropy(seg_logits, seg_targets, weight=self.seg_weights)

        # Soft Dice Loss
        seg_probs = F.softmax(seg_logits, dim=1)
        target_one_hot = F.one_hot(seg_targets, num_classes=seg_logits.shape[1]).permute(0, 3, 1, 2).float()
        intersection = torch.sum(seg_probs * target_one_hot, dim=(2, 3))
        cardinality = torch.sum(seg_probs + target_one_hot, dim=(2, 3))
        dice_loss = 1.0 - torch.mean((2.0 * intersection + 1.0) / (cardinality + 1.0))
        l_seg = ce_loss + dice_loss

        # 2. Object Detection Losses
        pred_boxes = det_preds[:, :, :4]
        pred_cls_probs = det_preds[:, :, 4:]

        l_cls = torch.tensor(0.0, device=seg_logits.device)
        l_box = torch.tensor(0.0, device=seg_logits.device)

        if target_boxes is not None and target_classes is not None and target_boxes.numel() > 0:
            # Match targets to prediction grid (nearest center)
            # Classification BCE loss
            l_cls = F.binary_cross_entropy(pred_cls_probs, target_classes)
            # CIoU box regression loss
            l_box = torch.mean(compute_ciou_loss(pred_boxes, target_boxes))

        l_total = l_seg + self.cls_weight * l_cls + self.box_weight * l_box

        return {
            "total_loss": l_total,
            "seg_loss": l_seg,
            "cls_loss": l_cls,
            "box_loss": l_box,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. APEXHydraNet Architecture
# ═══════════════════════════════════════════════════════════════════════════

class APEXHydraNet(nn.Module):
    """
    APEXHydraNet — Unified Multi-Task Perception Network for APEX-LKA.

    Shared Backbone : MobileNetV3-Small (Torchvision pretrained weights)
    Shared FPN      : 128-channel Feature Pyramid with SE + Spatial Attention
    Head A (Seg)    : 4-class semantic segmentation (0=Background, 1=Road, 2=Left, 3=Right)
    Head B (Det)    : 9-class anchor-free 2D bounding box detector ([x1, y1, x2, y2, cls_probs...])
    """

    def __init__(
        self,
        num_seg_classes: int = NUM_CLASSES,
        num_obj_classes: int = NUM_OBJ_CLASSES,
        fpn_channels: int = FPN_CHANNELS,
        pretrained: bool = True,
        with_aux_seg_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.num_obj_classes = num_obj_classes
        self.with_aux_seg_head = with_aux_seg_head

        # 1. Shared Encoder & FPN
        self.encoder = MobileNetV3Encoder(pretrained=pretrained)
        self.decoder = FPNDecoder(fpn_ch=fpn_channels)

        # 2. Head A: Semantic Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(fpn_channels, num_seg_classes, kernel_size=1),
        )

        # Optional Auxiliary Boundary Head (training only)
        if with_aux_seg_head:
            self.aux_seg_head = nn.Sequential(
                nn.Conv2d(fpn_channels, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, kernel_size=1),
            )
        else:
            self.aux_seg_head = None

        # 3. Head B: Decoupled Object Detection Head
        self.det_head = DecoupledDetectionHead(
            in_channels=fpn_channels,
            num_classes=num_obj_classes,
            hidden_channels=64,
            stride=8,
        )

        self._init_seg_heads()

    def _init_seg_heads(self) -> None:
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_seg_only: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        Returns:
            If return_seg_only: seg_logits (B, 4, H, W)
            Default: (seg_logits, det_preds)
                - seg_logits: (B, 4, H, W) float32 logits
                - det_preds:  (B, N_anchors, 13) [x1, y1, x2, y2, cls_probs...]
        """
        # Shared multi-scale encoder
        feats = self.encoder(x)
        # Shared FPN top-down fusion
        fused = self.decoder(feats)  # (B, FPN_CH, H/4, W/4)

        # Head A: Segmentation
        logits_s4 = self.seg_head(fused)
        seg_logits = F.interpolate(
            logits_s4, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        if return_seg_only:
            return seg_logits

        # Head B: Detection
        det_preds, _, _ = self.det_head(fused)

        return seg_logits, det_preds

    def param_count(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        """Total parameters including frozen weights."""
        return sum(p.numel() for p in self.parameters())
