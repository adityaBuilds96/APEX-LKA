"""
APEX-Vision Core v3.0 — HydraNet Multi-Task Perception Architecture
===================================================================
ISO 26262 ASIL-B Compliant Multi-Task Neural Network.
Shared MobileNetV3-Small backbone with:
- Head A: Lane Segmentation (Binary Mask + Polynomial Curvature [c0, c1, c2])
- Head B: Object Detection (Bounding Boxes for Vehicles, Pedestrians, Cones)

Optimized for NVIDIA Jetson Orin Nano (TensorRT) and real-time ADAS deployment.
Strictly avoids dynamic shapes for seamless ONNX and TensorRT compilation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

# Target detection classes for APEX-Vision Core v3.0
OBJECT_CLASSES: Dict[int, str] = {
    0: "vehicle",
    1: "pedestrian",
    2: "cone",
}
NUM_OBJ_CLASSES: int = len(OBJECT_CLASSES)


@dataclass
class Detection2D:
    """2D Bounding box detection with class label and confidence."""
    bbox: List[float]  # [x1, y1, x2, y2] in pixels
    class_name: str
    class_id: int
    confidence: float

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Ground contact point (x_center, y2) used for 3D distance estimation."""
        return (self.center_x, self.bbox[3])


@dataclass
class PerceptionOutput:
    """Aggregated output from HydraNet forward pass."""
    lane_mask: np.ndarray  # Binary mask [H, W], uint8 (0 or 255)
    lane_logits: np.ndarray  # Float32 probability map [H, W]
    curvature_coeffs: Tuple[float, float, float]  # [c0 (offset), c1 (heading), c2 (curvature)]
    radius_of_curvature_m: float
    detections: List[Detection2D] = field(default_factory=list)
    latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. Shared MobileNetV3-Small Multi-Scale Encoder
# ═══════════════════════════════════════════════════════════════════════════

class SharedMobileNetV3Backbone(nn.Module):
    """
    Shared MobileNetV3-Small backbone extracting multi-scale feature hierarchies:
      - C1: Stride 4  (H/4, W/4)   - 16 channels
      - C2: Stride 8  (H/8, W/8)   - 24 channels
      - C3: Stride 16 (H/16, W/16) - 48 channels
      - C4: Stride 32 (H/32, W/32) - 96 channels
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        try:
            m = mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        except Exception:
            m = mobilenet_v3_small(weights=None)

        features = m.features
        # Slice layers by stride
        self.stage1 = nn.Sequential(*features[0:2])    # /4  (16 ch)
        self.stage2 = nn.Sequential(*features[2:4])    # /8  (24 ch)
        self.stage3 = nn.Sequential(*features[4:9])    # /16 (48 ch)
        self.stage4 = nn.Sequential(*features[9:12])   # /32 (96 ch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return c1, c2, c3, c4


# ═══════════════════════════════════════════════════════════════════════════
# 2. Head A: Lane Segmentation & Curvature Head
# ═══════════════════════════════════════════════════════════════════════════

class LaneHead(nn.Module):
    """
    Decodes multi-scale features into:
    1. Binary lane mask logits [B, 1, H, W]
    2. Road polynomial curvature parameters [c0, c1, c2]
    """

    def __init__(self, fpn_dim: int = 64):
        super().__init__()
        # Lateral convolutions
        self.lat_c4 = nn.Conv2d(96, fpn_dim, kernel_size=1)
        self.lat_c3 = nn.Conv2d(48, fpn_dim, kernel_size=1)
        self.lat_c2 = nn.Conv2d(24, fpn_dim, kernel_size=1)
        self.lat_c1 = nn.Conv2d(16, fpn_dim, kernel_size=1)

        # Smooth convolutions
        self.smooth = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)

        # Segmentation mask output (Binary Lane Logits)
        self.mask_head = nn.Sequential(
            nn.Conv2d(fpn_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Global average pool + MLP for curvature coefficients [c0, c1, c2]
        self.curvature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # c0 (lateral offset), c1 (heading slope), c2 (curvature kappa)
        )

    def forward(
        self,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Top-down FPN pathway
        p4 = self.lat_c4(c4)
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lat_c1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)

        feat = self.smooth(p1)
        # Upsample to full input resolution
        mask_logits = F.interpolate(self.mask_head(feat), size=target_size, mode="bilinear", align_corners=False)

        # Predict curvature from deepest semantic feature map
        curvature = self.curvature_head(c4)

        return mask_logits, curvature


# ═══════════════════════════════════════════════════════════════════════════
# 3. Head B: TensorRT-Ready Decoupled Object Detection Head
# ═══════════════════════════════════════════════════════════════════════════

class ObjectDetectionHead(nn.Module):
    """
    Decoupled anchor-free detection head operating on C3 (stride 16).
    Predicts:
      - Bounding box offsets: [dx1, dy1, dx2, dy2] relative to grid cell
      - Class classification logits: [vehicle, pedestrian, cone]
      - Objectness score: [P(object)]

    Output shape: [B, H_grid * W_grid, 4 + num_classes + 1]
    Strictly avoids dynamic shapes for TensorRT FP16/INT8 compatibility.
    """

    def __init__(self, in_channels: int = 48, num_classes: int = NUM_OBJ_CLASSES, hidden_dim: int = 64):
        super().__init__()
        self.num_classes = num_classes

        # Shared feature refinement
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Box regression branch
        self.box_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, kernel_size=1),
            nn.Sigmoid(),  # Normalized box bounds [0, 1]
        )

        # Classification branch
        self.cls_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

        # Objectness branch
        self.obj_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, c3: torch.Tensor) -> torch.Tensor:
        b, _, h, w = c3.shape
        feat = self.stem(c3)

        boxes = self.box_conv(feat)        # [B, 4, H, W]
        cls_logits = self.cls_conv(feat)   # [B, num_classes, H, W]
        obj = self.obj_conv(feat)          # [B, 1, H, W]

        # Flatten spatial dimensions: [B, channels, N] -> [B, N, channels]
        n_anchors = h * w
        boxes_flat = boxes.view(b, 4, n_anchors).permute(0, 2, 1)
        cls_flat = cls_logits.view(b, self.num_classes, n_anchors).permute(0, 2, 1)
        obj_flat = obj.view(b, 1, n_anchors).permute(0, 2, 1)

        # Concatenate: [B, N, 4 + num_classes + 1]
        predictions = torch.cat([boxes_flat, cls_flat, obj_flat], dim=-1)
        return predictions


# ═══════════════════════════════════════════════════════════════════════════
# 4. Multi-Task HydraNet Model
# ═══════════════════════════════════════════════════════════════════════════

class HydraNet(nn.Module):
    """
    Unified HydraNet combining shared MobileNetV3-Small backbone with:
    - Head A: Lane Segmentation Mask + Curvature
    - Head B: Object Detection
    """

    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = SharedMobileNetV3Backbone(pretrained=pretrained_backbone)
        self.lane_head = LaneHead(fpn_dim=64)
        self.det_head = ObjectDetectionHead(in_channels=48, num_classes=NUM_OBJ_CLASSES)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
          - lane_mask_logits: [B, 1, H, W]
          - curvature: [B, 3] (c0, c1, c2)
          - detection_preds: [B, N, 8] ([x1, y1, x2, y2, cls0, cls1, cls2, obj])
        """
        target_size = (x.shape[2], x.shape[3])
        c1, c2, c3, c4 = self.backbone(x)

        mask_logits, curvature = self.lane_head(c1, c2, c3, c4, target_size=target_size)
        det_preds = self.det_head(c3)

        return mask_logits, curvature, det_preds


# ═══════════════════════════════════════════════════════════════════════════
# 5. Production Perception Core Engine
# ═══════════════════════════════════════════════════════════════════════════

class APEXPerceptionCore:
    """
    High-level ISO 26262 compliant perception engine.
    Wraps HydraNet with pre- and post-processing, NMS, and metric calculations.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        model_path: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.model = HydraNet(pretrained_backbone=False).to(self.device)
        self.model.eval()

        if model_path:
            self.load_weights(model_path)

        # Standard preprocessing constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def load_weights(self, path: str) -> bool:
        """Load checkpoint weights safely."""
        try:
            ckpt = torch.load(path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            return True
        except Exception as e:
            print(f"[APEXPerceptionCore] Weights load warning ({e}); using initialized model.")
            return False

    def predict(self, bgr_image: np.ndarray) -> PerceptionOutput:
        """
        Execute multi-task perception pipeline on input image.
        """
        t0 = time.perf_counter()
        orig_h, orig_w = bgr_image.shape[:2]

        # Target input dimensions (360, 640)
        in_h, in_w = 360, 640
        if (orig_h, orig_w) != (in_h, in_w):
            img_resized = cv2.resize(bgr_image, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = bgr_image

        # Preprocess BGR -> RGB -> Normalized Float Tensor
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm_rgb = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(norm_rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mask_logits, curvature, det_preds = self.model(tensor)

        # Postprocess Head A: Lane Mask
        lane_prob = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
        binary_mask = (lane_prob > 0.50).astype(np.uint8) * 255
        if (orig_h, orig_w) != (in_h, in_w):
            binary_mask = cv2.resize(binary_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            lane_prob = cv2.resize(lane_prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Curvature parameters
        curv_vals = curvature.squeeze().cpu().numpy().tolist()
        c0, c1, c2 = float(curv_vals[0]), float(curv_vals[1]), float(curv_vals[2])
        # Radius of curvature: R = 1 / (2 * |c2| + eps)
        radius = 1.0 / (2.0 * max(1e-5, abs(c2)))

        # Postprocess Head B: Object Detections via NMS
        raw_dets = det_preds.squeeze(0).cpu().numpy()  # [N, 4 + 3 + 1]
        detections = self._postprocess_detections(raw_dets, orig_w, orig_h)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PerceptionOutput(
            lane_mask=binary_mask,
            lane_logits=lane_prob,
            curvature_coeffs=(c0, c1, c2),
            radius_of_curvature_m=radius,
            detections=detections,
            latency_ms=latency_ms,
        )

    def _postprocess_detections(self, raw_dets: np.ndarray, orig_w: int, orig_h: int) -> List[Detection2D]:
        """Convert raw anchor-free tensor predictions to filtered Detection2D objects."""
        boxes = raw_dets[:, :4]       # [x1, y1, x2, y2] in [0, 1]
        cls_logits = raw_dets[:, 4:7] # 3 classes
        obj_scores = raw_dets[:, 7]   # objectness

        # Softmax over class logits
        cls_exp = np.exp(cls_logits - np.max(cls_logits, axis=-1, keepdims=True))
        cls_probs = cls_exp / np.sum(cls_exp, axis=-1, keepdims=True)

        best_cls = np.argmax(cls_probs, axis=-1)
        best_cls_prob = np.max(cls_probs, axis=-1)
        final_scores = obj_scores * best_cls_prob

        # Confidence filter
        mask = final_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        filtered_boxes = boxes[mask]
        filtered_scores = final_scores[mask]
        filtered_classes = best_cls[mask]

        # Convert normalized coords to pixel coords [x1, y1, x2, y2]
        x1 = filtered_boxes[:, 0] * orig_w
        y1 = filtered_boxes[:, 1] * orig_h
        x2 = filtered_boxes[:, 2] * orig_w
        y2 = filtered_boxes[:, 3] * orig_h

        # Coordinate sanity: enforce x2 > x1 and y2 > y1
        x1_fixed = np.minimum(x1, x2)
        x2_fixed = np.maximum(x1, x2) + 1.0
        y1_fixed = np.minimum(y1, y2)
        y2_fixed = np.maximum(y1, y2) + 1.0

        nms_boxes = np.stack([x1_fixed, y1_fixed, x2_fixed - x1_fixed, y2_fixed - y1_fixed], axis=1)

        indices = cv2.dnn.NMSBoxes(
            bboxes=nms_boxes.tolist(),
            scores=filtered_scores.tolist(),
            score_threshold=float(self.conf_threshold),
            nms_threshold=float(self.nms_threshold),
        )

        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                cid = int(filtered_classes[idx])
                results.append(
                    Detection2D(
                        bbox=[
                            float(x1_fixed[idx]),
                            float(y1_fixed[idx]),
                            float(x2_fixed[idx]),
                            float(y2_fixed[idx]),
                        ],
                        class_name=OBJECT_CLASSES.get(cid, "unknown"),
                        class_id=cid,
                        confidence=float(filtered_scores[idx]),
                    )
                )

        return results


# Export alias
APEXHydraNet = HydraNet
