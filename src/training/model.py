"""
src/training/model.py
======================
LaneSegNet — Lightweight 4-class lane segmentation network.

Architecture
------------
Encoder : MobileNetV3-Small (ImageNet-pretrained)
          Exposes intermediate feature maps at 4 scales:
            • P2  —  H/4   ×  W/4   (24 ch)
            • P3  —  H/8   ×  W/8   (40 ch)
            • P4  —  H/16  ×  W/16  (96 ch)
            • P5  —  H/32  ×  W/32  (576 ch)

Decoder : Feature Pyramid Network (FPN)
          Each level is processed through a Squeeze-and-Excitation (SE)
          block before being upsampled and fused with the next coarser level.
          SE attention focuses the network on thin lane-marking features.

Head    : 1×1 conv → 4-class logit map at H/4, upsampled to H×W.

Output
------
Shape  : (B, 4, H, W)   float32 logits (pre-softmax)
Classes: 0 = background
         1 = road
         2 = left_lane
         3 = right_lane
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


# ── Configuration ─────────────────────────────────────────────────────────────
NUM_CLASSES: int = 4
FPN_CHANNELS: int = 128   # Unified FPN feature width
SE_REDUCTION:  int = 8    # Squeeze-and-Excitation reduction ratio


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Squeeze-and-Excitation block
# ═══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation attention.

    Recalibrates channel feature responses to emphasise thin lane markings
    over the wide background region.

    Parameters
    ----------
    channels  : int  — number of input channels
    reduction : int  — bottleneck reduction ratio (default 8)
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, 1, 1)
            nn.Flatten(),                     # (B, C)
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FPN Lateral + output convolutions
# ═══════════════════════════════════════════════════════════════════════════════

class LateralBlock(nn.Module):
    """
    Projects an encoder feature map to FPN_CHANNELS and applies SE attention.

    Used at each FPN level to reduce channel count before top-down fusion.
    """

    def __init__(self, in_channels: int, out_channels: int = FPN_CHANNELS) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.proj(x))


class FPNOutputBlock(nn.Module):
    """
    Smoothing conv after top-down FPN merge (3×3 depthwise-separable).
    """

    def __init__(self, channels: int = FPN_CHANNELS) -> None:
        super().__init__()
        self.dw = nn.Sequential(
            # Depthwise
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            # Pointwise
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dw(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MobileNetV3-Small encoder — feature extractor
# ═══════════════════════════════════════════════════════════════════════════════

class MobileNetV3Encoder(nn.Module):
    """
    Wraps torchvision's MobileNetV3-Small and exposes intermediate feature maps
    at four pyramid scales.

    Probed tap-points on a 360×640 input (block → out shape):
        block  0: (B, 16,  180, 320)   stride /2
        block  2: (B, 24,   45,  80)   stride /4   → P2
        block  4: (B, 40,   23,  40)   stride /8   → P3
        block  8: (B, 48,   23,  40)   stride /16  → P4
        block 11: (B, 96,   12,  20)   stride /32  → P5
        block 12: (B, 576,  12,  20)   (expansion conv, NOT tapped)

    NOTE: Block 12 is a 576-ch pointwise expansion at the same spatial
    resolution as block 11. We tap block 11 (96 ch) to keep things lightweight.
    """

    # Layer tap indices in MobileNetV3-Small features sequential
    _TAPS = (2, 4, 8, 11)
    # Corresponding out-channels — verified by probe
    _CHANNELS = (24, 40, 48, 96)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        # Keep only the features sequential (drop classifier & avgpool)
        self.features = backbone.features  # nn.Sequential of 13 InvertedResiduals
        # Freeze first 3 encoder stages to preserve pretrained low-level filters
        for i, block in enumerate(self.features):
            if i < 3:
                for p in block.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns
        -------
        [p2, p3, p4, p5] — feature maps at 1/4, 1/8, 1/16, 1/32 input resolution.
        """
        outs: List[torch.Tensor] = []
        tap_set = set(self._TAPS)
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in tap_set:
                outs.append(x)
        return outs  # len == 4


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FPN Decoder
# ═══════════════════════════════════════════════════════════════════════════════

class FPNDecoder(nn.Module):
    """
    Top-down FPN decoder.

    Fuses [p2, p3, p4, p5] from the encoder into a single feature pyramid
    by upsampling and adding the next level, then applying a smoothing conv.
    """

    # Input channel sizes that match MobileNetV3-Small tap points (probed)
    _IN_CHANNELS = (24, 40, 48, 96)

    def __init__(self, fpn_ch: int = FPN_CHANNELS) -> None:
        super().__init__()
        # One lateral per tap level
        self.laterals = nn.ModuleList(
            [LateralBlock(c, fpn_ch) for c in self._IN_CHANNELS]
        )
        # Smoothing after top-down merge (for p2, p3, p4 — p5 needs none)
        self.outputs = nn.ModuleList(
            [FPNOutputBlock(fpn_ch) for _ in range(len(self._IN_CHANNELS) - 1)]
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        feats : [p2, p3, p4, p5]   from encoder, coarsest last

        Returns
        -------
        fused : (B, FPN_CHANNELS, H/4, W/4) — finest FPN level
        """
        # Lateral projections
        lat = [lat_blk(f) for lat_blk, f in zip(self.laterals, feats)]

        # Top-down: start from coarsest (p5) and upsample into finer levels
        top = lat[-1]   # p5 (coarsest)
        merged: List[torch.Tensor] = []
        for i in range(len(lat) - 2, -1, -1):
            top = F.interpolate(top, size=lat[i].shape[-2:], mode="nearest")
            top = top + lat[i]
            merged.insert(0, top)

        # merged[0] corresponds to p2 (finest)
        # Apply smoothing to all but the very last (p5 already smoothed implicitly)
        out = self.outputs[0](merged[0])
        return out  # (B, FPN_CH, H/4, W/4)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LaneSegNet — the complete model
# ═══════════════════════════════════════════════════════════════════════════════

class LaneSegNet(nn.Module):
    """
    APEX LKA — LaneSegNet segmentation network.

    Encoder : MobileNetV3-Small (pretrained by default)
    Decoder : FPN + SE attention
    Head    : 1×1 conv → num_classes logits

    Input
    -----
    x : (B, 3, H, W)   float32   ImageNet-normalised RGB

    Output
    ------
    logits : (B, num_classes, H, W)   float32   (NOT softmax'd)

    Usage
    -----
    >>> model = LaneSegNet()
    >>> logits = model(torch.randn(2, 3, 360, 640))
    >>> logits.shape
    torch.Size([2, 4, 360, 640])
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        fpn_channels: int = FPN_CHANNELS,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.encoder = MobileNetV3Encoder(pretrained=pretrained)
        self.decoder = FPNDecoder(fpn_ch=fpn_channels)
        self.head = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )
        self._init_head()

    def _init_head(self) -> None:
        """Kaiming-uniform init on the classification head conv."""
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        (B, num_classes, H, W) — logits at full input resolution
        """
        feats = self.encoder(x)              # [p2, p3, p4, p5]
        fpn_out = self.decoder(feats)        # (B, FPN_CH, H/4, W/4)
        logits_s4 = self.head(fpn_out)       # (B, C, H/4, W/4)
        # Upsample to full input resolution
        logits = F.interpolate(
            logits_s4, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return logits

    # ── Convenience helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference and return class-index map.

        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        (B, H, W)   int64 class indices
        """
        self.eval()
        logits = self.forward(x)
        return logits.argmax(dim=1)

    def param_count(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
