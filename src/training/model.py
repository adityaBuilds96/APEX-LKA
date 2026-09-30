"""
src/training/model.py
======================
LaneSegNet — Universal 4-Class Semantic Segmentation Neural Network.

Architecture
------------
Encoder : MobileNetV3-Small (ImageNet-pretrained via torchvision)
          Multi-scale intermediate feature maps:
            • P2  —  H/4   ×  W/4   (24 ch)
            • P3  —  H/8   ×  W/8   (40 ch)
            • P4  —  H/16  ×  W/16  (48 ch)
            • P5  —  H/32  ×  W/32  (96 ch)

Decoder : Feature Pyramid Network (FPN) with Dual Attention:
            • Lateral 1×1 projections to unified FPN feature space (256 ch)
            • Top-down progressive feature fusion
            • Squeeze-and-Excitation (SE) channel attention
            • Spatial Attention (7×7 conv) for road and lane focus

Heads   :
  1. Primary Segmentation Head:
     1×1 conv → 4-class logits (Background, Road, Left Lane, Right Lane)
  2. Auxiliary Boundary Head (training only, discarded at inference):
     1-channel boundary distance/edge logits to sharpen thin lane contours

Parameters: ~3.4 million (optimized for RTX 4060 training and Jetson Orin Nano deployment).
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


# ── Configuration Defaults ───────────────────────────────────────────────────
NUM_CLASSES: int = 4
FPN_CHANNELS: int = 224
SE_REDUCTION: int = 16


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Attention Modules
# ═══════════════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation attention.
    Recalibrates channel responses to emphasize thin lane markings over background.
    """

    def __init__(self, channels: int, reduction: int = SE_REDUCTION) -> None:
        super().__init__()
        mid = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    """
    Lightweight 7×7 Spatial Attention module.
    Focuses the network on drivable corridors and road boundaries.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FPN Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class LateralBlock(nn.Module):
    """
    Projects encoder feature map to FPN_CHANNELS and applies SE channel attention.
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


class FPNFusionBlock(nn.Module):
    """
    Fuses top-down upsampled features with lateral features.
    Applies standard 3×3 convolutions, SE channel attention, and spatial attention.
    """

    def __init__(self, channels: int = FPN_CHANNELS) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(channels)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.se(out)
        out = self.spatial(out)
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MobileNetV3-Small Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class MobileNetV3Encoder(nn.Module):
    """
    MobileNetV3-Small backbone exposing 4 pyramid scales (1/4, 1/8, 1/16, 1/32).
    """

    _TAPS = (2, 4, 8, 11)
    _CHANNELS = (24, 40, 48, 96)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features

        # Freeze early stages to preserve low-level edge filters
        for i, block in enumerate(self.features):
            if i < 3:
                for p in block.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs: List[torch.Tensor] = []
        tap_set = set(self._TAPS)
        for idx, block in enumerate(self.features):
            x = block(x)
            if idx in tap_set:
                outs.append(x)
        return outs  # [p2 (1/4), p3 (1/8), p4 (1/16), p5 (1/32)]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FPNDecoder
# ═══════════════════════════════════════════════════════════════════════════════

class FPNDecoder(nn.Module):
    """
    Top-down Feature Pyramid Network decoder with dual attention fusion.
    """

    _IN_CHANNELS = (24, 40, 48, 96)

    def __init__(self, fpn_ch: int = FPN_CHANNELS) -> None:
        super().__init__()
        self.laterals = nn.ModuleList([
            LateralBlock(c, fpn_ch) for c in self._IN_CHANNELS
        ])
        # Smooth and refine merged levels
        self.smooth_p5 = nn.Sequential(
            nn.Conv2d(fpn_ch, fpn_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_ch),
            nn.ReLU(inplace=True),
        )
        self.fusion_blocks = nn.ModuleList([
            FPNFusionBlock(fpn_ch) for _ in range(len(self._IN_CHANNELS) - 1)
        ])

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Returns finest fused feature map at 1/4 resolution (B, fpn_ch, H/4, W/4).
        """
        # Lateral 1x1 projections
        lat = [lat_blk(f) for lat_blk, f in zip(self.laterals, feats)]

        # Top-down feature aggregation
        top = self.smooth_p5(lat[-1])
        for i in range(len(lat) - 2, -1, -1):
            top = F.interpolate(top, size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
            top = top + lat[i]
            top = self.fusion_blocks[i](top)

        return top  # (B, FPN_CH, H/4, W/4)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. LaneSegNet Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class LaneSegNet(nn.Module):
    """
    Universal Lane & Path Segmentation Neural Network.

    Encoder : MobileNetV3-Small (pretrained)
    Decoder : FPN with SE Channel Attention and Spatial Attention
    Heads   :
      - Primary Head: 1x1 conv → num_classes (4 classes)
      - Auxiliary Head: boundary distance estimator (training only)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        fpn_channels: int = FPN_CHANNELS,
        with_aux_head: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.with_aux_head = with_aux_head

        self.encoder = MobileNetV3Encoder(pretrained=pretrained)
        self.decoder = FPNDecoder(fpn_ch=fpn_channels)

        # Primary segmentation head
        self.head = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(fpn_channels, num_classes, kernel_size=1),
        )

        # Auxiliary boundary distance head
        self.aux_head = nn.Sequential(
            nn.Conv2d(fpn_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self._init_heads()

    def _init_heads(self) -> None:
        """Kaiming uniform initialization on classification and auxiliary convs."""
        for head in (self.head, self.aux_head):
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Parameters
        ----------
        x : (B, 3, H, W)
        return_aux : bool
            If True, returns tuple (logits, aux_boundary_logits).
            Defaults to False for standard inference and export.

        Returns
        -------
        logits : (B, num_classes, H, W) float32 logits
        aux    : Optional (B, 1, H, W) float32 boundary logits
        """
        feats = self.encoder(x)
        fused = self.decoder(feats)           # (B, FPN_CH, H/4, W/4)

        # Primary segmentation head
        logits_s4 = self.head(fused)
        logits = F.interpolate(
            logits_s4, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        if return_aux and self.with_aux_head:
            aux_s4 = self.aux_head(fused)
            aux_out = F.interpolate(
                aux_s4, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
            return logits, aux_out

        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference and return class-index segmentation mask (B, H, W)."""
        self.eval()
        logits = self.forward(x, return_aux=False)
        return logits.argmax(dim=1)

    def param_count(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        """Total parameters including frozen weights."""
        return sum(p.numel() for p in self.parameters())
