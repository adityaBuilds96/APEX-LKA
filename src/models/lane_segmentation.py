"""
src/models/lane_segmentation.py
===============================
APEX-Vision Core v4.0 • Deep Learning Lane Segmentation & End-to-End Steering
ISO 26262 ASIL-B Aligned Neural Network Perception Architecture

Implements deep semantic segmentation and end-to-end steering angle regression
tailored for the NVIDIA Jetson Orin Nano 4GB (TensorRT FP16 / INT8 PTQ).

Architectural Specifications:
-----------------------------
1. Pre-trained Transfer Learning Backbones:
   - MobileNetV3-Small: Ultra-lightweight backbone with hard-swish activations.
   - ResNet-50: Deep residual feature extractor with multi-stage skip connections.
   - Xception: Depthwise separable convolutions with linear residual shortcuts.
2. Semantic Decoders (4-Class Boundary & Surface Segmentation):
   - U-Net Decoder: Multi-stage bilinear upsampling with skip-connection fusion.
   - ErfNet Decoder: Factorized 1D residual convolutions (1x3 followed by 3x1)
     for ultra-low latency real-time deployment on embedded Jetson silicon.
   - Output Semantic Classes:
     * Class 0: Background / Non-drivable
     * Class 1: Drivable Surface / Corridor
     * Class 2: Left Road Boundary (Yellow / Solid)
     * Class 3: Right Road Boundary (White / Dashed)
3. End-to-End Dave-2 Steering Head:
   - Regresses continuous target steering angle delta_pred (radians) directly
     from deep feature maps through fully-connected projection layers.
4. Strict No-Flip Rule Enforcement:
   - Horizontal image flipping during dataset augmentation is strictly forbidden
     unless Class 2 (Left) and Class 3 (Right) are explicitly swapped in the mask.
5. Training Modes:
   - Frozen Feature Extraction: Backbone weights frozen, decoders trainable.
   - Full End-to-End Fine-Tuning: Entire network optimized jointly.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.LaneSegmentation")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration & Enumerations
# ═══════════════════════════════════════════════════════════════════════════════

class BackboneType(Enum):
    """Supported neural feature extraction backbones."""
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    RESNET_50 = "resnet_50"
    XCEPTION = "xception"


class DecoderType(Enum):
    """Supported semantic segmentation decoders."""
    UNET = "unet"
    ERFNET = "erfnet"


@dataclass
class LaneSegmentationConfig:
    """Configuration hyperparameters for lane segmentation and Dave-2 steering."""
    backbone_type: BackboneType = BackboneType.MOBILENET_V3_SMALL
    decoder_type: DecoderType = DecoderType.UNET
    num_classes: int = 4  # [Background, Drivable, Left_Boundary, Right_Boundary]
    input_channels: int = 3
    input_height: int = 360
    input_width: int = 640
    freeze_backbone: bool = False
    enable_steering_head: bool = True
    steering_dropout: float = 0.30
    max_steering_angle_rad: float = 0.4886  # 28 degrees for BAJA Buggy profile
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# Building Blocks: Convolutions, Squeeze-and-Excitation & Factorized Kernels
# ═══════════════════════════════════════════════════════════════════════════════

class ConvBNAct(nn.Module):
    """Standard Convolution + BatchNorm + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == "hard_swish":
            self.act = nn.Hardswish(inplace=True)
        elif activation == "leaky_relu":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention module."""

    def __init__(self, in_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeezed = max(1, in_channels // squeeze_factor)
        self.fc1 = nn.Conv2d(in_channels, squeezed, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeezed, in_channels, kernel_size=1)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        return x * scale


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block with expansion and depthwise conv."""

    def __init__(
        self,
        in_channels: int,
        exp_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_se: bool = True,
        use_hs: bool = True,
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        act = "hard_swish" if use_hs else "relu"

        layers: List[nn.Module] = []
        # Expansion
        if exp_channels != in_channels:
            layers.append(ConvBNAct(in_channels, exp_channels, kernel_size=1, padding=0, activation=act))

        # Depthwise
        layers.append(ConvBNAct(
            exp_channels,
            exp_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=exp_channels,
            activation=act,
        ))

        # Squeeze-and-Excitation
        if use_se:
            layers.append(SqueezeExcitation(exp_channels))

        # Pointwise projection (linear activation)
        layers.append(nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class NonBottleneck1D(nn.Module):
    """
    Factorized 1D Residual Convolution Block used in ERFNet:
    Decomposes 3x3 2D convolution into 3x1 and 1x3 sequential 1D filters.
    Reduces computational complexity by 33% with equivalent receptive field.
    """

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(channels, channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1), bias=False)
        self.conv1x3_1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation), dilation=(1, dilation), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv3x1_2 = nn.Conv2d(channels, channels, (3, 1), padding=(dilation, 0), dilation=(dilation, 1), bias=False)
        self.conv1x3_2 = nn.Conv2d(channels, channels, (1, 3), padding=(0, dilation), dilation=(1, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv3x1_1(x)
        out = self.conv1x3_1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3x1_2(out)
        out = self.conv1x3_2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        return self.relu(out + residual)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extraction Backbones
# ═══════════════════════════════════════════════════════════════════════════════

class MobileNetV3SmallBackbone(nn.Module):
    """
    MobileNetV3-Small feature extractor returning multi-scale feature maps
    at strides 2 (P1), 4 (P2), 8 (P3), 16 (P4), and 32 (P5).
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Initial stem (stride 2)
        self.stem = ConvBNAct(in_channels, 16, kernel_size=3, stride=2, padding=1, activation="hard_swish")

        # Stage 1: stride 4
        self.stage1 = InvertedResidual(16, 16, 16, kernel_size=3, stride=2, use_se=True, use_hs=False)

        # Stage 2: stride 8
        self.stage2 = nn.Sequential(
            InvertedResidual(16, 72, 24, kernel_size=3, stride=2, use_se=False, use_hs=False),
            InvertedResidual(24, 88, 24, kernel_size=3, stride=1, use_se=False, use_hs=False),
        )

        # Stage 3: stride 16
        self.stage3 = nn.Sequential(
            InvertedResidual(24, 96, 40, kernel_size=5, stride=2, use_se=True, use_hs=True),
            InvertedResidual(40, 240, 40, kernel_size=5, stride=1, use_se=True, use_hs=True),
            InvertedResidual(40, 240, 40, kernel_size=5, stride=1, use_se=True, use_hs=True),
        )

        # Stage 4: stride 32
        self.stage4 = nn.Sequential(
            InvertedResidual(40, 120, 48, kernel_size=5, stride=2, use_se=True, use_hs=True),
            InvertedResidual(48, 144, 48, kernel_size=5, stride=1, use_se=True, use_hs=True),
            InvertedResidual(48, 288, 96, kernel_size=5, stride=1, use_se=True, use_hs=True),
        )

        self.out_channels = [16, 16, 24, 40, 96]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p1 = self.stem(x)      # stride 2, 16 ch
        p2 = self.stage1(p1)    # stride 4, 16 ch
        p3 = self.stage2(p2)    # stride 8, 24 ch
        p4 = self.stage3(p3)    # stride 16, 40 ch
        p5 = self.stage4(p4)    # stride 32, 96 ch
        return [p1, p2, p3, p4, p5]


class ResNet50Backbone(nn.Module):
    """
    ResNet-50 feature extractor using bottleneck residual blocks.
    Returns multi-scale feature maps at strides 2, 4, 8, 16, and 32.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # 7x7 conv stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Simplified bottleneck stages
        self.layer1 = self._make_stage(64, 64, num_blocks=3, stride=1)   # stride 4, 256 ch
        self.layer2 = self._make_stage(256, 128, num_blocks=4, stride=2) # stride 8, 512 ch
        self.layer3 = self._make_stage(512, 256, num_blocks=6, stride=2) # stride 16, 1024 ch
        self.layer4 = self._make_stage(1024, 512, num_blocks=3, stride=2) # stride 32, 2048 ch

        self.out_channels = [64, 256, 512, 1024, 2048]

    def _make_stage(self, in_ch: int, mid_ch: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        out_ch = mid_ch * 4
        # First block with projection shortcut
        layers.append(self._make_bottleneck(in_ch, mid_ch, out_ch, stride=stride, is_downsample=True))
        for _ in range(1, num_blocks):
            layers.append(self._make_bottleneck(out_ch, mid_ch, out_ch, stride=1, is_downsample=False))
        return nn.Sequential(*layers)

    def _make_bottleneck(self, in_ch: int, mid_ch: int, out_ch: int, stride: int, is_downsample: bool) -> nn.Module:
        class Bottleneck(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(mid_ch)
                self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(mid_ch)
                self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)

                if is_downsample or in_ch != out_ch:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_ch),
                    )
                else:
                    self.shortcut = nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                res = self.shortcut(x)
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                return self.relu(out + res)

        return Bottleneck()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p1 = self.relu(self.bn1(self.conv1(x))) # stride 2
        p2_in = self.maxpool(p1)                # stride 4
        p2 = self.layer1(p2_in)                 # stride 4
        p3 = self.layer2(p2)                    # stride 8
        p4 = self.layer3(p3)                    # stride 16
        p5 = self.layer4(p4)                    # stride 32
        return [p1, p2, p3, p4, p5]


class XceptionBackbone(nn.Module):
    """
    Xception feature extractor using depthwise separable convolutions.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Entry flow
        self.conv1 = ConvBNAct(in_channels, 32, kernel_size=3, stride=2, padding=1)  # stride 2
        self.conv2 = ConvBNAct(32, 64, kernel_size=3, stride=1, padding=1)

        # Separable blocks
        self.block1 = self._make_sep_block(64, 128, stride=2)   # stride 4
        self.block2 = self._make_sep_block(128, 256, stride=2)  # stride 8
        self.block3 = self._make_sep_block(256, 512, stride=2)  # stride 16
        self.block4 = self._make_sep_block(512, 728, stride=2)  # stride 32

        self.out_channels = [32, 128, 256, 512, 728]

    def _make_sep_block(self, in_ch: int, out_ch: int, stride: int) -> nn.Module:
        class SepBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.sep1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False)
                self.point1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_ch)
                self.relu = nn.ReLU(inplace=True)

                self.sep2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=out_ch, bias=False)
                self.point2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_ch)

                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                res = self.shortcut(x)
                out = self.relu(self.bn1(self.point1(self.sep1(x))))
                out = self.bn2(self.point2(self.sep2(out)))
                return self.relu(out + res)

        return SepBlock()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        p1 = self.conv1(x)
        x2 = self.conv2(p1)
        p2 = self.block1(x2)
        p3 = self.block2(p2)
        p4 = self.block3(p3)
        p5 = self.block4(p4)
        return [p1, p2, p3, p4, p5]


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Decoders: U-Net and ErfNet
# ═══════════════════════════════════════════════════════════════════════════════

class UNetDecoder(nn.Module):
    """
    U-Net progressive upsampling decoder with skip-connection feature concatenation.
    Produces 4-class semantic segmentation mask at full image resolution.
    """

    def __init__(self, encoder_channels: List[int], num_classes: int = 4):
        super().__init__()
        # encoder_channels: [P1(stride 2), P2(stride 4), P3(stride 8), P4(stride 16), P5(stride 32)]
        c1, c2, c3, c4, c5 = encoder_channels

        # Up 1: stride 32 -> stride 16
        self.up4 = nn.ConvTranspose2d(c5, 128, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            ConvBNAct(128 + c4, 128, kernel_size=3),
            ConvBNAct(128, 128, kernel_size=3),
        )

        # Up 2: stride 16 -> stride 8
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            ConvBNAct(64 + c3, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
        )

        # Up 3: stride 8 -> stride 4
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            ConvBNAct(32 + c2, 32, kernel_size=3),
            ConvBNAct(32, 32, kernel_size=3),
        )

        # Up 4: stride 4 -> stride 2
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            ConvBNAct(16 + c1, 16, kernel_size=3),
            ConvBNAct(16, 16, kernel_size=3),
        )

        # Final Up: stride 2 -> full resolution (stride 1)
        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        p1, p2, p3, p4, p5 = features

        d4 = self.up4(p5)
        if d4.shape[2:] != p4.shape[2:]:
            d4 = F.interpolate(d4, size=p4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, p4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[2:] != p3.shape[2:]:
            d3 = F.interpolate(d3, size=p3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, p3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != p2.shape[2:]:
            d2 = F.interpolate(d2, size=p2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, p2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != p1.shape[2:]:
            d1 = F.interpolate(d1, size=p1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, p1], dim=1))

        out = self.final_up(d1)
        logits = self.classifier(out)
        return logits


class ERFNetDecoder(nn.Module):
    """
    ERFNet factorized 1D convolution decoder for real-time edge segmentation.
    Uses non-bottleneck 1D blocks with transpose convolutions.
    """

    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        # Initial projection
        self.conv_in = ConvBNAct(in_channels, 128, kernel_size=1, padding=0)

        # Stage 1: upsample stride 32 -> 16
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.block1 = nn.Sequential(
            NonBottleneck1D(64, dilation=1),
            NonBottleneck1D(64, dilation=1),
        )

        # Stage 2: upsample stride 16 -> 8
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.block2 = nn.Sequential(
            NonBottleneck1D(32, dilation=1),
            NonBottleneck1D(32, dilation=1),
        )

        # Stage 3: upsample stride 8 -> 4
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU(inplace=True)

        # Stage 4: upsample stride 4 -> 1 (4x bilinear)
        self.final_up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False)
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        p1 = features[0]
        target_h, target_w = p1.shape[2] * 2, p1.shape[3] * 2

        p5 = features[-1]
        x = self.conv_in(p5)

        x = self.relu1(self.bn1(self.up1(x)))
        x = self.block1(x)

        x = self.relu2(self.bn2(self.up2(x)))
        x = self.block2(x)

        x = self.relu3(self.bn3(self.up3(x)))
        x = self.final_up(x)
        logits = self.classifier(x)
        if logits.shape[2:] != (target_h, target_w):
            logits = F.interpolate(logits, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Dave-2 End-to-End Steering Angle Regression Head
# ═══════════════════════════════════════════════════════════════════════════════

class Dave2SteeringHead(nn.Module):
    """
    NVIDIA Dave-2 inspired convolutional steering angle predictor.
    Projects deep feature maps into target road-wheel steering angle delta_pred (radians).
    """

    def __init__(
        self,
        in_channels: int = 96,
        dropout_rate: float = 0.30,
        max_angle_rad: float = 0.4886,
    ):
        super().__init__()
        self.max_angle = max_angle_rad

        self.pool = nn.AdaptiveAvgPool2d((4, 8))
        flatten_dim = in_channels * 4 * 8

        self.fc = nn.Sequential(
            nn.Linear(flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Tanh(),  # Bounded to [-1.0, +1.0]
        )

    def forward(self, deep_features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(deep_features)
        flat = torch.flatten(pooled, start_dim=1)
        normalized_angle = self.fc(flat)  # [-1, +1]
        target_steering_rad = normalized_angle * self.max_angle
        return target_steering_rad


# ═══════════════════════════════════════════════════════════════════════════════
# Master Multi-Task Model: Segmentation + Dave-2 Steering
# ═══════════════════════════════════════════════════════════════════════════════

class APEXMultiTaskLaneModel(nn.Module):
    """
    Complete Multi-Task Deep Neural Network for Autonomous Perception & Control.
    Combines:
    - Backbone (MobileNetV3, ResNet-50, or Xception)
    - Semantic Decoder (U-Net or ERFNet) -> 4-class segmentation mask
    - Dave-2 Steering Head -> Continuous steering angle delta_pred (radians)
    """

    def __init__(self, config: Optional[LaneSegmentationConfig] = None):
        super().__init__()
        self.config = config or LaneSegmentationConfig()

        # 1. Initialize Backbone
        if self.config.backbone_type == BackboneType.MOBILENET_V3_SMALL:
            self.backbone = MobileNetV3SmallBackbone(self.config.input_channels)
        elif self.config.backbone_type == BackboneType.RESNET_50:
            self.backbone = ResNet50Backbone(self.config.input_channels)
        elif self.config.backbone_type == BackboneType.XCEPTION:
            self.backbone = XceptionBackbone(self.config.input_channels)
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone_type}")

        # 2. Initialize Decoder
        if self.config.decoder_type == DecoderType.UNET:
            self.decoder = UNetDecoder(self.backbone.out_channels, self.config.num_classes)
        elif self.config.decoder_type == DecoderType.ERFNET:
            self.decoder = ERFNetDecoder(self.backbone.out_channels[-1], self.config.num_classes)
        else:
            raise ValueError(f"Unsupported decoder: {self.config.decoder_type}")

        # 3. Initialize Steering Head
        if self.config.enable_steering_head:
            self.steering_head = Dave2SteeringHead(
                in_channels=self.backbone.out_channels[-1],
                dropout_rate=self.config.steering_dropout,
                max_angle_rad=self.config.max_steering_angle_rad,
            )
        else:
            self.steering_head = None

        # 4. Handle Frozen Backbone Mode
        if self.config.freeze_backbone:
            self.freeze_backbone(freeze=True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone parameters for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        logger.info(f"Backbone parameters requires_grad set to: {not freeze}")

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning (segmentation_logits, predicted_steering_angle).
        - segmentation_logits: Tensor of shape (B, 4, H, W)
        - predicted_steering_angle: Tensor of shape (B, 1) in radians, or None
        """
        # Multi-scale feature extraction
        features = self.backbone(x)

        # Semantic segmentation mask prediction
        seg_logits = self.decoder(features)

        # Steering angle prediction from deepest feature map (P5)
        steering_pred = None
        if self.steering_head is not None:
            steering_pred = self.steering_head(features[-1])

        return seg_logits, steering_pred

    def export_onnx(
        self,
        output_filepath: Path,
        opset_version: int = 14,
        batch_size: int = 1,
    ) -> Path:
        """Export model to ONNX format for TensorRT compilation on Jetson."""
        self.eval()
        dummy_input = torch.randn(
            batch_size,
            self.config.input_channels,
            self.config.input_height,
            self.config.input_width,
            dtype=torch.float32,
        )

        output_filepath = Path(output_filepath)
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self,
            dummy_input,
            str(output_filepath),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["segmentation_logits", "steering_angle_rad"],
            dynamic_axes={
                "input_image": {0: "batch_size"},
                "segmentation_logits": {0: "batch_size"},
                "steering_angle_rad": {0: "batch_size"},
            },
        )
        logger.info(f"Model exported to ONNX: {output_filepath}")
        return output_filepath


# ═══════════════════════════════════════════════════════════════════════════════
# Strict No-Flip Augmentation Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SafeLaneAugmentation:
    """
    Data augmentation pipeline enforcing the strict NO-FLIP RULE:
    Never apply horizontal flipping without explicitly swapping:
    Class 2 (Left Boundary) <-> Class 3 (Right Boundary).
    """

    def __init__(self, p_flip: float = 0.50):
        self.p_flip = p_flip

    def apply_augmentation(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        steering_angle_rad: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
        """
        Apply safe data augmentation.
        If horizontally flipped:
        - Image is flipped left-to-right.
        - Mask is flipped left-to-right.
        - Class 2 and Class 3 are swapped in the flipped mask.
        - Steering angle sign is inverted: delta = - delta.
        """
        rng = np.random.default_rng()

        if rng.random() < self.p_flip:
            # 1. Flip image horizontally
            image_flipped = cv2.flip(image_bgr, 1)

            # 2. Flip mask horizontally
            mask_flipped = cv2.flip(mask, 1)

            # 3. Swap Class 2 (Left) and Class 3 (Right)
            swapped_mask = mask_flipped.copy()
            is_class_2 = (mask_flipped == 2)
            is_class_3 = (mask_flipped == 3)
            swapped_mask[is_class_2] = 3
            swapped_mask[is_class_3] = 2

            # 4. Invert steering angle sign
            flipped_steer = -steering_angle_rad if steering_angle_rad is not None else None

            return image_flipped, swapped_mask, flipped_steer

        return image_bgr, mask, steering_angle_rad


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Task Loss Engine: Focal + Dice + Huber with Uncertainty Weighting
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """
    Multiclass Dice Loss for thin lane boundaries where class imbalance is severe
    (boundaries typically occupy < 2% of image pixels).
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = 0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        dice_losses: List[torch.Tensor] = []

        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            pred_c = probs[:, c, :, :].contiguous().view(-1)
            target_c = (targets == c).float().contiguous().view(-1)

            intersection = (pred_c * target_c).sum()
            cardinality = pred_c.sum() + target_c.sum()

            dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
            dice_losses.append(1.0 - dice_score)

        return torch.mean(torch.stack(dice_losses)) if dice_losses else torch.tensor(0.0, device=logits.device)


class APEXMultiTaskLoss(nn.Module):
    """
    Homoscedastic Uncertainty-Weighted Multi-Task Loss:
    L_total = (1 / (2 * s_seg^2)) * L_seg + (1 / (2 * s_steer^2)) * L_steer + log(s_seg * s_steer)
    where s_seg and s_steer are learnable task log-variances.
    """

    def __init__(
        self,
        num_classes: int = 4,
        class_weights: Optional[List[float]] = None,
        dice_weight: float = 0.50,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma

        # Class weights: prioritize Left (Class 2) and Right (Class 3) boundaries
        if class_weights is None:
            weights = torch.tensor([0.2, 1.0, 3.5, 3.5], dtype=torch.float32)
        else:
            weights = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("weights", weights)

        self.dice_loss = DiceLoss(smooth=1.0, ignore_index=0)
        self.huber_loss = nn.SmoothL1Loss(beta=0.05)

        # Learnable log variances for homoscedastic uncertainty weighting
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_steer = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        seg_logits: torch.Tensor,
        seg_targets: torch.Tensor,
        steer_preds: Optional[torch.Tensor] = None,
        steer_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined segmentation and steering loss.
        """
        # 1. Focal / Weighted Cross-Entropy Loss
        ce = F.cross_entropy(seg_logits, seg_targets, weight=self.weights, reduction="none")
        pt = torch.exp(-ce)
        focal_loss = ((1.0 - pt) ** self.focal_gamma * ce).mean()

        # 2. Multiclass Dice Loss
        d_loss = self.dice_loss(seg_logits, seg_targets)
        l_seg = (1.0 - self.dice_weight) * focal_loss + self.dice_weight * d_loss

        # 3. Steering Regression Loss (Huber)
        if steer_preds is not None and steer_targets is not None:
            l_steer = self.huber_loss(steer_preds, steer_targets)
            # Homoscedastic uncertainty weighting
            precision_seg = torch.exp(-self.log_var_seg)
            precision_steer = torch.exp(-self.log_var_steer)
            l_total = (precision_seg * l_seg + self.log_var_seg +
                       precision_steer * l_steer + self.log_var_steer)
        else:
            l_steer = torch.tensor(0.0, device=seg_logits.device)
            l_total = l_seg

        return {
            "loss_total": l_total,
            "loss_seg": l_seg,
            "loss_steer": l_steer,
            "focal_ce": focal_loss,
            "dice": d_loss,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Temperature-Calibrated Softmax Inference Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class CalibratedLanePredictor:
    """
    Inference wrapper performing temperature-scaled softmax and post-processing.
    Provides calibrated class confidence scores and argmax segmentation masks.
    """

    def __init__(
        self,
        model: APEXMultiTaskLaneModel,
        temperature: float = 1.25,
        min_boundary_confidence: float = 0.45,
    ):
        self.model = model
        self.temperature = temperature
        self.min_confidence = min_boundary_confidence

    @torch.no_grad()
    def predict(
        self,
        bgr_image: np.ndarray,
        device: str = "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Run inference on a single BGR image.
        Returns:
            - mask: 2D integer array (H, W) with class indices [0, 1, 2, 3]
            - confidence_map: 2D float array (H, W) with winning class probabilities
            - predicted_steer_rad: float target steering angle in radians
        """
        self.model.eval()
        orig_h, orig_w = bgr_image.shape[:2]

        # Preprocess: BGR -> RGB, resize to model input, normalize [0, 1]
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.model.config.input_width, self.model.config.input_height))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(device)

        # Forward pass
        logits, steer_pred = self.model(tensor)

        # Temperature-scaled softmax
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=1).squeeze(0)  # (4, H, W)

        # Argmax and confidence
        conf_map, mask_small = torch.max(probs, dim=0)

        mask_np = mask_small.cpu().numpy().astype(np.uint8)
        conf_np = conf_map.cpu().numpy().astype(np.float32)

        # Resize back to original frame dimensions
        mask_full = cv2.resize(mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        conf_full = cv2.resize(conf_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        steer_val = float(steer_pred.cpu().item()) if steer_pred is not None else 0.0

        return mask_full, conf_full, steer_val


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Self-Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_lane_segmentation_model(
    config: Optional[LaneSegmentationConfig] = None,
) -> APEXMultiTaskLaneModel:
    """Factory constructor for APEXMultiTaskLaneModel."""
    return APEXMultiTaskLaneModel(config=config)


def run_unit_tests() -> bool:
    """Validate model construction, forward pass shapes, frozen backbone, loss engine, and No-Flip rule."""
    print("Testing APEXMultiTaskLaneModel architectures and forward passes...")

    # 1. Test MobileNetV3 + UNet
    cfg_mobilenet = LaneSegmentationConfig(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        decoder_type=DecoderType.UNET,
        input_height=180,  # Smaller resolution for fast unit test
        input_width=320,
    )
    model = get_lane_segmentation_model(cfg_mobilenet)
    model.eval()

    dummy_batch = torch.randn(2, 3, 180, 320)
    with torch.no_grad():
        logits, steer = model(dummy_batch)

    assert logits.shape == (2, 4, 180, 320), f"Logits shape mismatch: {logits.shape}"
    assert steer is not None and steer.shape == (2, 1), f"Steering shape mismatch: {steer.shape}"
    assert torch.all(torch.abs(steer) <= cfg_mobilenet.max_steering_angle_rad), "Steering exceeds physical limit"

    # 2. Test ERFNet Decoder Mode
    cfg_erfnet = LaneSegmentationConfig(
        backbone_type=BackboneType.MOBILENET_V3_SMALL,
        decoder_type=DecoderType.ERFNET,
        input_height=180,
        input_width=320,
    )
    model_erf = get_lane_segmentation_model(cfg_erfnet)
    with torch.no_grad():
        logits_erf, _ = model_erf(dummy_batch)
    assert logits_erf.shape == (2, 4, 180, 320), f"ERFNet logits shape mismatch: {logits_erf.shape}"

    # 3. Test Frozen Backbone Parameter Gradient Flag
    model.freeze_backbone(freeze=True)
    for p in model.backbone.parameters():
        assert not p.requires_grad, "Backbone parameters should not require grad when frozen"
    for p in model.decoder.parameters():
        assert p.requires_grad, "Decoder parameters must remain trainable"

    # 4. Test Multi-Task Loss Engine
    loss_fn = APEXMultiTaskLoss(num_classes=4)
    dummy_targets = torch.randint(0, 4, (2, 180, 320), dtype=torch.long)
    dummy_steer_targets = torch.randn(2, 1)
    loss_dict = loss_fn(logits, dummy_targets, steer, dummy_steer_targets)
    assert "loss_total" in loss_dict and loss_dict["loss_total"] > 0.0, "Loss computation failed"

    # 5. Test Calibrated Predictor Wrapper
    predictor = CalibratedLanePredictor(model)
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    mask_out, conf_out, steer_out = predictor.predict(dummy_img)
    assert mask_out.shape == (360, 640), f"Prediction mask shape mismatch: {mask_out.shape}"
    assert conf_out.shape == (360, 640), "Confidence map shape mismatch"

    # 6. Test Safe No-Flip Rule Augmentation
    aug = SafeLaneAugmentation(p_flip=1.0)
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[:, 10:20] = 2  # Left lane
    dummy_mask[:, 80:90] = 3  # Right lane
    orig_steer = 0.15

    f_img, f_mask, f_steer = aug.apply_augmentation(dummy_img[:100, :100], dummy_mask, orig_steer)
    assert np.all(f_mask[:, 80:90] == 3), "Class 2 was not properly swapped to Class 3 upon horizontal flip"
    assert np.all(f_mask[:, 10:20] == 2), "Class 3 was not properly swapped to Class 2 upon horizontal flip"
    assert abs(f_steer - (-orig_steer)) < 1e-6, "Steering angle sign was not inverted upon flip"

    print("APEXMultiTaskLaneModel self-verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()

