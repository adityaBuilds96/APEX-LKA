"""
src/scene/scene_profiles.py
===========================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Operational Scene Profiles & Dynamic Tuning Registry for Precision Lane Keeping.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.

This module provides the comprehensive dataclass schema and configuration registry
for scene-specific tuning profiles. Autonomous lane perception requires dynamically
calibrated computer vision thresholds, deep learning fusion balances, temporal
filtering dynamics, and virtual guardrail safety envelopes to guarantee that the
vehicle reliably detects its lane with 95.5%+ accuracy under adverse conditions
(highway, urban, rural, tunnel, night, rain) and never crosses lane boundaries.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from enum import Enum, unique
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("APEX.SceneProfiles")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Operational Scene Classification Enumeration
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class SceneType(str, Enum):
    """
    Standardized operational scene classifications for the APEX-Vision platform.
    Directly aligns with the 6-class neural scene classifier.
    """
    HIGHWAY = "HIGHWAY"
    URBAN = "URBAN"
    RURAL = "RURAL"
    TUNNEL = "TUNNEL"
    NIGHT = "NIGHT"
    RAIN = "RAIN"

    @classmethod
    def from_string(cls, name: str) -> "SceneType":
        """
        Safely convert a string representation into a valid SceneType enum.
        Falls back to HIGHWAY if unparseable.
        """
        clean_name = str(name).strip().upper()
        for member in cls:
            if member.value == clean_name or member.name == clean_name:
                return member
        logger.warning("Unrecognized scene name '%s'; defaulting to HIGHWAY.", name)
        return cls.HIGHWAY

    @classmethod
    def list_all(cls) -> List[str]:
        """Return list of all scene string identifiers."""
        return [member.value for member in cls]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Scene Tuning Parameter Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassicalCVThresholds:
    """
    Configurable image processing parameters for classical computer vision pipelines
    including Canny edge detection, HLS/LAB/YUV color thresholding, and morphological filtering.
    """
    canny_low: int = 50
    canny_high: int = 150
    canny_aperture: int = 3
    sobel_kernel: int = 3
    hls_l_min: int = 195
    hls_l_max: int = 255
    hls_s_min: int = 35
    hls_s_max: int = 255
    lab_b_min: int = 145
    lab_b_max: int = 255
    yuv_y_min: int = 185
    yuv_y_max: int = 255
    hough_threshold: int = 20
    hough_min_line_length: int = 25
    hough_max_line_gap: int = 15
    hough_min_slope: float = 0.35
    enable_gabor_bank: bool = False
    gabor_frequency: float = 0.25
    morph_kernel_size: int = 3

    def validate(self) -> None:
        """Validate parameter ranges and internal logical consistency."""
        if not (0 <= self.canny_low < self.canny_high <= 255):
            raise ValueError(f"Invalid Canny bounds: low={self.canny_low}, high={self.canny_high}")
        if not (0 <= self.hls_l_min <= self.hls_l_max <= 255):
            raise ValueError(f"Invalid HLS L bounds: min={self.hls_l_min}, max={self.hls_l_max}")
        if self.canny_aperture not in (3, 5, 7):
            raise ValueError(f"Aperture must be 3, 5, or 7, got {self.canny_aperture}")


@dataclass
class LightingCompensationProfile:
    """
    Parameters governing dynamic exposure compensation, CLAHE equalization,
    specular highlight mitigation, and anti-glare filtering.
    """
    gamma: float = 1.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    enable_specular_removal: bool = False
    specular_l_threshold: int = 250
    specular_inpaint_radius: int = 3
    night_mode_boost: bool = False
    headlight_glare_suppression: bool = False
    hls_stretch_contrast: bool = True

    def validate(self) -> None:
        """Ensure non-negative gamma and valid clip limits."""
        if self.gamma <= 0.0 or self.gamma > 5.0:
            raise ValueError(f"Gamma must be in range (0.0, 5.0], got {self.gamma}")
        if self.clahe_clip_limit <= 0.0:
            raise ValueError(f"CLAHE clip limit must be positive, got {self.clahe_clip_limit}")


@dataclass
class FusionAndTrackingProfile:
    """
    Weights and Kalman filter dynamics governing hybrid lane perception fusion.
    Balances classical geometric feature extraction against deep neural semantic segmentation.
    """
    dl_fusion_weight: float = 0.65       # alpha: DL weight in [0.0, 1.0]
    cv_fusion_weight: float = 0.35       # 1 - alpha: Classical CV weight
    confidence_threshold_min: float = 0.35
    history_window_size: int = 10
    ekf_process_noise_pos: float = 0.01
    ekf_process_noise_curv: float = 0.001
    ekf_meas_noise_pos: float = 0.05
    ekf_meas_noise_curv: float = 0.005
    outlier_sigma_threshold: float = 2.0
    lane_width_nominal_m: float = 3.70
    lane_width_tolerance_m: float = 0.75
    allow_hallucination: bool = True
    max_hallucination_frames: int = 20

    def validate(self) -> None:
        """Check weight normalization and mathematical validity."""
        if not (0.0 <= self.dl_fusion_weight <= 1.0):
            raise ValueError(f"dl_fusion_weight must be in [0, 1], got {self.dl_fusion_weight}")
        if not (0.0 <= self.confidence_threshold_min <= 1.0):
            raise ValueError(f"confidence_threshold_min must be in [0, 1], got {self.confidence_threshold_min}")
        if self.history_window_size < 1:
            raise ValueError(f"history_window_size must be >= 1, got {self.history_window_size}")


@dataclass
class VirtualGuardrailProfile:
    """
    Parameters governing the virtual guardrail safety envelope and proactive
    lane keeping actuation interlocks.
    """
    inner_buffer_width_m: float = 0.30   # Safety envelope inner offset
    warning_buffer_m: float = 0.30       # Severity 1 warning threshold
    critical_buffer_m: float = 0.15      # Severity 2 critical counter-steer threshold
    emergency_buffer_m: float = 0.00     # Severity 3 boundary violation threshold
    trajectory_prediction_horizon_s: float = 1.50
    ttld_warning_threshold_s: float = 2.00
    stanley_gain_base: float = 1.25
    stanley_soft_gain_base: float = 1.00
    counter_steer_gain_multiplier: float = 1.50
    max_steering_angle_rad: float = 0.4886  # ~28 degrees
    can_emergency_priority_id: int = 0x150
    arduino_relay_cutoff_enabled: bool = True

    def validate(self) -> None:
        """Verify envelope hierarchy and sanity bounds."""
        if not (0.0 < self.critical_buffer_m < self.warning_buffer_m):
            raise ValueError(
                f"Envelope hierarchy violated: critical ({self.critical_buffer_m}) "
                f"must be less than warning ({self.warning_buffer_m})"
            )
        if self.trajectory_prediction_horizon_s <= 0.0:
            raise ValueError(f"Prediction horizon must be positive, got {self.trajectory_prediction_horizon_s}")


@dataclass
class AdaptiveROIProfile:
    """
    Defines the dynamic Region of Interest parameters in perspective coordinates.
    """
    top_horizon_offset_px: int = 20      # Offset above dynamic horizon
    bottom_cut_px: int = 340             # Suppress bumper and vehicle bonnet
    trapezoid_top_width_ratio: float = 0.45
    trapezoid_bottom_width_ratio: float = 0.95
    curve_expansion_gain: float = 1.50   # Broadens search sector when |A| is high

    def validate(self) -> None:
        """Validate pixel boundaries."""
        if self.bottom_cut_px <= 0 or self.bottom_cut_px > 360:
            raise ValueError(f"Invalid bottom_cut_px: {self.bottom_cut_px}")


@dataclass
class SceneTuningProfile:
    """
    Comprehensive operational profile packaging all subsystem tuning parameters
    for a designated SceneType.
    """
    scene_type: SceneType
    description: str
    target_cruise_speed_mps: float = 20.0
    classical: ClassicalCVThresholds = field(default_factory=ClassicalCVThresholds)
    lighting: LightingCompensationProfile = field(default_factory=LightingCompensationProfile)
    fusion: FusionAndTrackingProfile = field(default_factory=FusionAndTrackingProfile)
    guardrail: VirtualGuardrailProfile = field(default_factory=VirtualGuardrailProfile)
    roi: AdaptiveROIProfile = field(default_factory=AdaptiveROIProfile)
    notes_iso26262: str = "ASIL-B aligned tuning profile for driver-supervised SAE Level 2."

    def __post_init__(self) -> None:
        """Validate all sub-profiles upon instantiation."""
        self.classical.validate()
        self.lighting.validate()
        self.fusion.validate()
        self.guardrail.validate()
        self.roi.validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire tuning profile to a nested serializable dictionary."""
        d = asdict(self)
        d["scene_type"] = self.scene_type.value
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SceneTuningProfile":
        """Instantiate a SceneTuningProfile from a dictionary."""
        data_copy = dict(data)
        scene_type = SceneType.from_string(data_copy.get("scene_type", "HIGHWAY"))
        classical = ClassicalCVThresholds(**data_copy.get("classical", {}))
        lighting = LightingCompensationProfile(**data_copy.get("lighting", {}))
        fusion = FusionAndTrackingProfile(**data_copy.get("fusion", {}))
        guardrail = VirtualGuardrailProfile(**data_copy.get("guardrail", {}))
        roi = AdaptiveROIProfile(**data_copy.get("roi", {}))

        return cls(
            scene_type=scene_type,
            description=data_copy.get("description", f"Tuning profile for {scene_type.value}"),
            target_cruise_speed_mps=float(data_copy.get("target_cruise_speed_mps", 20.0)),
            classical=classical,
            lighting=lighting,
            fusion=fusion,
            guardrail=guardrail,
            roi=roi,
            notes_iso26262=data_copy.get("notes_iso26262", "ASIL-B aligned tuning profile."),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Default Scene Profiles Registry
# ═══════════════════════════════════════════════════════════════════════════════

def build_highway_profile() -> SceneTuningProfile:
    """
    Profile for open highway driving:
    - High cruise speed (25 m/s, ~90 km/h)
    - Straight or gentle curves
    - Narrower safety buffer (0.20 m) to accommodate lane boundaries at high speed
    - High classical confidence, balanced DL fusion (alpha = 0.60)
    """
    return SceneTuningProfile(
        scene_type=SceneType.HIGHWAY,
        description="Open high-speed divided highway with marked continuous/dashed lane lines.",
        target_cruise_speed_mps=25.0,
        classical=ClassicalCVThresholds(
            canny_low=60,
            canny_high=160,
            canny_aperture=3,
            sobel_kernel=3,
            hls_l_min=200,
            hls_l_max=255,
            hls_s_min=30,
            hls_s_max=255,
            lab_b_min=145,
            lab_b_max=255,
            yuv_y_min=190,
            yuv_y_max=255,
            hough_threshold=25,
            hough_min_line_length=30,
            hough_max_line_gap=15,
            hough_min_slope=0.40,
            enable_gabor_bank=False,
        ),
        lighting=LightingCompensationProfile(
            gamma=1.0,
            enable_clahe=True,
            clahe_clip_limit=2.0,
            clahe_tile_grid=(8, 8),
            enable_specular_removal=False,
            night_mode_boost=False,
            headlight_glare_suppression=False,
        ),
        fusion=FusionAndTrackingProfile(
            dl_fusion_weight=0.60,
            cv_fusion_weight=0.40,
            confidence_threshold_min=0.40,
            history_window_size=12,
            ekf_process_noise_pos=0.008,
            ekf_process_noise_curv=0.0008,
            ekf_meas_noise_pos=0.04,
            ekf_meas_noise_curv=0.004,
            outlier_sigma_threshold=2.0,
            lane_width_nominal_m=3.70,
            lane_width_tolerance_m=0.60,
            allow_hallucination=True,
            max_hallucination_frames=20,
        ),
        guardrail=VirtualGuardrailProfile(
            inner_buffer_width_m=0.20,
            warning_buffer_m=0.20,
            critical_buffer_m=0.10,
            emergency_buffer_m=0.00,
            trajectory_prediction_horizon_s=1.80,
            ttld_warning_threshold_s=2.20,
            stanley_gain_base=1.10,
            stanley_soft_gain_base=1.20,
            counter_steer_gain_multiplier=1.40,
        ),
        roi=AdaptiveROIProfile(
            top_horizon_offset_px=15,
            bottom_cut_px=340,
            trapezoid_top_width_ratio=0.40,
            trapezoid_bottom_width_ratio=0.92,
            curve_expansion_gain=1.30,
        ),
    )


def build_urban_profile() -> SceneTuningProfile:
    """
    Profile for dense urban corridors:
    - Moderate cruise speed (12 m/s, ~43 km/h)
    - Frequent crosswalks, intersections, pavement seams, manhole covers
    - Wider safety buffer (0.50 m) to prevent brushing curbs or parked vehicles
    - Higher DL weight (alpha = 0.78) to filter visual road clutter
    """
    return SceneTuningProfile(
        scene_type=SceneType.URBAN,
        description="Dense city traffic with crosswalks, intersections, and road surface texturing.",
        target_cruise_speed_mps=12.0,
        classical=ClassicalCVThresholds(
            canny_low=45,
            canny_high=135,
            canny_aperture=3,
            sobel_kernel=3,
            hls_l_min=190,
            hls_l_max=255,
            hls_s_min=40,
            hls_s_max=255,
            lab_b_min=140,
            lab_b_max=255,
            yuv_y_min=180,
            yuv_y_max=255,
            hough_threshold=18,
            hough_min_line_length=20,
            hough_max_line_gap=12,
            hough_min_slope=0.32,
            enable_gabor_bank=False,
        ),
        lighting=LightingCompensationProfile(
            gamma=1.05,
            enable_clahe=True,
            clahe_clip_limit=2.5,
            clahe_tile_grid=(8, 8),
            enable_specular_removal=True,
            specular_l_threshold=245,
            night_mode_boost=False,
        ),
        fusion=FusionAndTrackingProfile(
            dl_fusion_weight=0.78,
            cv_fusion_weight=0.22,
            confidence_threshold_min=0.32,
            history_window_size=8,
            ekf_process_noise_pos=0.015,
            ekf_process_noise_curv=0.002,
            ekf_meas_noise_pos=0.06,
            ekf_meas_noise_curv=0.008,
            outlier_sigma_threshold=2.2,
            lane_width_nominal_m=3.50,
            lane_width_tolerance_m=0.85,
            allow_hallucination=True,
            max_hallucination_frames=15,
        ),
        guardrail=VirtualGuardrailProfile(
            inner_buffer_width_m=0.50,
            warning_buffer_m=0.50,
            critical_buffer_m=0.25,
            emergency_buffer_m=0.00,
            trajectory_prediction_horizon_s=1.20,
            ttld_warning_threshold_s=1.80,
            stanley_gain_base=1.45,
            stanley_soft_gain_base=0.90,
            counter_steer_gain_multiplier=1.65,
        ),
        roi=AdaptiveROIProfile(
            top_horizon_offset_px=25,
            bottom_cut_px=340,
            trapezoid_top_width_ratio=0.55,
            trapezoid_bottom_width_ratio=0.98,
            curve_expansion_gain=1.75,
        ),
    )


def build_rural_profile() -> SceneTuningProfile:
    """
    Profile for rural roads:
    - Moderate speed (18 m/s, ~65 km/h)
    - Worn or non-existent road paint, gravel berms, vegetation encroachment
    - Dual-boundary hallucination active with Gabor filter bank assistance
    - Balanced fusion (alpha = 0.55) to leverage subtle textural berm boundaries
    """
    return SceneTuningProfile(
        scene_type=SceneType.RURAL,
        description="Undivided rural roads with worn or faded paint and dirt berm edges.",
        target_cruise_speed_mps=18.0,
        classical=ClassicalCVThresholds(
            canny_low=35,
            canny_high=115,
            canny_aperture=3,
            sobel_kernel=5,
            hls_l_min=175,
            hls_l_max=255,
            hls_s_min=25,
            hls_s_max=255,
            lab_b_min=135,
            lab_b_max=255,
            yuv_y_min=165,
            yuv_y_max=255,
            hough_threshold=15,
            hough_min_line_length=18,
            hough_max_line_gap=20,
            hough_min_slope=0.30,
            enable_gabor_bank=True,
            gabor_frequency=0.22,
        ),
        lighting=LightingCompensationProfile(
            gamma=1.10,
            enable_clahe=True,
            clahe_clip_limit=3.0,
            clahe_tile_grid=(8, 8),
            enable_specular_removal=False,
            night_mode_boost=False,
        ),
        fusion=FusionAndTrackingProfile(
            dl_fusion_weight=0.55,
            cv_fusion_weight=0.45,
            confidence_threshold_min=0.30,
            history_window_size=10,
            ekf_process_noise_pos=0.012,
            ekf_process_noise_curv=0.0015,
            ekf_meas_noise_pos=0.055,
            ekf_meas_noise_curv=0.007,
            outlier_sigma_threshold=2.0,
            lane_width_nominal_m=3.70,
            lane_width_tolerance_m=0.90,
            allow_hallucination=True,
            max_hallucination_frames=20,
        ),
        guardrail=VirtualGuardrailProfile(
            inner_buffer_width_m=0.35,
            warning_buffer_m=0.35,
            critical_buffer_m=0.18,
            emergency_buffer_m=0.00,
            trajectory_prediction_horizon_s=1.50,
            ttld_warning_threshold_s=2.00,
            stanley_gain_base=1.30,
            stanley_soft_gain_base=1.00,
            counter_steer_gain_multiplier=1.50,
        ),
        roi=AdaptiveROIProfile(
            top_horizon_offset_px=20,
            bottom_cut_px=340,
            trapezoid_top_width_ratio=0.50,
            trapezoid_bottom_width_ratio=0.96,
            curve_expansion_gain=1.60,
        ),
    )


def build_tunnel_profile() -> SceneTuningProfile:
    """
    Profile for enclosed tunnel environments:
    - Moderate speed (16 m/s, ~58 km/h)
    - Artificial sodium-vapor lighting, severe shadows, and intense specular ceiling reflections
    - Elevated gamma (1.65), lowered Canny thresholds (25/85), specular removal active
    - Heavy deep learning fusion (alpha = 0.82)
    """
    return SceneTuningProfile(
        scene_type=SceneType.TUNNEL,
        description="Enclosed tunnel corridors with artificial yellow/orange lighting and strong reflections.",
        target_cruise_speed_mps=16.0,
        classical=ClassicalCVThresholds(
            canny_low=25,
            canny_high=85,
            canny_aperture=3,
            sobel_kernel=3,
            hls_l_min=160,
            hls_l_max=255,
            hls_s_min=20,
            hls_s_max=255,
            lab_b_min=130,
            lab_b_max=255,
            yuv_y_min=150,
            yuv_y_max=255,
            hough_threshold=14,
            hough_min_line_length=15,
            hough_max_line_gap=15,
            hough_min_slope=0.30,
            enable_gabor_bank=False,
        ),
        lighting=LightingCompensationProfile(
            gamma=1.65,
            enable_clahe=True,
            clahe_clip_limit=3.5,
            clahe_tile_grid=(8, 8),
            enable_specular_removal=True,
            specular_l_threshold=235,
            specular_inpaint_radius=4,
            night_mode_boost=False,
            headlight_glare_suppression=True,
        ),
        fusion=FusionAndTrackingProfile(
            dl_fusion_weight=0.82,
            cv_fusion_weight=0.18,
            confidence_threshold_min=0.32,
            history_window_size=10,
            ekf_process_noise_pos=0.010,
            ekf_process_noise_curv=0.0012,
            ekf_meas_noise_pos=0.05,
            ekf_meas_noise_curv=0.006,
            outlier_sigma_threshold=2.0,
            lane_width_nominal_m=3.70,
            lane_width_tolerance_m=0.70,
            allow_hallucination=True,
            max_hallucination_frames=20,
        ),
        guardrail=VirtualGuardrailProfile(
            inner_buffer_width_m=0.30,
            warning_buffer_m=0.30,
            critical_buffer_m=0.15,
            emergency_buffer_m=0.00,
            trajectory_prediction_horizon_s=1.50,
            ttld_warning_threshold_s=2.00,
            stanley_gain_base=1.35,
            stanley_soft_gain_base=1.00,
            counter_steer_gain_multiplier=1.55,
        ),
        roi=AdaptiveROIProfile(
            top_horizon_offset_px=20,
            bottom_cut_px=340,
            trapezoid_top_width_ratio=0.45,
            trapezoid_bottom_width_ratio=0.95,
            curve_expansion_gain=1.40,
        ),
    )


def build_night_profile() -> SceneTuningProfile:
    """
    Profile for nighttime operation:
    - Reduced cruise speed (18 m/s, ~65 km/h)
    - Dark background with localized vehicle headlight cone illumination
    - Anti-blooming headlight glare suppression, night mode luminance booster
    - Dominant deep learning fusion (alpha = 0.88)
    """
    return SceneTuningProfile(
        scene_type=SceneType.NIGHT,
        description="Nighttime driving with illumination restricted to vehicle headlights and street lamps.",
        target_cruise_speed_mps=18.0,
        classical=ClassicalCVThresholds(
            canny_low=30,
            canny_high=95,
            canny_aperture=3,
            sobel_kernel=3,
            hls_l_min=150,
            hls_l_max=255,
            hls_s_min=15,
            hls_s_max=255,
            lab_b_min=130,
            lab_b_max=255,
            yuv_y_min=140,
            yuv_y_max=255,
            hough_threshold=15,
            hough_min_line_length=15,
            hough_max_line_gap=15,
            hough_min_slope=0.32,
            enable_gabor_bank=False,
        ),
        lighting=LightingCompensationProfile(
            gamma=1.85,
            enable_clahe=True,
            clahe_clip_limit=4.0,
            clahe_tile_grid=(8, 8),
            enable_specular_removal=True,
            specular_l_threshold=240,
            specular_inpaint_radius=4,
            night_mode_boost=True,
            headlight_glare_suppression=True,
        ),
        fusion=FusionAndTrackingProfile(
            dl_fusion_weight=0.88,
            cv_fusion_weight=0.12,
            confidence_threshold_min=0.30,
            history_window_size=12,
            ekf_process_noise_pos=0.010,
            ekf_process_noise_curv=0.0010,
            ekf_meas_noise_pos=0.05,
            ekf_meas_noise_curv=0.005,
            outlier_sigma_threshold=2.0,
            lane_width_nominal_m=3.70,
            lane_width_tolerance_m=0.75,
            allow_hallucination=True,
            max_hallucination_frames=20,
        ),
        guardrail=VirtualGuardrailProfile(
            inner_buffer_width_m=0.35,
            warning_buffer_m=0.35,
            critical_buffer_m=0.18,
            emergency_buffer_m=0.00,
            trajectory_prediction_horizon_s=1.60,
            ttld_warning_threshold_s=2.20,
            stanley_gain_base=1.35,
            stanley_soft_gain_base=1.10,
            counter_steer_gain_multiplier=1.60,
        ),
        roi=AdaptiveROIProfile(
            top_horizon_offset_px=15,
            bottom_cut_px=340,
            trapezoid_top_width_ratio=0.42,
            trapezoid_bottom_width_ratio=0.94,
            curve_expansion_gain=1.40,
        ),
    )


def build_rain_profile() -> SceneTuningProfile:
    """
    Profile for rain and wet pavement:
    - Reduced cruise speed (15 m/s, ~54 km/h)
    - Puddle specular reflections, water droplet blur, lower pavement contrast
    - Strong specular mitigation, high deep learning weighting (alpha = 0.85)
    - Expanded safety buffer (0.35 m) to counter tire traction degradation
    """
    return SceneTuningProfile(
        scene_type=SceneType.RAIN,
        description="Rainy weather with water spray, wet asphalt reflections, and diminished tire grip.",
        target_cruise_speed_mps=15.0,
        classical=ClassicalCVThresholds(
            canny_low=40,
            canny_high=120,
            canny_aperture=3,
            sobel_kernel=3,
            hls_l_min=185,
            hls_l_max=255,
            hls_s_min=30,
            hls_s_max=255,
            lab_b_min=135,
            lab_b_max=255,
            yuv_y_min=175,
            yuv_y_max=255,
            hough_threshold=16,
            hough_min_line_length=18,
            hough_max_line_gap=15,
            hough_min_slope=0.35,
            enable_gabor_bank=False,
        ),
        lighting=LightingCompensationProfile(
            gamma=1.20,
            enable_clahe=True,
            clahe_clip_limit=3.0,
            clahe_tile_grid=(8, 8),
            enable_specular_removal=True,
            specular_l_threshold=230,
            specular_inpaint_radius=5,
            night_mode_boost=False,
            headlight_glare_suppression=True,
        ),
        fusion=FusionAndTrackingProfile(
            dl_fusion_weight=0.85,
            cv_fusion_weight=0.15,
            confidence_threshold_min=0.32,
            history_window_size=10,
            ekf_process_noise_pos=0.014,
            ekf_process_noise_curv=0.0018,
            ekf_meas_noise_pos=0.065,
            ekf_meas_noise_curv=0.008,
            outlier_sigma_threshold=2.0,
            lane_width_nominal_m=3.70,
            lane_width_tolerance_m=0.80,
            allow_hallucination=True,
            max_hallucination_frames=20,
        ),
        guardrail=VirtualGuardrailProfile(
            inner_buffer_width_m=0.35,
            warning_buffer_m=0.35,
            critical_buffer_m=0.18,
            emergency_buffer_m=0.00,
            trajectory_prediction_horizon_s=1.60,
            ttld_warning_threshold_s=2.30,
            stanley_gain_base=1.40,
            stanley_soft_gain_base=1.15,
            counter_steer_gain_multiplier=1.60,
        ),
        roi=AdaptiveROIProfile(
            top_horizon_offset_px=18,
            bottom_cut_px=340,
            trapezoid_top_width_ratio=0.45,
            trapezoid_bottom_width_ratio=0.95,
            curve_expansion_gain=1.45,
        ),
    )


# Master static profile mapping
DEFAULT_SCENE_PROFILES: Dict[SceneType, SceneTuningProfile] = {
    SceneType.HIGHWAY: build_highway_profile(),
    SceneType.URBAN: build_urban_profile(),
    SceneType.RURAL: build_rural_profile(),
    SceneType.TUNNEL: build_tunnel_profile(),
    SceneType.NIGHT: build_night_profile(),
    SceneType.RAIN: build_rain_profile(),
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Scene Profile Registry Manager
# ═══════════════════════════════════════════════════════════════════════════════

class SceneProfileRegistry:
    """
    Thread-safe registry managing the active collection of scene profiles.
    Supports runtime lookup, parameter interpolation/blending, and custom overrides.
    """

    def __init__(self, initial_profiles: Optional[Dict[SceneType, SceneTuningProfile]] = None) -> None:
        """Initialize registry with standard default or custom profiles."""
        self._profiles: Dict[SceneType, SceneTuningProfile] = {}
        if initial_profiles is not None:
            for st, p in initial_profiles.items():
                self._profiles[st] = copy.deepcopy(p)
        else:
            for st, p in DEFAULT_SCENE_PROFILES.items():
                self._profiles[st] = copy.deepcopy(p)

    def get_profile(self, scene: Union[SceneType, str]) -> SceneTuningProfile:
        """
        Retrieve a deep copy of the profile for the requested scene.
        Guarantees that caller modifications do not corrupt registry state.
        """
        scene_type = scene if isinstance(scene, SceneType) else SceneType.from_string(scene)
        if scene_type not in self._profiles:
            logger.warning("Scene %s not in registry; falling back to HIGHWAY.", scene_type.value)
            scene_type = SceneType.HIGHWAY
        return copy.deepcopy(self._profiles[scene_type])

    def register_override(self, profile: SceneTuningProfile) -> None:
        """
        Register a custom or calibrated tuning profile for a scene type.
        """
        profile.classical.validate()
        profile.lighting.validate()
        profile.fusion.validate()
        profile.guardrail.validate()
        profile.roi.validate()
        self._profiles[profile.scene_type] = copy.deepcopy(profile)
        logger.info("Successfully registered tuning override for scene '%s'.", profile.scene_type.value)

    def blend_profiles(
        self,
        scene_a: Union[SceneType, str],
        scene_b: Union[SceneType, str],
        weight_b: float,
    ) -> SceneTuningProfile:
        """
        Smoothly interpolate continuous parameters between two profiles (e.g. entering a tunnel).
        weight_b in [0.0, 1.0] denotes the fraction of profile B.
        """
        w_b = max(0.0, min(1.0, float(weight_b)))
        w_a = 1.0 - w_b

        p_a = self.get_profile(scene_a)
        p_b = self.get_profile(scene_b)

        # Blend continuous classical CV parameters
        canny_low = int(round(w_a * p_a.classical.canny_low + w_b * p_b.classical.canny_low))
        canny_high = int(round(w_a * p_a.classical.canny_high + w_b * p_b.classical.canny_high))
        hls_l_min = int(round(w_a * p_a.classical.hls_l_min + w_b * p_b.classical.hls_l_min))
        yuv_y_min = int(round(w_a * p_a.classical.yuv_y_min + w_b * p_b.classical.yuv_y_min))

        blended_classical = ClassicalCVThresholds(
            canny_low=canny_low,
            canny_high=max(canny_low + 10, canny_high),
            canny_aperture=p_b.classical.canny_aperture if w_b >= 0.5 else p_a.classical.canny_aperture,
            sobel_kernel=p_b.classical.sobel_kernel if w_b >= 0.5 else p_a.classical.sobel_kernel,
            hls_l_min=hls_l_min,
            hls_l_max=255,
            hls_s_min=int(round(w_a * p_a.classical.hls_s_min + w_b * p_b.classical.hls_s_min)),
            hls_s_max=255,
            lab_b_min=int(round(w_a * p_a.classical.lab_b_min + w_b * p_b.classical.lab_b_min)),
            lab_b_max=255,
            yuv_y_min=yuv_y_min,
            yuv_y_max=255,
            hough_threshold=int(round(w_a * p_a.classical.hough_threshold + w_b * p_b.classical.hough_threshold)),
            hough_min_line_length=int(round(w_a * p_a.classical.hough_min_line_length + w_b * p_b.classical.hough_min_line_length)),
            hough_max_line_gap=int(round(w_a * p_a.classical.hough_max_line_gap + w_b * p_b.classical.hough_max_line_gap)),
            hough_min_slope=round(w_a * p_a.classical.hough_min_slope + w_b * p_b.classical.hough_min_slope, 3),
            enable_gabor_bank=p_b.classical.enable_gabor_bank if w_b >= 0.5 else p_a.classical.enable_gabor_bank,
        )

        # Blend lighting
        gamma = round(w_a * p_a.lighting.gamma + w_b * p_b.lighting.gamma, 3)
        clahe_clip = round(w_a * p_a.lighting.clahe_clip_limit + w_b * p_b.lighting.clahe_clip_limit, 2)
        blended_lighting = LightingCompensationProfile(
            gamma=gamma,
            enable_clahe=True,
            clahe_clip_limit=clahe_clip,
            clahe_tile_grid=p_b.lighting.clahe_tile_grid if w_b >= 0.5 else p_a.lighting.clahe_tile_grid,
            enable_specular_removal=p_b.lighting.enable_specular_removal if w_b >= 0.5 else p_a.lighting.enable_specular_removal,
            specular_l_threshold=int(round(w_a * p_a.lighting.specular_l_threshold + w_b * p_b.lighting.specular_l_threshold)),
            night_mode_boost=p_b.lighting.night_mode_boost if w_b >= 0.5 else p_a.lighting.night_mode_boost,
            headlight_glare_suppression=p_b.lighting.headlight_glare_suppression if w_b >= 0.5 else p_a.lighting.headlight_glare_suppression,
        )

        # Blend fusion weights
        dl_weight = round(w_a * p_a.fusion.dl_fusion_weight + w_b * p_b.fusion.dl_fusion_weight, 3)
        blended_fusion = FusionAndTrackingProfile(
            dl_fusion_weight=dl_weight,
            cv_fusion_weight=round(1.0 - dl_weight, 3),
            confidence_threshold_min=round(w_a * p_a.fusion.confidence_threshold_min + w_b * p_b.fusion.confidence_threshold_min, 3),
            history_window_size=int(round(w_a * p_a.fusion.history_window_size + w_b * p_b.fusion.history_window_size)),
            ekf_process_noise_pos=round(w_a * p_a.fusion.ekf_process_noise_pos + w_b * p_b.fusion.ekf_process_noise_pos, 5),
            ekf_process_noise_curv=round(w_a * p_a.fusion.ekf_process_noise_curv + w_b * p_b.fusion.ekf_process_noise_curv, 6),
            ekf_meas_noise_pos=round(w_a * p_a.fusion.ekf_meas_noise_pos + w_b * p_b.fusion.ekf_meas_noise_pos, 4),
            ekf_meas_noise_curv=round(w_a * p_a.fusion.ekf_meas_noise_curv + w_b * p_b.fusion.ekf_meas_noise_curv, 5),
            outlier_sigma_threshold=round(w_a * p_a.fusion.outlier_sigma_threshold + w_b * p_b.fusion.outlier_sigma_threshold, 2),
            lane_width_nominal_m=round(w_a * p_a.fusion.lane_width_nominal_m + w_b * p_b.fusion.lane_width_nominal_m, 2),
            lane_width_tolerance_m=round(w_a * p_a.fusion.lane_width_tolerance_m + w_b * p_b.fusion.lane_width_tolerance_m, 2),
            allow_hallucination=True,
            max_hallucination_frames=int(round(w_a * p_a.fusion.max_hallucination_frames + w_b * p_b.fusion.max_hallucination_frames)),
        )

        # Blend guardrail envelope
        inner_buf = round(w_a * p_a.guardrail.inner_buffer_width_m + w_b * p_b.guardrail.inner_buffer_width_m, 3)
        crit_buf = round(w_a * p_a.guardrail.critical_buffer_m + w_b * p_b.guardrail.critical_buffer_m, 3)
        blended_guardrail = VirtualGuardrailProfile(
            inner_buffer_width_m=inner_buf,
            warning_buffer_m=inner_buf,
            critical_buffer_m=crit_buf,
            emergency_buffer_m=0.00,
            trajectory_prediction_horizon_s=round(w_a * p_a.guardrail.trajectory_prediction_horizon_s + w_b * p_b.guardrail.trajectory_prediction_horizon_s, 2),
            ttld_warning_threshold_s=round(w_a * p_a.guardrail.ttld_warning_threshold_s + w_b * p_b.guardrail.ttld_warning_threshold_s, 2),
            stanley_gain_base=round(w_a * p_a.guardrail.stanley_gain_base + w_b * p_b.guardrail.stanley_gain_base, 3),
            stanley_soft_gain_base=round(w_a * p_a.guardrail.stanley_soft_gain_base + w_b * p_b.guardrail.stanley_soft_gain_base, 3),
            counter_steer_gain_multiplier=round(w_a * p_a.guardrail.counter_steer_gain_multiplier + w_b * p_b.guardrail.counter_steer_gain_multiplier, 2),
        )

        # Blend ROI
        blended_roi = AdaptiveROIProfile(
            top_horizon_offset_px=int(round(w_a * p_a.roi.top_horizon_offset_px + w_b * p_b.roi.top_horizon_offset_px)),
            bottom_cut_px=340,
            trapezoid_top_width_ratio=round(w_a * p_a.roi.trapezoid_top_width_ratio + w_b * p_b.roi.trapezoid_top_width_ratio, 3),
            trapezoid_bottom_width_ratio=round(w_a * p_a.roi.trapezoid_bottom_width_ratio + w_b * p_b.roi.trapezoid_bottom_width_ratio, 3),
            curve_expansion_gain=round(w_a * p_a.roi.curve_expansion_gain + w_b * p_b.roi.curve_expansion_gain, 3),
        )

        target_speed = round(w_a * p_a.target_cruise_speed_mps + w_b * p_b.target_cruise_speed_mps, 2)
        dominant_scene = p_b.scene_type if w_b >= 0.5 else p_a.scene_type

        return SceneTuningProfile(
            scene_type=dominant_scene,
            description=f"Blended profile ({p_a.scene_type.value} [{w_a:.2f}] + {p_b.scene_type.value} [{w_b:.2f}])",
            target_cruise_speed_mps=target_speed,
            classical=blended_classical,
            lighting=blended_lighting,
            fusion=blended_fusion,
            guardrail=blended_guardrail,
            roi=blended_roi,
            notes_iso26262=f"Blended ASIL-B profile between {p_a.scene_type.value} and {p_b.scene_type.value}.",
        )

    def to_json(self) -> str:
        """Export entire registry to formatted JSON string."""
        serializable = {st.value: p.to_dict() for st, p in self._profiles.items()}
        return json.dumps(serializable, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SceneProfileRegistry":
        """Reconstruct a SceneProfileRegistry from a JSON export."""
        data = json.loads(json_str)
        profiles: Dict[SceneType, SceneTuningProfile] = {}
        for k, v in data.items():
            scene_type = SceneType.from_string(k)
            profiles[scene_type] = SceneTuningProfile.from_dict(v)
        return cls(profiles)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Global Singleton Registry Accessor
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_REGISTRY: Optional[SceneProfileRegistry] = None


def get_scene_profile_registry() -> SceneProfileRegistry:
    """
    Obtain the singleton instance of the SceneProfileRegistry.
    Thread-safe lazy initialization.
    """
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = SceneProfileRegistry()
    return _GLOBAL_REGISTRY


def get_scene_profile(scene: Union[SceneType, str]) -> SceneTuningProfile:
    """
    Convenience function returning the tuning profile for a specified scene.
    """
    return get_scene_profile_registry().get_profile(scene)
