"""
LKA Project — Shared Configuration Loader
==========================================
Loads project_config.yaml and resolves all paths relative to the project root.
Import this in any script:

    from src.config import cfg, PATHS
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

# ── Project root is three levels above this file (src/config/__init__.py) ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# Phase 2: Configuration Dataclasses
# ==============================================================================

@dataclass
class UploadConfig:
    staging_dir: str
    max_file_size_mb: int
    max_batch_size_gb: int
    accepted_image_formats: List[str]
    accepted_video_formats: List[str]
    accepted_archive_formats: List[str]
    thumbnail_size: Tuple[int, int]
    thumbnail_quality: int
    thumbnail_dir: str
    cleanup_staging_after_hours: int


@dataclass
class AutoAnnotationConfig:
    enabled: bool
    road_surface: dict
    lane_marking: dict
    confidence: dict
    output: dict


@dataclass
class QualityScoringConfig:
    dimension_mismatch_severity: str
    illegal_class_id_severity: str
    road_coverage_min: float
    road_coverage_max: float
    lane_coverage_min: float
    lane_coverage_max: float
    lane_width_min_px: int
    lane_width_max_px: int
    max_lane_fragments: int
    sky_region_top_fraction: float
    weights: dict
    thresholds: dict


@dataclass
class PipelineConfig:
    batch_id_method: str
    state_file: str
    auto_split_after_annotation: bool
    parallel_workers: int
    progress_poll_interval_sec: float


# ==============================================================================
# Phase 4 Section 5: Optical Extrinsics Dataclass
# ==============================================================================

@dataclass
class CameraExtrinsics:
    """Camera extrinsic calibration parameters relative to vehicle ground plane and front axle."""
    mount_height_m: float = 1.20
    pitch_rad: float = 0.08726646
    roll_rad: float = 0.0
    yaw_rad: float = 0.0
    x_offset_m: float = 0.0
    z_offset_m: float = 0.50
    y_horizon_px: Optional[int] = None
    calibration_method: str = "default"
    timestamp: str = ""

    def compute_horizon_y(self, cy: float = 180.0, fy: float = 500.0) -> int:
        """y_horizon = cy - fy * tan(theta_pitch). Down-positive pitch lowers the horizon in image coords."""
        return int(round(cy - fy * math.tan(self.pitch_rad)))

    def validate(
        self,
        image_height: int = 360,
        cy: float = 180.0,
        fy: float = 500.0,
    ) -> Tuple[bool, List[str]]:
        """Validate extrinsics sanity; returns (is_valid, list_of_issues)."""
        issues = []
        if self.mount_height_m <= 0.0:
            issues.append(f"mount_height_m must be > 0, got {self.mount_height_m}")
        if abs(self.pitch_rad) > math.radians(60.0):
            issues.append(f"pitch_rad exceeds 60 deg ceiling: {math.degrees(self.pitch_rad):.2f} deg")
        if abs(self.roll_rad) >= math.radians(5.0):
            issues.append(f"Warning: absolute roll_rad exceeds 5 deg: {math.degrees(self.roll_rad):.2f} deg")

        y_h = self.compute_horizon_y(cy, fy)
        if y_h < 0 or y_h >= image_height:
            issues.append(f"Computed y_horizon ({y_h}) is outside image bounds [0, {image_height})")

        has_fatal = any(not i.startswith("Warning:") for i in issues)
        return not has_fatal, issues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mount_height_m": float(self.mount_height_m),
            "pitch_rad": float(self.pitch_rad),
            "roll_rad": float(self.roll_rad),
            "yaw_rad": float(self.yaw_rad),
            "x_offset_m": float(self.x_offset_m),
            "z_offset_m": float(self.z_offset_m),
            "y_horizon_px": int(self.y_horizon_px) if self.y_horizon_px is not None else None,
            "calibration_method": str(self.calibration_method),
            "timestamp": str(self.timestamp),
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CameraExtrinsics":
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Extrinsics file not found: {p}")
        with open(p, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


cfg = _load_config()


def resolve(relative_path: str) -> Path:
    """Resolve a config path relative to project root."""
    return PROJECT_ROOT / relative_path


class _Paths:
    """Convenience object: PATHS.raw_frames, PATHS.train, etc."""
    def __init__(self, path_cfg: dict):
        for key, value in path_cfg.items():
            setattr(self, key, resolve(value))

    def all(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


PATHS = _Paths(cfg["paths"])


def ensure_dirs():
    """Create all project directories if they don't exist."""
    for path in PATHS.all().values():
        path.mkdir(parents=True, exist_ok=True)
    # sub-dirs that aren't in config
    for sub in ["images", "masks"]:
        (PATHS.annotated / sub).mkdir(parents=True, exist_ok=True)
        (PATHS.train / sub).mkdir(parents=True, exist_ok=True)
        (PATHS.val / sub).mkdir(parents=True, exist_ok=True)
        (PATHS.test / sub).mkdir(parents=True, exist_ok=True)
    (PATHS.logs / "training").mkdir(parents=True, exist_ok=True)
    (PATHS.logs / "inference").mkdir(parents=True, exist_ok=True)
    (PATHS.results / "plots").mkdir(parents=True, exist_ok=True)
    (PATHS.results / "metrics").mkdir(parents=True, exist_ok=True)


# ==============================================================================
# Phase 5 Section 8: Universal Vehicle Profile System Exports
# ==============================================================================
from src.config.vehicle_profile import (
    ActuationProfile,
    CameraExtrinsicsConfig,
    KinematicsConfig,
    SafetyLimitsConfig,
    VehicleProfile,
    VehicleProfileRegistry,
)

__all__ = [
    "cfg",
    "PATHS",
    "PROJECT_ROOT",
    "resolve",
    "ensure_dirs",
    "UploadConfig",
    "AutoAnnotationConfig",
    "QualityScoringConfig",
    "PipelineConfig",
    "CameraExtrinsics",
    "VehicleProfile",
    "VehicleProfileRegistry",
    "KinematicsConfig",
    "CameraExtrinsicsConfig",
    "ActuationProfile",
    "SafetyLimitsConfig",
]


if __name__ == "__main__":
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config loaded: {list(cfg.keys())}")
    ensure_dirs()
    print("All project directories verified.")
