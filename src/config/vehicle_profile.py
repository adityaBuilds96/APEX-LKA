"""
APEX LKA — Universal Vehicle Profile System
============================================
Allows hot-swapping physical vehicle profiles at runtime without recompiling.
Defines kinematics, camera extrinsics, non-linear actuation LUTs, and safety constraints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class KinematicsConfig:
    """Kinematic parameters of the vehicle chassis."""
    wheelbase_m: float = 1.45
    track_width_m: float = 1.25
    max_steer_angle_deg: float = 32.0
    ackermann_ratio: float = 0.85
    min_turning_radius_m: float = 3.2
    understeer_gradient: float = 0.002

    @property
    def max_steer_angle_rad(self) -> float:
        return math.radians(self.max_steer_angle_deg)


@dataclass
class CameraExtrinsicsConfig:
    """Camera mounting geometry and orientation relative to front axle."""
    mount_height_m: float = 1.25
    pitch_deg: float = -5.0
    roll_deg: float = 0.0
    yaw_deg: float = 0.0
    x_offset_m: float = 0.0
    z_offset_m: float = 0.40
    fov_deg: float = 90.0

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)

    @property
    def roll_rad(self) -> float:
        return math.radians(self.roll_deg)

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)


@dataclass
class ActuationProfile:
    """Non-linear actuation parameters and lookup tables for vehicle steering/throttle/brake."""
    pwm_min: int = 1000
    pwm_max: int = 2000
    pwm_center: int = 1500
    center_trim: int = 0
    direction_sign: int = 1  # 1 or -1
    rate_limit_deg_per_sec: float = 180.0
    # LUT mappings: list of [input_val, output_pwm] pairs
    lut_steering: List[List[float]] = field(default_factory=lambda: [
        [-32.0, 1000.0],
        [-16.0, 1250.0],
        [0.0, 1500.0],
        [16.0, 1750.0],
        [32.0, 2000.0],
    ])
    lut_throttle: List[List[float]] = field(default_factory=lambda: [
        [0.0, 1500.0],
        [0.5, 1750.0],
        [1.0, 2000.0],
    ])
    lut_brake: List[List[float]] = field(default_factory=lambda: [
        [0.0, 1500.0],
        [0.5, 1250.0],
        [1.0, 1000.0],
    ])

    def map_steering_deg_to_pwm(self, angle_deg: float) -> int:
        """Map requested steering angle in degrees to output PWM pulse width (us)."""
        sign_angle = angle_deg * self.direction_sign
        pwm_val = self._interpolate_lut(sign_angle, self.lut_steering, default=float(self.pwm_center))
        pwm_val += self.center_trim
        return int(round(max(float(self.pwm_min), min(float(self.pwm_max), pwm_val))))

    def map_throttle_to_pwm(self, norm_throttle: float) -> int:
        """Map normalized throttle [0.0, 1.0] to output PWM pulse width (us)."""
        clamped = max(0.0, min(1.0, float(norm_throttle)))
        pwm_val = self._interpolate_lut(clamped, self.lut_throttle, default=float(self.pwm_center))
        return int(round(max(float(self.pwm_min), min(float(self.pwm_max), pwm_val))))

    def map_brake_to_pwm(self, norm_brake: float) -> int:
        """Map normalized brake [0.0, 1.0] to output PWM pulse width (us)."""
        clamped = max(0.0, min(1.0, float(norm_brake)))
        pwm_val = self._interpolate_lut(clamped, self.lut_brake, default=float(self.pwm_center))
        return int(round(max(float(self.pwm_min), min(float(self.pwm_max), pwm_val))))

    @staticmethod
    def _interpolate_lut(val: float, lut: List[List[float]], default: float = 1500.0) -> float:
        if not lut:
            return default
        sorted_lut = sorted(lut, key=lambda p: p[0])
        if val <= sorted_lut[0][0]:
            return sorted_lut[0][1]
        if val >= sorted_lut[-1][0]:
            return sorted_lut[-1][1]
        for i in range(len(sorted_lut) - 1):
            x0, y0 = sorted_lut[i]
            x1, y1 = sorted_lut[i + 1]
            if x0 <= val <= x1:
                if abs(x1 - x0) < 1e-7:
                    return y0
                ratio = (val - x0) / (x1 - x0)
                return y0 + ratio * (y1 - y0)
        return default


@dataclass
class SafetyLimitsConfig:
    """Rigid safety envelope and operational limits for the vehicle profile."""
    max_speed_mps: float = 6.94  # ~25 km/h
    max_accel_mpss: float = 2.5
    max_decel_mpss: float = 4.5
    emergency_decel_mpss: float = 6.0
    min_standstill_distance_m: float = 1.5
    lateral_error_limit_m: float = 1.2
    max_steering_angle_deg: float = 32.0


@dataclass
class VehicleProfile:
    """Complete specification of a vehicle platform for universal deployment."""
    name: str
    vehicle_type: str
    description: str = ""
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)
    camera_extrinsics: CameraExtrinsicsConfig = field(default_factory=CameraExtrinsicsConfig)
    actuation: ActuationProfile = field(default_factory=ActuationProfile)
    safety_limits: SafetyLimitsConfig = field(default_factory=SafetyLimitsConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vehicle_type": self.vehicle_type,
            "description": self.description,
            "kinematics": {
                "wheelbase_m": self.kinematics.wheelbase_m,
                "track_width_m": self.kinematics.track_width_m,
                "max_steer_angle_deg": self.kinematics.max_steer_angle_deg,
                "ackermann_ratio": self.kinematics.ackermann_ratio,
                "min_turning_radius_m": self.kinematics.min_turning_radius_m,
                "understeer_gradient": self.kinematics.understeer_gradient,
            },
            "camera_extrinsics": {
                "mount_height_m": self.camera_extrinsics.mount_height_m,
                "pitch_deg": self.camera_extrinsics.pitch_deg,
                "roll_deg": self.camera_extrinsics.roll_deg,
                "yaw_deg": self.camera_extrinsics.yaw_deg,
                "x_offset_m": self.camera_extrinsics.x_offset_m,
                "z_offset_m": self.camera_extrinsics.z_offset_m,
                "fov_deg": self.camera_extrinsics.fov_deg,
            },
            "actuation": {
                "pwm_min": self.actuation.pwm_min,
                "pwm_max": self.actuation.pwm_max,
                "pwm_center": self.actuation.pwm_center,
                "center_trim": self.actuation.center_trim,
                "direction_sign": self.actuation.direction_sign,
                "rate_limit_deg_per_sec": self.actuation.rate_limit_deg_per_sec,
                "lut_steering": self.actuation.lut_steering,
                "lut_throttle": self.actuation.lut_throttle,
                "lut_brake": self.actuation.lut_brake,
            },
            "safety_limits": {
                "max_speed_mps": self.safety_limits.max_speed_mps,
                "max_accel_mpss": self.safety_limits.max_accel_mpss,
                "max_decel_mpss": self.safety_limits.max_decel_mpss,
                "emergency_decel_mpss": self.safety_limits.emergency_decel_mpss,
                "min_standstill_distance_m": self.safety_limits.min_standstill_distance_m,
                "lateral_error_limit_m": self.safety_limits.lateral_error_limit_m,
                "max_steering_angle_deg": self.safety_limits.max_steering_angle_deg,
            },
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VehicleProfile":
        kin_data = data.get("kinematics", {})
        cam_data = data.get("camera_extrinsics", {})
        act_data = data.get("actuation", {})
        saf_data = data.get("safety_limits", {})

        return cls(
            name=data.get("name", "unknown"),
            vehicle_type=data.get("vehicle_type", "generic"),
            description=data.get("description", ""),
            kinematics=KinematicsConfig(
                wheelbase_m=float(kin_data.get("wheelbase_m", 1.45)),
                track_width_m=float(kin_data.get("track_width_m", 1.25)),
                max_steer_angle_deg=float(kin_data.get("max_steer_angle_deg", 32.0)),
                ackermann_ratio=float(kin_data.get("ackermann_ratio", 0.85)),
                min_turning_radius_m=float(kin_data.get("min_turning_radius_m", 3.2)),
                understeer_gradient=float(kin_data.get("understeer_gradient", 0.002)),
            ),
            camera_extrinsics=CameraExtrinsicsConfig(
                mount_height_m=float(cam_data.get("mount_height_m", 1.25)),
                pitch_deg=float(cam_data.get("pitch_deg", -5.0)),
                roll_deg=float(cam_data.get("roll_deg", 0.0)),
                yaw_deg=float(cam_data.get("yaw_deg", 0.0)),
                x_offset_m=float(cam_data.get("x_offset_m", 0.0)),
                z_offset_m=float(cam_data.get("z_offset_m", 0.40)),
                fov_deg=float(cam_data.get("fov_deg", 90.0)),
            ),
            actuation=ActuationProfile(
                pwm_min=int(act_data.get("pwm_min", 1000)),
                pwm_max=int(act_data.get("pwm_max", 2000)),
                pwm_center=int(act_data.get("pwm_center", 1500)),
                center_trim=int(act_data.get("center_trim", 0)),
                direction_sign=int(act_data.get("direction_sign", 1)),
                rate_limit_deg_per_sec=float(act_data.get("rate_limit_deg_per_sec", 180.0)),
                lut_steering=act_data.get("lut_steering", [
                    [-32.0, 1000.0], [0.0, 1500.0], [32.0, 2000.0]
                ]),
                lut_throttle=act_data.get("lut_throttle", [
                    [0.0, 1500.0], [1.0, 2000.0]
                ]),
                lut_brake=act_data.get("lut_brake", [
                    [0.0, 1500.0], [1.0, 1000.0]
                ]),
            ),
            safety_limits=SafetyLimitsConfig(
                max_speed_mps=float(saf_data.get("max_speed_mps", 6.94)),
                max_accel_mpss=float(saf_data.get("max_accel_mpss", 2.5)),
                max_decel_mpss=float(saf_data.get("max_decel_mpss", 4.5)),
                emergency_decel_mpss=float(saf_data.get("emergency_decel_mpss", 6.0)),
                min_standstill_distance_m=float(saf_data.get("min_standstill_distance_m", 1.5)),
                lateral_error_limit_m=float(saf_data.get("lateral_error_limit_m", 1.2)),
                max_steering_angle_deg=float(saf_data.get("max_steering_angle_deg", 32.0)),
            ),
        )


class VehicleProfileRegistry:
    """
    Dynamic vehicle profile manager.
    Supports instant hot-swapping at runtime with zero recompilation.
    """

    def __init__(self, profile_dir: Optional[Union[str, Path]] = None):
        if profile_dir is None:
            # Default to vehicle_profiles in workspace root
            root = Path(__file__).resolve().parent.parent.parent
            self.profile_dir = root / "vehicle_profiles"
        else:
            self.profile_dir = Path(profile_dir)

        self._active_profile: Optional[VehicleProfile] = None
        self._cache: Dict[str, VehicleProfile] = {}
        self._subscribers: List[Callable[[VehicleProfile], None]] = []

    def list_available_profiles(self) -> List[str]:
        """List all available YAML profiles in the profile directory."""
        if not self.profile_dir.exists():
            return []
        return sorted([p.stem for p in self.profile_dir.glob("*.yaml")])

    def load_profile(self, profile_name_or_path: Union[str, Path]) -> VehicleProfile:
        """
        Load a vehicle profile by name (e.g. 'baja_2027') or full file path.
        Caches loaded profiles for instant retrieval.
        """
        raw_path = Path(profile_name_or_path)
        if raw_path.exists() and raw_path.is_file():
            target_path = raw_path
            key = target_path.stem
        else:
            name = str(profile_name_or_path).removesuffix(".yaml")
            target_path = self.profile_dir / f"{name}.yaml"
            key = name

        if not target_path.exists():
            raise FileNotFoundError(f"Vehicle profile not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        profile = VehicleProfile.from_dict(data)
        self._cache[key] = profile

        if self._active_profile is None:
            self._active_profile = profile

        return profile

    def switch_profile(self, profile_name_or_path: Union[str, Path]) -> VehicleProfile:
        """
        Hot-swap the active vehicle profile at runtime.
        Notifies all registered subscribers of the transition.
        """
        new_profile = self.load_profile(profile_name_or_path)
        self._active_profile = new_profile

        # Notify subscribers
        for cb in self._subscribers:
            try:
                cb(new_profile)
            except Exception as e:
                # Log or swallow to prevent subscriber failure from stopping execution
                print(f"[VehicleProfileRegistry] Subscriber callback error: {e}")

        return new_profile

    def get_active_profile(self) -> VehicleProfile:
        """Return currently active vehicle profile. Defaults to baja_2027 if none loaded."""
        if self._active_profile is None:
            avail = self.list_available_profiles()
            default_name = "baja_2027" if "baja_2027" in avail else (avail[0] if avail else "default")
            if default_name in avail:
                return self.load_profile(default_name)
            # Create a fallback default profile
            self._active_profile = VehicleProfile(name="default", vehicle_type="generic")
        return self._active_profile

    def register_callback(self, callback: Callable[[VehicleProfile], None]) -> None:
        """Register a listener for profile switch events."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unregister_callback(self, callback: Callable[[VehicleProfile], None]) -> None:
        """Unregister a listener."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    @staticmethod
    def validate_profile(profile: VehicleProfile) -> Tuple[bool, List[str]]:
        """Validate safety and kinematic sanity of a profile."""
        issues: List[str] = []
        if profile.kinematics.wheelbase_m <= 0.0:
            issues.append(f"Wheelbase must be > 0, got {profile.kinematics.wheelbase_m}")
        if profile.kinematics.track_width_m <= 0.0:
            issues.append(f"Track width must be > 0, got {profile.kinematics.track_width_m}")
        if profile.camera_extrinsics.mount_height_m <= 0.0:
            issues.append(f"Camera mount height must be > 0, got {profile.camera_extrinsics.mount_height_m}")
        if profile.actuation.pwm_min >= profile.actuation.pwm_max:
            issues.append("actuation.pwm_min must be strictly less than pwm_max")
        if profile.safety_limits.max_speed_mps <= 0.0:
            issues.append("Safety limit max_speed_mps must be positive")

        return (len(issues) == 0, issues)
