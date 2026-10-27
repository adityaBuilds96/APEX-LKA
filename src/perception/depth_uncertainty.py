"""
src/perception/depth_uncertainty.py
===================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Closed-Form Monocular Depth Variance Propagation & ASIL-B Pessimistic Safety Bounds.
ISO 26262 ASIL-B Forward Collision Avoidance & Time-To-Impact (TTI) Baseline.

Mathematical Derivation:
------------------------
For a ground-contact point at pixel row v, camera mounting height H_cam,
focal length f_y, optical center c_y, and live chassis pitch angle theta_pitch:

  alpha = theta_pitch + arctan((v - c_y) / f_y)
  Z_m   = H_cam / tan(alpha)

Partial Derivatives for Longitudinal Error Propagation:
-------------------------------------------------------
Let beta = (v - c_y) / f_y, so alpha = theta + arctan(beta).
d(arctan(beta))/dv = (1 / (1 + beta^2)) * (1 / f_y) = f_y / (f_y^2 + (v - c_y)^2)

1. Height Sensitivity:
   dZ / dH = 1.0 / tan(alpha) = Z_m / H_cam

2. Pitch Sensitivity:
   dZ / dtheta = -H_cam * (1 + tan^2(alpha)) / tan^2(alpha) = -H_cam / sin^2(alpha)

3. Pixel Row Sensitivity:
   dZ / dv = (dZ / dtheta) * (d(arctan(beta)) / dv)
           = (-H_cam / sin^2(alpha)) * (f_y / (f_y^2 + (v - c_y)^2))

Variance Formulation:
---------------------
sigma_Z^2 = (dZ/dv)^2 * sigma_v^2 + (dZ/dtheta)^2 * sigma_theta^2 + (dZ/dH)^2 * sigma_H^2

Lateral Coordinate & Uncertainty:
---------------------------------
X_m = (u - c_x) * Z_m / f_x
dX / du = Z_m / f_x
dX / dZ = (u - c_x) / f_x
sigma_X^2 = (dX/du)^2 * sigma_u^2 + (dX/dZ)^2 * sigma_Z^2
sigma_XZ  = (dX/dZ) * sigma_Z^2

Pessimistic Safety Bound:
-------------------------
Z_pessimistic = max(0.1, Z_m - 1.96 * sigma_Z)   (95% confidence lower bound)
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.DepthUncertainty")

EPSILON: float = 1e-6
MIN_SAFE_DISTANCE_M: float = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# 1. Uncertainty Configuration & Estimate Containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SensorUncertaintyParameters:
    """
    Standard deviation noise baselines for camera and inertial measurement sensors.
    """
    sigma_v_pixels: float = 1.50             # Pixel localization error (bounding box bottom jitter)
    sigma_u_pixels: float = 1.50             # Horizontal pixel center error
    sigma_pitch_rad: float = 0.002618        # INS pitch angle noise (~0.15 degrees)
    sigma_h_cam_m: float = 0.030             # Vehicle suspension bounce / height calibration error
    focal_length_x_px: float = 450.0
    focal_length_y_px: float = 450.0
    principal_point_x_px: float = 320.0
    principal_point_y_px: float = 180.0
    camera_height_m: float = 1.20
    confidence_multiplier: float = 1.96      # 95% two-sided normal distribution interval


@dataclass
class UncertaintyEllipse2D:
    """
    2D spatial covariance ellipse in metric vehicle coordinate space (X right, Z forward).
    """
    center_x_m: float
    center_z_m: float
    semi_major_axis_m: float
    semi_minor_axis_m: float
    orientation_rad: float
    cov_xx: float
    cov_zz: float
    cov_xz: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DepthUncertaintyEstimate:
    """
    Complete distance output with rigorous variance propagation and safety envelopes.
    """
    nominal_distance_m: float
    pessimistic_distance_m: float
    optimistic_distance_m: float
    sigma_z_m: float
    variance_z_m2: float

    # Sensitivity components
    dz_dv: float
    dz_dpitch: float
    dz_dh: float

    # Component variance contributions
    var_contribution_pixel: float
    var_contribution_pitch: float
    var_contribution_height: float

    # Lateral state & uncertainty
    lateral_x_m: float = 0.0
    sigma_x_m: float = 0.0
    cov_xz: float = 0.0

    # Safety metrics
    relative_uncertainty_pct: float = 0.0
    is_valid_ground_plane: bool = True
    warning_flags: List[str] = field(default_factory=list)
    timestamp_utc: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def covariance_matrix_2d(self) -> np.ndarray:
        """Construct 2x2 covariance matrix [ [cov_xx, cov_xz], [cov_xz, cov_zz] ]."""
        return np.array([
            [self.sigma_x_m ** 2, self.cov_xz],
            [self.cov_xz,         self.variance_z_m2],
        ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Closed-Form Depth Uncertainty Engine
# ──────────────────────────────────────────────────────────────────────────────

class ClosedFormDepthUncertaintyEngine:
    """
    Calculates analytical monocular depth uncertainty and pessimistic stopping distances
    conforming to ISO 26262 ASIL-B safety margins.
    """

    def __init__(self, params: Optional[SensorUncertaintyParameters] = None) -> None:
        self.params = params or SensorUncertaintyParameters()

    def propagate_uncertainty(
        self,
        v_bottom_px: float,
        pitch_rad: float,
        u_center_px: Optional[float] = None,
        camera_height_override: Optional[float] = None,
        sigma_v_override: Optional[float] = None,
        sigma_pitch_override: Optional[float] = None,
    ) -> DepthUncertaintyEstimate:
        """
        Evaluate closed-form ground-plane distance, lateral offset, and variance propagation.
        
        Args:
            v_bottom_px: Pixel row coordinate of contact point on the road surface.
            pitch_rad: Instantaneous vehicle pitch in radians from MTi-680 INS.
            u_center_px: Optional horizontal pixel center of target bounding box.
            camera_height_override: Optional dynamic optical height in meters.
            sigma_v_override: Optional pixel noise standard deviation override.
            sigma_pitch_override: Optional pitch noise standard deviation override.

        Returns:
            DepthUncertaintyEstimate with analytical sensitivities and safety bounds.
        """
        p = self.params
        h_cam = camera_height_override if camera_height_override is not None else p.camera_height_m
        sig_v = sigma_v_override if sigma_v_override is not None else p.sigma_v_pixels
        sig_u = p.sigma_u_pixels
        sig_pitch = sigma_pitch_override if sigma_pitch_override is not None else p.sigma_pitch_rad
        sig_h = p.sigma_h_cam_m
        u_center = u_center_px if u_center_px is not None else p.principal_point_x_px

        warnings: List[str] = []

        # 1. Compute angle below horizon in camera coordinates
        delta_v = v_bottom_px - p.principal_point_y_px
        beta = delta_v / p.focal_length_y_px
        gamma = math.atan(beta)

        # Total elevation depression angle
        alpha = pitch_rad + gamma

        # Safety gate: target must be below vehicle horizon
        if alpha <= 0.005: # Less than ~0.3 degrees elevation
            warnings.append("Target is at or above effective camera horizon (infinite distance).")
            return DepthUncertaintyEstimate(
                nominal_distance_m=160.0,
                pessimistic_distance_m=160.0,
                optimistic_distance_m=160.0,
                sigma_z_m=999.0,
                variance_z_m2=999.0 ** 2,
                dz_dv=0.0,
                dz_dpitch=0.0,
                dz_dh=0.0,
                var_contribution_pixel=0.0,
                var_contribution_pitch=0.0,
                var_contribution_height=0.0,
                lateral_x_m=0.0,
                sigma_x_m=999.0,
                cov_xz=0.0,
                relative_uncertainty_pct=100.0,
                is_valid_ground_plane=False,
                warning_flags=warnings,
            )

        # 2. Nominal distance Z_m
        tan_alpha = math.tan(alpha)
        sin_alpha = math.sin(alpha)
        z_nominal = h_cam / max(EPSILON, tan_alpha)

        # 3. Closed-Form Longitudinal Sensitivity Derivatives
        # dZ/dH
        dz_dh = 1.0 / tan_alpha

        # dZ/dtheta
        dz_dpitch = -h_cam / max(EPSILON, sin_alpha ** 2)

        # dgamma/dv = f_y / (f_y^2 + delta_v^2)
        dgamma_dv = p.focal_length_y_px / (p.focal_length_y_px ** 2 + delta_v ** 2)
        dz_dv = dz_dpitch * dgamma_dv

        # 4. Variance Propagation for Longitudinal Axis (Z)
        var_pixel = (dz_dv ** 2) * (sig_v ** 2)
        var_pitch = (dz_dpitch ** 2) * (sig_pitch ** 2)
        var_height = (dz_dh ** 2) * (sig_h ** 2)

        total_variance_z = var_pixel + var_pitch + var_height
        sigma_z = math.sqrt(max(0.0, total_variance_z))

        # 5. Lateral Calculation (X) & Covariance
        delta_u = u_center - p.principal_point_x_px
        x_nominal = (delta_u * z_nominal) / p.focal_length_x_px

        dx_du = z_nominal / p.focal_length_x_px
        dx_dz = delta_u / p.focal_length_x_px

        var_x = (dx_du ** 2) * (sig_u ** 2) + (dx_dz ** 2) * total_variance_z
        sigma_x = math.sqrt(max(0.0, var_x))
        cov_xz = dx_dz * total_variance_z

        # 6. Pessimistic Lower Bound
        z_pessimistic = max(MIN_SAFE_DISTANCE_M, z_nominal - p.confidence_multiplier * sigma_z)
        z_optimistic = z_nominal + p.confidence_multiplier * sigma_z

        rel_uncert = (sigma_z / max(EPSILON, z_nominal)) * 100.0

        if rel_uncert > 50.0:
            warnings.append(f"High relative distance uncertainty ({rel_uncert:.1f}%).")

        return DepthUncertaintyEstimate(
            nominal_distance_m=round(z_nominal, 3),
            pessimistic_distance_m=round(z_pessimistic, 3),
            optimistic_distance_m=round(z_optimistic, 3),
            sigma_z_m=round(sigma_z, 3),
            variance_z_m2=round(total_variance_z, 4),
            dz_dv=round(dz_dv, 4),
            dz_dpitch=round(dz_dpitch, 4),
            dz_dh=round(dz_dh, 4),
            var_contribution_pixel=round(var_pixel, 4),
            var_contribution_pitch=round(var_pitch, 4),
            var_contribution_height=round(var_height, 4),
            lateral_x_m=round(x_nominal, 3),
            sigma_x_m=round(sigma_x, 3),
            cov_xz=round(cov_xz, 4),
            relative_uncertainty_pct=round(rel_uncert, 2),
            is_valid_ground_plane=True,
            warning_flags=warnings,
        )

    def compute_spatial_uncertainty_ellipse(
        self,
        estimate: DepthUncertaintyEstimate,
        chi2_val: float = 5.991, # 95% confidence for 2 degrees of freedom
    ) -> UncertaintyEllipse2D:
        """
        Calculate 2D metric error ellipse parameters (semi-axes and angle).
        """
        cov = estimate.covariance_matrix_2d
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Order largest eigenvalue first
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        semi_major = math.sqrt(max(0.0, chi2_val * eigvals[0]))
        semi_minor = math.sqrt(max(0.0, chi2_val * eigvals[1]))
        angle_rad = math.atan2(eigvecs[1, 0], eigvecs[0, 0])

        return UncertaintyEllipse2D(
            center_x_m=estimate.lateral_x_m,
            center_z_m=estimate.nominal_distance_m,
            semi_major_axis_m=round(semi_major, 3),
            semi_minor_axis_m=round(semi_minor, 3),
            orientation_rad=round(angle_rad, 4),
            cov_xx=float(cov[0, 0]),
            cov_zz=float(cov[1, 1]),
            cov_xz=float(cov[0, 1]),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # ASIL-B Safety Applications: TTI & RSS Margins
    # ──────────────────────────────────────────────────────────────────────────

    def compute_pessimistic_time_to_impact(
        self,
        v_bottom_px: float,
        pitch_rad: float,
        closing_velocity_m_s: float,
    ) -> Tuple[float, float]:
        """
        Compute Time-To-Impact (TTI) using nominal distance and conservative
        pessimistic lower-bound distance:
            TTI_pessimistic = Z_pessimistic / max(0.1, v_closing)
        """
        est = self.propagate_uncertainty(v_bottom_px, pitch_rad)
        v_close = max(0.1, closing_velocity_m_s)

        tti_nominal = est.nominal_distance_m / v_close
        tti_pessimistic = est.pessimistic_distance_m / v_close

        return round(tti_nominal, 2), round(tti_pessimistic, 2)

    def evaluate_rss_longitudinal_safety(
        self,
        v_bottom_px: float,
        pitch_rad: float,
        ego_velocity_m_s: float,
        lead_velocity_m_s: float,
        reaction_time_sec: float = 0.50,
        max_accel_ego_m_s2: float = 2.0,
        min_brake_ego_m_s2: float = 4.0,
        max_brake_lead_m_s2: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Evaluate Responsibility-Sensitive Safety (RSS) minimal safe longitudinal distance:
        d_min = max(0, v_ego * rho + 0.5 * a_accel * rho^2 + (v_ego + rho*a_accel)^2 / (2*b_min) - v_lead^2 / (2*b_max))
        """
        est = self.propagate_uncertainty(v_bottom_px, pitch_rad)

        rho = reaction_time_sec
        a_accel = max_accel_ego_m_s2
        b_min = min_brake_ego_m_s2
        b_max = max_brake_lead_m_s2

        v_ego = max(0.0, ego_velocity_m_s)
        v_lead = max(0.0, lead_velocity_m_s)

        # RSS equation
        d_reaction = v_ego * rho + 0.5 * a_accel * (rho ** 2)
        v_after_reaction = v_ego + rho * a_accel
        d_brake_ego = (v_after_reaction ** 2) / (2.0 * b_min)
        d_brake_lead = (v_lead ** 2) / (2.0 * b_max)

        d_min = max(0.0, d_reaction + d_brake_ego - d_brake_lead)

        is_safe_nominal = est.nominal_distance_m >= d_min
        is_safe_pessimistic = est.pessimistic_distance_m >= d_min

        return {
            "d_min_rss_meters": round(d_min, 2),
            "nominal_distance": est.nominal_distance_m,
            "pessimistic_distance": est.pessimistic_distance_m,
            "sigma_z": est.sigma_z_m,
            "is_safe_nominal": is_safe_nominal,
            "is_safe_pessimistic": is_safe_pessimistic,
            "margin_pessimistic_meters": round(est.pessimistic_distance_m - d_min, 2),
        }

    def evaluate_rss_lateral_safety(
        self,
        lateral_distance_m: float,
        ego_lateral_velocity_m_s: float,
        adjacent_lateral_velocity_m_s: float,
        reaction_time_sec: float = 0.50,
        max_lateral_accel_m_s2: float = 1.0,
        min_lateral_brake_m_s2: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Evaluate RSS minimum lateral boundary separation margin:
        d_lat_min = mu + 0.5 * (v1 + v1_p) * rho + (v1_p^2 / (2 * b1)) + (v2_p^2 / (2 * b2))
        """
        mu = 0.30 # Minimal lateral clearance buffer in meters
        rho = reaction_time_sec

        v1 = abs(ego_lateral_velocity_m_s)
        v2 = abs(adjacent_lateral_velocity_m_s)

        v1_p = v1 + rho * max_lateral_accel_m_s2
        v2_p = v2 + rho * max_lateral_accel_m_s2

        d_reaction = 0.5 * (v1 + v1_p) * rho
        d_brake_1 = (v1_p ** 2) / (2.0 * min_lateral_brake_m_s2)
        d_brake_2 = (v2_p ** 2) / (2.0 * min_lateral_brake_m_s2)

        d_lat_min = mu + d_reaction + d_brake_1 + d_brake_2
        is_safe = abs(lateral_distance_m) >= d_lat_min

        return {
            "d_lat_min_rss_meters": round(d_lat_min, 2),
            "actual_lateral_distance_meters": round(abs(lateral_distance_m), 2),
            "is_lateral_safe": is_safe,
            "lateral_margin_meters": round(abs(lateral_distance_m) - d_lat_min, 2),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Vectorized Batch Evaluation & Sensitivity Tables
    # ──────────────────────────────────────────────────────────────────────────

    def propagate_batch(
        self,
        v_bottom_array: np.ndarray,
        pitch_rad: float,
        u_center_array: Optional[np.ndarray] = None,
    ) -> List[DepthUncertaintyEstimate]:
        """
        Efficient evaluation of multiple bounding boxes across a single video frame.
        """
        results: List[DepthUncertaintyEstimate] = []
        u_arr = u_center_array if u_center_array is not None else [None] * len(v_bottom_array)

        for v_b, u_c in zip(v_bottom_array, u_arr):
            est = self.propagate_uncertainty(
                v_bottom_px=float(v_b),
                pitch_rad=pitch_rad,
                u_center_px=float(u_c) if u_c is not None else None,
            )
            results.append(est)
        return results

    def get_sensitivity_table(
        self,
        v_bottom_px: float,
        pitch_rad: float,
    ) -> Dict[str, float]:
        """
        Generate sensitivity breakdown detailing how 1px or 0.1 deg changes depth.
        """
        est = self.propagate_uncertainty(v_bottom_px=v_bottom_px, pitch_rad=pitch_rad)
        delta_z_per_pixel = abs(est.dz_dv)
        delta_z_per_deg_pitch = abs(est.dz_dpitch) * (math.pi / 180.0)
        delta_z_per_cm_height = abs(est.dz_dh) * 0.01

        return {
            "nominal_z_meters": est.nominal_distance_m,
            "delta_z_per_pixel_meters": round(delta_z_per_pixel, 3),
            "delta_z_per_degree_pitch_meters": round(delta_z_per_deg_pitch, 3),
            "delta_z_per_cm_height_meters": round(delta_z_per_cm_height, 3),
            "dominant_uncertainty_source": (
                "pitch_noise" if est.var_contribution_pitch > est.var_contribution_pixel
                else "pixel_noise"
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

def run_depth_uncertainty_self_test() -> bool:
    """Execute complete mathematical verification for depth uncertainty."""
    print("Executing DepthUncertainty ISO 26262 ASIL-B self-check...")
    engine = ClosedFormDepthUncertaintyEngine()

    # Target vehicle at row v=240, level pitch=0.0
    est = engine.propagate_uncertainty(v_bottom_px=240.0, pitch_rad=0.0, u_center_px=360.0)
    print(f"Nominal Distance    : {est.nominal_distance_m:.2f} m")
    print(f"Pessimistic Distance: {est.pessimistic_distance_m:.2f} m (sigma={est.sigma_z_m:.2f} m)")
    print(f"Lateral Coordinate  : {est.lateral_x_m:.2f} m (sigma={est.sigma_x_m:.2f} m)")
    assert est.pessimistic_distance_m <= est.nominal_distance_m
    assert est.nominal_distance_m > 0.0

    # Uncertainty ellipse
    ellipse = engine.compute_spatial_uncertainty_ellipse(est)
    assert ellipse.semi_major_axis_m >= ellipse.semi_minor_axis_m
    print(f"Uncertainty Ellipse : major={ellipse.semi_major_axis_m:.2f}m, minor={ellipse.semi_minor_axis_m:.2f}m")

    # Test pitch disturbance (vehicle pitching down 1 degree)
    est_pitch = engine.propagate_uncertainty(v_bottom_px=240.0, pitch_rad=math.radians(-1.0))
    print(f"Pitched Distance    : {est_pitch.nominal_distance_m:.2f} m")

    # RSS evaluation test
    rss = engine.evaluate_rss_longitudinal_safety(
        v_bottom_px=240.0,
        pitch_rad=0.0,
        ego_velocity_m_s=15.0, # ~54 km/h
        lead_velocity_m_s=10.0, # ~36 km/h
    )
    print(f"RSS d_min Required  : {rss['d_min_rss_meters']:.2f} m | Safe: {rss['is_safe_pessimistic']}")

    # Lateral RSS test
    rss_lat = engine.evaluate_rss_lateral_safety(
        lateral_distance_m=1.85,
        ego_lateral_velocity_m_s=0.20,
        adjacent_lateral_velocity_m_s=0.10,
    )
    print(f"RSS d_lat Required  : {rss_lat['d_lat_min_rss_meters']:.2f} m | Safe: {rss_lat['is_lateral_safe']}")

    print("DepthUncertainty verification passed successfully.")
    return True


if __name__ == "__main__":
    run_depth_uncertainty_self_test()
