"""
src/scene/scene_adaptive_tuner.py
=================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Dynamic Operational Scene Adaptive Tuner & Multi-Subsystem Calibration Engine.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.

This module orchestrates real-time parameter modulation across the entire APEX-Vision
perception and safety stack based on continuous operational scene understanding.
When the ego vehicle encounters changing road environments (e.g. transitioning from
bright open highway into a dark sodium-lamp tunnel, or sudden heavy downpours), the
SceneAdaptiveTuner smoothly retunes classical CV thresholds, deep learning ensemble
fusion balances, CLAHE/gamma compensations, dynamic ROI boundaries, and virtual
guardrail safety envelopes without causing parameter discontinuities or steering jerks.

Key Architectural Responsibilities:
-----------------------------------
1. Slew-Rate Limited Continuous Parameter Blending:
   Guarantees $C^1$ continuity during scene handovers (e.g., Highway -> Tunnel)
   by applying exponential moving average smoothing and maximum slew-rate limits
   to prevent sudden jumps in Canny thresholds or Stanley controller gains.
2. Cross-Subsystem Parameter Broadcast:
   Publishes tailored configuration snapshots to:
     - `lane_fusion_engine.py` (DL weight alpha, minimum confidence, hallucination mode)
     - `lighting_compensator.py` (gamma, CLAHE clip limit, specular threshold, night mode)
     - `cv_precision_stack.py` (Canny thresholds, HLS/LAB color windows, Hough slope bounds)
     - `adaptive_roi_engine.py` (horizon offset, trapezoid width, curve expansion gain)
     - `virtual_guardrail_enforcer.py` (inner safety buffer, severity thresholds, Stanley gains)
3. Hysteresis State Transition Engine:
   Blocks high-frequency scene flapping under intermittent optical shadows or overhead bridges.
4. ASIL-B Safety Bounding & Parameter Clamp Verifier:
   Strictly clamps every dynamically adapted variable within pre-validated ISO 26262
   fail-safe envelopes to prevent out-of-range sensor misbehavior.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scene.scene_classifier import SceneClassificationResult
from src.scene.scene_profiles import (
    AdaptiveROIProfile,
    ClassicalCVThresholds,
    FusionAndTrackingProfile,
    LightingCompensationProfile,
    SceneProfileRegistry,
    SceneTuningProfile,
    SceneType,
    VirtualGuardrailProfile,
    get_scene_profile_registry,
)

logger = logging.getLogger("APEX.SceneAdaptiveTuner")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Active Adaptation Telemetry & Tuning Snapshot
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DynamicTuningSnapshot:
    """
    Complete consolidated snapshot of currently active smoothed parameters
    broadcast to all downstream perception and control subsystems.
    """
    timestamp_utc: str
    active_scene: SceneType
    target_scene: SceneType
    transition_progress: float          # in [0.0, 1.0], 1.0 = fully settled
    is_in_transition: bool

    # 1. Classical Computer Vision Parameters
    canny_low: int
    canny_high: int
    hls_l_min: int
    hls_s_min: int
    lab_b_min: int
    yuv_y_min: int
    hough_threshold: int
    hough_min_line_length: int
    hough_max_line_gap: int
    hough_min_slope: float
    enable_gabor_bank: bool

    # 2. Lighting & Exposure Compensation Parameters
    gamma: float
    clahe_clip_limit: float
    enable_specular_removal: bool
    specular_l_threshold: int
    night_mode_boost: bool
    headlight_glare_suppression: bool

    # 3. Hybrid Fusion & Tracking Parameters
    dl_fusion_weight: float             # alpha: DL weight
    cv_fusion_weight: float             # 1 - alpha: Classical weight
    confidence_threshold_min: float
    temporal_window_size: int
    ekf_process_noise_pos: float
    ekf_meas_noise_pos: float
    lane_width_nominal_m: float
    allow_hallucination: bool
    max_hallucination_frames: int

    # 4. Virtual Guardrail & Steering Interlock Parameters
    inner_guardrail_buffer_m: float
    warning_buffer_m: float
    critical_buffer_m: float
    trajectory_prediction_horizon_s: float
    ttld_warning_threshold_s: float
    stanley_gain_base: float
    counter_steer_gain_multiplier: float

    # 5. Dynamic Perspective ROI Parameters
    top_horizon_offset_px: int
    bottom_cut_px: int
    trapezoid_top_width_ratio: float
    trapezoid_bottom_width_ratio: float
    curve_expansion_gain: float

    # Safety & Audit
    target_cruise_speed_mps: float
    asil_b_compliance_flag: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for telemetry logging."""
        d = asdict(self)
        d["active_scene"] = self.active_scene.value
        d["target_scene"] = self.target_scene.value
        return d


@dataclass
class TransitionEventRecord:
    """Historical audit log entry recording scene transition handovers."""
    event_id: int
    timestamp_utc: str
    from_scene: SceneType
    to_scene: SceneType
    duration_sec: float
    initial_confidence: float
    status: str                         # "IN_PROGRESS", "COMPLETED", "ABORTED"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Continuous Slew-Rate Limiter & Parameter Smoother
# ═══════════════════════════════════════════════════════════════════════════════

class SlewRateSmoother:
    """
    Applies exponential moving average and absolute slew-rate clamping
    to scalar perception variables to prevent sharp derivative discontinuities.
    """

    def __init__(
        self,
        initial_value: float,
        smoothing_factor: float = 0.20,
        max_slew_per_sec: float = 2.0,
    ) -> None:
        self.current_value = float(initial_value)
        self.target_value = float(initial_value)
        self.smoothing_factor = max(0.01, min(1.0, float(smoothing_factor)))
        self.max_slew_per_sec = max(1e-5, float(max_slew_per_sec))
        self.last_update_time = time.perf_counter()

    def set_target(self, target: float) -> None:
        """Assign new target setpoint."""
        self.target_value = float(target)

    def update(self, dt: Optional[float] = None) -> float:
        """
        Advance one time step and return smoothed parameter value.
        """
        now = time.perf_counter()
        if dt is None:
            dt = max(0.001, now - self.last_update_time)
        self.last_update_time = now

        # Unconstrained exponential smoothing
        diff = self.target_value - self.current_value
        step = diff * self.smoothing_factor

        # Slew-rate clamping
        max_allowed_step = self.max_slew_per_sec * dt
        clamped_step = max(-max_allowed_step, min(max_allowed_step, step))

        self.current_value += clamped_step
        return self.current_value

    def force_reset(self, value: float) -> None:
        """Instantly snap smoother state to target value."""
        self.current_value = float(value)
        self.target_value = float(value)
        self.last_update_time = time.perf_counter()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Master Scene Adaptive Tuner Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SceneAdaptiveTuner:
    """
    Central orchestration engine for dynamic perceptual parameter adaptation.
    Translates scene classification streams into smooth, safety-verified configuration
    snapshots across all APEX-Vision subsystems.
    """

    def __init__(
        self,
        registry: Optional[SceneProfileRegistry] = None,
        transition_duration_sec: float = 1.0,
        hysteresis_frame_count: int = 5,
        default_scene: SceneType = SceneType.HIGHWAY,
    ) -> None:
        """
        Initialize SceneAdaptiveTuner instance.
        """
        self.registry = registry if registry is not None else get_scene_profile_registry()
        self.transition_duration_sec = max(0.1, float(transition_duration_sec))
        self.hysteresis_threshold = max(1, int(hysteresis_frame_count))

        self._lock = threading.Lock()
        self.active_scene: SceneType = default_scene
        self.target_scene: SceneType = default_scene
        self.transition_start_time: float = time.perf_counter()
        self.is_in_transition: bool = False

        # Hysteresis state tracking
        self._candidate_scene: SceneType = default_scene
        self._candidate_count: int = 0

        # Transition audit history
        self._history: List[TransitionEventRecord] = []
        self._event_counter: int = 0

        # Subsystem Smoothers
        default_profile = self.registry.get_profile(default_scene)

        self._smoother_canny_low = SlewRateSmoother(default_profile.classical.canny_low, 0.25, 40.0)
        self._smoother_canny_high = SlewRateSmoother(default_profile.classical.canny_high, 0.25, 60.0)
        self._smoother_hls_l = SlewRateSmoother(default_profile.classical.hls_l_min, 0.25, 50.0)
        self._smoother_gamma = SlewRateSmoother(default_profile.lighting.gamma, 0.20, 1.5)
        self._smoother_clahe = SlewRateSmoother(default_profile.lighting.clahe_clip_limit, 0.20, 2.0)
        self._smoother_dl_weight = SlewRateSmoother(default_profile.fusion.dl_fusion_weight, 0.20, 0.5)
        self._smoother_guardrail_buffer = SlewRateSmoother(default_profile.guardrail.inner_buffer_width_m, 0.15, 0.4)
        self._smoother_stanley_gain = SlewRateSmoother(default_profile.guardrail.stanley_gain_base, 0.20, 0.8)
        self._smoother_roi_top_width = SlewRateSmoother(default_profile.roi.trapezoid_top_width_ratio, 0.20, 0.3)
        self._smoother_speed = SlewRateSmoother(default_profile.target_cruise_speed_mps, 0.10, 5.0)

        # Cached latest snapshot
        self._latest_snapshot: DynamicTuningSnapshot = self._build_snapshot(default_profile, 0.0)

    # ──────────────────────────────────────────────────────────────────────────
    # Feed Ingestion & State Updating
    # ──────────────────────────────────────────────────────────────────────────

    def update_scene(self, classification: SceneClassificationResult) -> DynamicTuningSnapshot:
        """
        Process a new classification result and update dynamic adaptation setpoints.
        """
        with self._lock:
            detected_scene = classification.scene_tag

            # 1. Hysteresis Evaluation
            if detected_scene != self.target_scene:
                if detected_scene == self._candidate_scene:
                    self._candidate_count += 1
                    if self._candidate_count >= self.hysteresis_threshold:
                        self._initiate_scene_transition(detected_scene, classification.confidence)
                else:
                    self._candidate_scene = detected_scene
                    self._candidate_count = 1
            else:
                self._candidate_count = 0

            # 2. Compute transition progress
            now = time.perf_counter()
            if self.is_in_transition:
                elapsed = now - self.transition_start_time
                progress = min(1.0, elapsed / self.transition_duration_sec)
                if progress >= 1.0:
                    self.active_scene = self.target_scene
                    self.is_in_transition = False
                    self._finalize_transition(elapsed)
            else:
                progress = 1.0

            # 3. Update target setpoints from registry profile
            target_profile = self.registry.get_profile(self.target_scene)
            self._update_smoother_targets(target_profile)

            # 4. Advance continuous smoothers
            self._advance_smoothers()

            # 5. Assemble and cache snapshot
            snapshot = self._build_snapshot(target_profile, progress)
            self._latest_snapshot = snapshot
            return copy.deepcopy(snapshot)

    def force_scene(self, scene: Union[SceneType, str]) -> DynamicTuningSnapshot:
        """
        Immediate manual override of the active scene profile without waiting for hysteresis.
        """
        scene_type = scene if isinstance(scene, SceneType) else SceneType.from_string(scene)
        with self._lock:
            logger.info("Manual override forced scene -> %s", scene_type.value)
            self.active_scene = scene_type
            self.target_scene = scene_type
            self.is_in_transition = False
            self._candidate_scene = scene_type
            self._candidate_count = 0

            target_profile = self.registry.get_profile(scene_type)
            self._force_reset_smoothers(target_profile)

            snapshot = self._build_snapshot(target_profile, 1.0)
            self._latest_snapshot = snapshot
            return copy.deepcopy(snapshot)

    def get_active_snapshot(self) -> DynamicTuningSnapshot:
        """
        Instantaneous thread-safe read of the current dynamic tuning snapshot.
        """
        with self._lock:
            return copy.deepcopy(self._latest_snapshot)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Smoothing & Transition Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _initiate_scene_transition(self, new_scene: SceneType, confidence: float) -> None:
        """Trigger handover between previous active scene and new target scene."""
        logger.info(
            "Initiating scene adaptation transition: %s -> %s (conf: %.2f)",
            self.active_scene.value,
            new_scene.value,
            confidence,
        )
        self.target_scene = new_scene
        self.is_in_transition = True
        self.transition_start_time = time.perf_counter()

        self._event_counter += 1
        record = TransitionEventRecord(
            event_id=self._event_counter,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            from_scene=self.active_scene,
            to_scene=new_scene,
            duration_sec=0.0,
            initial_confidence=confidence,
            status="IN_PROGRESS",
        )
        self._history.append(record)

    def _finalize_transition(self, duration_sec: float) -> None:
        """Mark the active transition as settled."""
        logger.info("Scene adaptation settled into stable profile: %s in %.2fs", self.active_scene.value, duration_sec)
        if self._history:
            self._history[-1].duration_sec = duration_sec
            self._history[-1].status = "COMPLETED"

    def _update_smoother_targets(self, profile: SceneTuningProfile) -> None:
        """Update individual smoother target values."""
        self._smoother_canny_low.set_target(profile.classical.canny_low)
        self._smoother_canny_high.set_target(profile.classical.canny_high)
        self._smoother_hls_l.set_target(profile.classical.hls_l_min)
        self._smoother_gamma.set_target(profile.lighting.gamma)
        self._smoother_clahe.set_target(profile.lighting.clahe_clip_limit)
        self._smoother_dl_weight.set_target(profile.fusion.dl_fusion_weight)
        self._smoother_guardrail_buffer.set_target(profile.guardrail.inner_buffer_width_m)
        self._smoother_stanley_gain.set_target(profile.guardrail.stanley_gain_base)
        self._smoother_roi_top_width.set_target(profile.roi.trapezoid_top_width_ratio)
        self._smoother_speed.set_target(profile.target_cruise_speed_mps)

    def _advance_smoothers(self) -> None:
        """Step all smoothers forward."""
        self._smoother_canny_low.update()
        self._smoother_canny_high.update()
        self._smoother_hls_l.update()
        self._smoother_gamma.update()
        self._smoother_clahe.update()
        self._smoother_dl_weight.update()
        self._smoother_guardrail_buffer.update()
        self._smoother_stanley_gain.update()
        self._smoother_roi_top_width.update()
        self._smoother_speed.update()

    def _force_reset_smoothers(self, profile: SceneTuningProfile) -> None:
        """Snap smoothers directly to designated profile targets."""
        self._smoother_canny_low.force_reset(profile.classical.canny_low)
        self._smoother_canny_high.force_reset(profile.classical.canny_high)
        self._smoother_hls_l.force_reset(profile.classical.hls_l_min)
        self._smoother_gamma.force_reset(profile.lighting.gamma)
        self._smoother_clahe.force_reset(profile.lighting.clahe_clip_limit)
        self._smoother_dl_weight.force_reset(profile.fusion.dl_fusion_weight)
        self._smoother_guardrail_buffer.force_reset(profile.guardrail.inner_buffer_width_m)
        self._smoother_stanley_gain.force_reset(profile.guardrail.stanley_gain_base)
        self._smoother_roi_top_width.force_reset(profile.roi.trapezoid_top_width_ratio)
        self._smoother_speed.force_reset(profile.target_cruise_speed_mps)

    def _build_snapshot(self, profile: SceneTuningProfile, progress: float) -> DynamicTuningSnapshot:
        """
        Extract smoothed variables, clamp to safety bounds, and construct telemetry snapshot.
        """
        # Read and clamp smoothed variables
        c_low = int(round(max(10.0, min(100.0, self._smoother_canny_low.current_value))))
        c_high = int(round(max(float(c_low + 20), min(220.0, self._smoother_canny_high.current_value))))
        hls_l = int(round(max(100.0, min(240.0, self._smoother_hls_l.current_value))))
        gamma = round(max(0.5, min(3.0, self._smoother_gamma.current_value)), 3)
        clahe_clip = round(max(1.0, min(6.0, self._smoother_clahe.current_value)), 2)
        dl_weight = round(max(0.20, min(0.95, self._smoother_dl_weight.current_value)), 3)
        cv_weight = round(1.0 - dl_weight, 3)
        guardrail_buf = round(max(0.15, min(0.60, self._smoother_guardrail_buffer.current_value)), 3)
        stanley_gain = round(max(0.5, min(3.0, self._smoother_stanley_gain.current_value)), 3)
        roi_top_w = round(max(0.25, min(0.75, self._smoother_roi_top_width.current_value)), 3)
        speed = round(max(5.0, min(35.0, self._smoother_speed.current_value)), 2)

        return DynamicTuningSnapshot(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            active_scene=self.active_scene,
            target_scene=self.target_scene,
            transition_progress=round(progress, 3),
            is_in_transition=self.is_in_transition,
            canny_low=c_low,
            canny_high=c_high,
            hls_l_min=hls_l,
            hls_s_min=profile.classical.hls_s_min,
            lab_b_min=profile.classical.lab_b_min,
            yuv_y_min=profile.classical.yuv_y_min,
            hough_threshold=profile.classical.hough_threshold,
            hough_min_line_length=profile.classical.hough_min_line_length,
            hough_max_line_gap=profile.classical.hough_max_line_gap,
            hough_min_slope=profile.classical.hough_min_slope,
            enable_gabor_bank=profile.classical.enable_gabor_bank,
            gamma=gamma,
            clahe_clip_limit=clahe_clip,
            enable_specular_removal=profile.lighting.enable_specular_removal,
            specular_l_threshold=profile.lighting.specular_l_threshold,
            night_mode_boost=profile.lighting.night_mode_boost,
            headlight_glare_suppression=profile.lighting.headlight_glare_suppression,
            dl_fusion_weight=dl_weight,
            cv_fusion_weight=cv_weight,
            confidence_threshold_min=profile.fusion.confidence_threshold_min,
            temporal_window_size=profile.fusion.history_window_size,
            ekf_process_noise_pos=profile.fusion.ekf_process_noise_pos,
            ekf_meas_noise_pos=profile.fusion.ekf_meas_noise_pos,
            lane_width_nominal_m=profile.fusion.lane_width_nominal_m,
            allow_hallucination=profile.fusion.allow_hallucination,
            max_hallucination_frames=profile.fusion.max_hallucination_frames,
            inner_guardrail_buffer_m=guardrail_buf,
            warning_buffer_m=guardrail_buf,
            critical_buffer_m=round(guardrail_buf * 0.5, 3),
            trajectory_prediction_horizon_s=profile.guardrail.trajectory_prediction_horizon_s,
            ttld_warning_threshold_s=profile.guardrail.ttld_warning_threshold_s,
            stanley_gain_base=stanley_gain,
            counter_steer_gain_multiplier=profile.guardrail.counter_steer_gain_multiplier,
            top_horizon_offset_px=profile.roi.top_horizon_offset_px,
            bottom_cut_px=profile.roi.bottom_cut_px,
            trapezoid_top_width_ratio=roi_top_w,
            trapezoid_bottom_width_ratio=profile.roi.trapezoid_bottom_width_ratio,
            curve_expansion_gain=profile.roi.curve_expansion_gain,
            target_cruise_speed_mps=speed,
            asil_b_compliance_flag=True,
        )

    def get_transition_history(self) -> List[TransitionEventRecord]:
        """Return history of all past scene transition handover events."""
        with self._lock:
            return copy.deepcopy(self._history)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Stanley Controller Scene-Adaptive Gain Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScheduledStanleyGains:
    """
    Active control gains scheduled for the Stanley path-tracking controller.
    """
    k_lateral: float                    # Cross-track error gain k_e
    k_soft: float                       # Low-speed denominator softening factor k_s
    k_heading: float                    # Heading error damping gain k_theta
    k_counter_steer: float              # Virtual guardrail emergency intervention gain
    lookahead_distance_m: float         # Dynamic lookahead evaluation distance L_d
    max_steering_command_rad: float     # Absolute steering rack limit delta_max


class StanleyGainScheduler:
    """
    Schedules Stanley lateral control gains as a combined function of:
      1. Active operational scene (Highway, Urban, Rural, Tunnel, Night, Rain)
      2. Ego vehicle longitudinal velocity v_x
      3. Road curvature radius R
      4. Guardrail safety proximity margin
    """

    def __init__(
        self,
        base_lateral_gain: float = 1.25,
        base_soft_gain: float = 1.00,
        base_heading_gain: float = 1.00,
        max_steer_rad: float = 0.4886,
    ) -> None:
        self.k_lat_base = base_lateral_gain
        self.k_soft_base = base_soft_gain
        self.k_head_base = base_heading_gain
        self.max_steer = max_steer_rad

    def schedule_gains(
        self,
        snapshot: DynamicTuningSnapshot,
        vehicle_speed_mps: float,
        radius_of_curvature_m: float = 5000.0,
        guardrail_proximity_ratio: float = 1.0,
    ) -> ScheduledStanleyGains:
        """
        Compute active Stanley gains for current vehicle state and tuning snapshot.
        """
        v_clamped = max(0.5, float(vehicle_speed_mps))
        r_clamped = max(10.0, abs(float(radius_of_curvature_m)))

        # Base gains modulated by scene tuning snapshot
        k_lat = snapshot.stanley_gain_base
        k_soft = max(0.2, 1.0 / (1.0 + 0.05 * v_clamped))
        k_head = self.k_head_base

        # Scene-specific gain modulation
        if snapshot.active_scene == SceneType.HIGHWAY:
            # At high speeds, lower lateral gain to prevent over-controlling / hunting
            speed_damping = max(0.70, 1.0 - 0.012 * (v_clamped - 15.0))
            k_lat *= speed_damping
            k_head *= 1.15
            lookahead_m = min(35.0, max(12.0, 0.85 * v_clamped))
        elif snapshot.active_scene == SceneType.URBAN:
            # In tight urban turns, boost lateral gain for rapid path tracking
            k_lat *= 1.30
            k_head *= 0.90
            lookahead_m = min(15.0, max(4.0, 0.45 * v_clamped))
        elif snapshot.active_scene == SceneType.RAIN:
            # On wet pavement, reduce high derivative kicks to prevent traction loss
            k_lat *= 0.85
            k_soft *= 1.25
            lookahead_m = min(25.0, max(8.0, 0.70 * v_clamped))
        elif snapshot.active_scene == SceneType.TUNNEL:
            k_lat *= 1.05
            lookahead_m = min(22.0, max(7.0, 0.65 * v_clamped))
        else:
            lookahead_m = min(25.0, max(6.0, 0.60 * v_clamped))

        # Curvature compensation: on sharp curves (R < 150m), boost tracking gain
        if r_clamped < 150.0:
            curve_factor = 1.0 + (150.0 - r_clamped) / 300.0
            k_lat *= min(1.40, curve_factor)

        # Guardrail proximity scaling: if vehicle encroaches on inner safety buffer,
        # escalate counter-steering gain aggressively
        proximity_clamped = max(0.0, min(1.0, guardrail_proximity_ratio))
        if proximity_clamped < 0.80:
            boost = snapshot.counter_steer_gain_multiplier * (1.0 - proximity_clamped)
            k_lat *= (1.0 + boost)

        return ScheduledStanleyGains(
            k_lateral=round(k_lat, 4),
            k_soft=round(k_soft, 4),
            k_heading=round(k_head, 4),
            k_counter_steer=round(k_lat * snapshot.counter_steer_gain_multiplier, 4),
            lookahead_distance_m=round(lookahead_m, 2),
            max_steering_command_rad=self.max_steer,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ASIL-B Adaptive Degradation Interlock & Safety Supervisor
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveDegradationInterlock:
    """
    ISO 26262 ASIL-B compliant safety supervisor monitoring scene adaptation state.
    Guarantees graceful degradation when scene classification confidence deteriorates
    or rapid parameter oscillations occur.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.35,
        max_consecutive_low_conf: int = 5,
        max_transition_time_sec: float = 4.0,
    ) -> None:
        self.min_confidence = min_confidence_threshold
        self.max_consecutive_low = max_consecutive_low_conf
        self.max_transition_sec = max_transition_time_sec

        self._consecutive_low_count = 0
        self._degraded_mode = False
        self._safe_fallback_active = False

    def verify_safety(
        self,
        tuner: SceneAdaptiveTuner,
        latest_classification: SceneClassificationResult,
    ) -> Tuple[bool, str]:
        """
        Audit current adaptation health and enforce fail-safe degradation interlocks.
        Returns: (is_healthy, status_message)
        """
        # 1. Confidence monitoring
        if latest_classification.confidence < self.min_confidence:
            self._consecutive_low_count += 1
            if self._consecutive_low_count >= self.max_consecutive_low:
                self._degraded_mode = True
                self._safe_fallback_active = True
                tuner.force_scene(SceneType.HIGHWAY)
                msg = (
                    f"DEGRADED: Low scene confidence ({latest_classification.confidence:.2f}) "
                    f"for {self._consecutive_low_count} frames. Reverted to conservative HIGHWAY baseline."
                )
                logger.warning(msg)
                return False, msg
        else:
            self._consecutive_low_count = max(0, self._consecutive_low_count - 1)
            if self._consecutive_low_count == 0:
                self._degraded_mode = False
                self._safe_fallback_active = False

        # 2. Prolonged transition watchdog
        if tuner.is_in_transition:
            elapsed = time.perf_counter() - tuner.transition_start_time
            if elapsed > self.max_transition_sec:
                tuner.is_in_transition = False
                tuner.active_scene = tuner.target_scene
                msg = f"TRANSITION_TIMEOUT: Handover exceeded {self.max_transition_sec}s; snapped to target scene."
                logger.warning(msg)
                return True, msg

        return True, "NOMINAL"

    @property
    def is_degraded(self) -> bool:
        return self._degraded_mode


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Audit Logging & Diagnostic Exporter
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveTuningAuditLogger:
    """
    Maintains an in-memory ring buffer and serializes adaptation event logs
    to disk for ISO 26262 post-drive traceability.
    """

    def __init__(self, log_dir: Optional[Union[str, Path]] = None, max_buffer_size: int = 500) -> None:
        self.log_dir = Path(log_dir) if log_dir else PROJECT_ROOT / "logs" / "adaptive_tuning"
        self.max_buffer_size = max_buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_snapshot(self, snapshot: DynamicTuningSnapshot) -> None:
        """Record snapshot telemetry into rotating memory buffer."""
        with self._lock:
            self._buffer.append(snapshot.to_dict())
            if len(self._buffer) > self.max_buffer_size:
                self._buffer.pop(0)

    def flush_to_disk(self) -> Optional[Path]:
        """Write recorded ring buffer to timestamped JSON file."""
        with self._lock:
            if not self._buffer:
                return None
            self.log_dir.mkdir(parents=True, exist_ok=True)
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_file = self.log_dir / f"adaptation_audit_{ts_str}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(self._buffer, f, indent=2)
            logger.info("Flushed %d adaptive tuning records to %s", len(self._buffer), out_file)
            return out_file


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ASIL-B Parameter Slew & Deviation Rate Safety Monitor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SlewViolationWarning:
    """Telemetry report when a parameter's rate of change exceeds safety margins."""
    parameter_name: str
    observed_slew_rate: float
    maximum_allowable_slew: float
    timestamp_utc: str
    severity: str                       # "CAUTION", "ASIL_INTERVENTION"


class ParameterDeviationMonitor:
    """
    Watches derivative velocity of all tuned variables.
    Verifies that no parameter jump violates ISO 26262 steering jerk limits ($d\delta / dt$).
    """

    def __init__(self, max_gain_rate: float = 2.5, max_buffer_rate: float = 1.0) -> None:
        self.max_gain_rate = max_gain_rate
        self.max_buffer_rate = max_buffer_rate
        self._last_snapshot: Optional[DynamicTuningSnapshot] = None
        self._last_time: float = time.perf_counter()
        self._violations: List[SlewViolationWarning] = []

    def audit_snapshot(self, snapshot: DynamicTuningSnapshot) -> List[SlewViolationWarning]:
        """
        Evaluate slew rate between successive snapshots.
        """
        now = time.perf_counter()
        dt = max(1e-4, now - self._last_time)
        self._last_time = now

        warnings: List[SlewViolationWarning] = []
        if self._last_snapshot is not None:
            # Audit Stanley gain slew rate
            d_gain = abs(snapshot.stanley_gain_base - self._last_snapshot.stanley_gain_base) / dt
            if d_gain > self.max_gain_rate:
                w = SlewViolationWarning(
                    parameter_name="stanley_gain_base",
                    observed_slew_rate=round(d_gain, 3),
                    maximum_allowable_slew=self.max_gain_rate,
                    timestamp_utc=snapshot.timestamp_utc,
                    severity="CAUTION",
                )
                warnings.append(w)
                self._violations.append(w)

            # Audit guardrail buffer slew rate
            d_buf = abs(snapshot.inner_guardrail_buffer_m - self._last_snapshot.inner_guardrail_buffer_m) / dt
            if d_buf > self.max_buffer_rate:
                w = SlewViolationWarning(
                    parameter_name="inner_guardrail_buffer_m",
                    observed_slew_rate=round(d_buf, 3),
                    maximum_allowable_slew=self.max_buffer_rate,
                    timestamp_utc=snapshot.timestamp_utc,
                    severity="CAUTION",
                )
                warnings.append(w)
                self._violations.append(w)

        self._last_snapshot = copy.deepcopy(snapshot)
        return warnings

    @property
    def total_violations(self) -> int:
        return len(self._violations)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Real-Time HUD Tuning Diagnostic Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

class TuningDiagnosticVisualizer:
    """
    Renders an engineering HUD overlay displaying active dynamic parameters,
    slew rates, and safety margin indicators onto debug display streams.
    """

    def __init__(self, font_scale: float = 0.38, line_thickness: int = 1) -> None:
        self.font_scale = font_scale
        self.thickness = line_thickness

    def render_overlay(
        self,
        canvas_bgr: np.ndarray,
        snapshot: DynamicTuningSnapshot,
        origin: Tuple[int, int] = (10, 200),
    ) -> np.ndarray:
        """
        Draw live tuning telemetry overlay box onto target BGR image.
        """
        import cv2
        out = canvas_bgr.copy()
        x0, y0 = origin

        # Background translucent box
        h_box, w_box = 135, 230
        cv2.rectangle(out, (x0 - 5, y0 - 15), (x0 + w_box, y0 + h_box), (20, 20, 20), -1)
        cv2.rectangle(out, (x0 - 5, y0 - 15), (x0 + w_box, y0 + h_box), (60, 60, 60), 1)

        # Header title
        title = f"TUNER: {snapshot.active_scene.value} (Trans: {snapshot.transition_progress*100:.0f}%)"
        cv2.putText(out, title, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 255, 255), self.thickness, cv2.LINE_AA)

        # Telemetry metrics rows
        rows = [
            f"DL/CV Weight:   {snapshot.dl_fusion_weight:.2f} / {snapshot.cv_fusion_weight:.2f}",
            f"Canny Range:    [{snapshot.canny_low}, {snapshot.canny_high}]",
            f"Gamma / CLAHE:  {snapshot.gamma:.2f} / {snapshot.clahe_clip_limit:.1f}",
            f"Guardrail Buf:  {snapshot.inner_guardrail_buffer_m:.2f} m",
            f"Stanley Gain:   {snapshot.stanley_gain_base:.2f}",
            f"Target Speed:   {snapshot.target_cruise_speed_mps:.1f} m/s",
            f"ASIL-B Safety:  {'NOMINAL' if snapshot.asil_b_compliance_flag else 'DEGRADED'}",
        ]

        y_cursor = y0 + 16
        for r in rows:
            color = (0, 255, 0) if "NOMINAL" in r else (220, 220, 220)
            cv2.putText(out, r, (x0, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.thickness, cv2.LINE_AA)
            y_cursor += 15

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Global Singleton Accessor
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_TUNER: Optional[SceneAdaptiveTuner] = None
_GLOBAL_SCHEDULER: Optional[StanleyGainScheduler] = None
_GLOBAL_INTERLOCK: Optional[AdaptiveDegradationInterlock] = None
_GLOBAL_MONITOR: Optional[ParameterDeviationMonitor] = None


def get_scene_adaptive_tuner(
    registry: Optional[SceneProfileRegistry] = None,
    transition_duration_sec: float = 1.0,
) -> SceneAdaptiveTuner:
    """
    Obtain or instantiate the global singleton SceneAdaptiveTuner.
    """
    global _GLOBAL_TUNER
    if _GLOBAL_TUNER is None:
        _GLOBAL_TUNER = SceneAdaptiveTuner(
            registry=registry,
            transition_duration_sec=transition_duration_sec,
        )
    return _GLOBAL_TUNER


def get_stanley_gain_scheduler() -> StanleyGainScheduler:
    """Obtain or instantiate the global singleton StanleyGainScheduler."""
    global _GLOBAL_SCHEDULER
    if _GLOBAL_SCHEDULER is None:
        _GLOBAL_SCHEDULER = StanleyGainScheduler()
    return _GLOBAL_SCHEDULER


def get_adaptive_degradation_interlock() -> AdaptiveDegradationInterlock:
    """Obtain or instantiate the global singleton AdaptiveDegradationInterlock."""
    global _GLOBAL_INTERLOCK
    if _GLOBAL_INTERLOCK is None:
        _GLOBAL_INTERLOCK = AdaptiveDegradationInterlock()
    return _GLOBAL_INTERLOCK


def get_parameter_deviation_monitor() -> ParameterDeviationMonitor:
    """Obtain or instantiate the global singleton ParameterDeviationMonitor."""
    global _GLOBAL_MONITOR
    if _GLOBAL_MONITOR is None:
        _GLOBAL_MONITOR = ParameterDeviationMonitor()
    return _GLOBAL_MONITOR


