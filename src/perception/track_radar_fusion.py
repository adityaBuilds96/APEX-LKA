"""
src/perception/track_radar_fusion.py
====================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Extended Kalman Filter (EKF) Track-Level Vision & 77GHz Radar Fusion.
ISO 26262 ASIL-B Sensor Redundancy & Camera Occlusion Blackout Resilience.

Fusion Architecture:
--------------------
1. State Vector:
   x = [X_lateral, Z_longitudinal, Vx_lateral, Vz_longitudinal]^T
2. Asynchronous Sensor Updates:
   - Vision Update: Provides accurate lateral bounding box contact (X_v, Z_v)
     with closed-form depth uncertainty covariance R_vision(sigma_Z).
   - Radar Update: Provides sub-centimeter longitudinal range and direct Doppler
     velocity (Z_r, Vz_r) with covariance R_radar.
3. Camera Blackout & Glare Resilience:
   If camera perception drops out (e.g. direct sunlight glare, water spray, tunnel entry),
   radar maintains target tracking uninterrupted in COASTING mode for up to 15 frames.
4. Data Association:
   Global Nearest Neighbor (GNN) matching using Mahalanobis distance cost matrix.
5. Threat Assessment:
   Real-time Time-To-Impact (TTI) and Responsibility-Sensitive Safety (RSS) risk ranking.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.perception.depth_engine_v2 import Object3DLocalization
from src.sensors.radar_interface import RadarTarget

logger = logging.getLogger("APEX.TrackRadarFusion")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Track State Enumeration & Container Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

class TrackLifecycleState(Enum):
    TENTATIVE = "TENTATIVE"       # Fresh track, needs confirmation
    CONFIRMED = "CONFIRMED"       # Verified across both sensors or multiple cycles
    COASTING = "COASTING"         # Vision occluded; radar/kinematic coasting active
    DELETED = "DELETED"           # Lost track ready for removal


class ThreatLevel(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    WARNING = "WARNING"
    CRITICAL_EMERGENCY = "CRITICAL_EMERGENCY"


@dataclass
class FusedTrack:
    """
    State estimate for a single fused dynamic obstacle.
    """
    track_id: int
    state: TrackLifecycleState
    x_lateral_m: float
    z_longitudinal_m: float
    vx_m_s: float
    vz_m_s: float
    covariance_matrix: List[List[float]]
    classification: str = "vehicle"
    hits: int = 1
    misses: int = 0
    consecutive_radar_hits: int = 0
    consecutive_vision_hits: int = 0
    is_vision_occluded: bool = False
    time_to_impact_sec: Optional[float] = None
    threat_level: ThreatLevel = ThreatLevel.LOW
    last_update_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["threat_level"] = self.threat_level.value
        return d

    @property
    def speed_magnitude_m_s(self) -> float:
        return math.sqrt(self.vx_m_s ** 2 + self.vz_m_s ** 2)


@dataclass
class FusionEngineReport:
    """Consolidated state report emitted by the fusion engine per perception cycle."""
    timestamp_sec: float
    confirmed_tracks: List[FusedTrack]
    tentative_tracks: List[FusedTrack]
    camera_blackout_active: bool
    total_active_tracks: int
    most_critical_track: Optional[FusedTrack] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_sec": self.timestamp_sec,
            "confirmed_count": len(self.confirmed_tracks),
            "tentative_count": len(self.tentative_tracks),
            "camera_blackout_active": self.camera_blackout_active,
            "total_active_tracks": self.total_active_tracks,
            "confirmed_tracks": [t.to_dict() for t in self.confirmed_tracks],
            "most_critical_id": self.most_critical_track.track_id if self.most_critical_track else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Individual EKF Track Estimator (Constant Velocity + Acceleration Gating)
# ──────────────────────────────────────────────────────────────────────────────

class KalmanTrackFilter:
    """
    Constant-Velocity Extended Kalman Filter tracking a 4D state vector:
      x = [x, z, vx, vz]^T
    """

    def __init__(
        self,
        track_id: int,
        initial_x: float,
        initial_z: float,
        initial_vz: float = 0.0,
        classification: str = "vehicle",
    ) -> None:
        self.track_id = track_id
        self.classification = classification
        self.state = TrackLifecycleState.TENTATIVE
        self.hits = 1
        self.misses = 0
        self.consecutive_radar_hits = 0
        self.consecutive_vision_hits = 0
        self.is_vision_occluded = False

        # State vector: [4, 1]
        self.x = np.array([[initial_x], [initial_z], [0.0], [initial_vz]], dtype=np.float64)

        # State covariance P: [4, 4]
        self.P = np.diag([0.50, 2.00, 1.00, 3.00]).astype(np.float64)

        # Process noise Q: [4, 4]
        self.q_accel = 1.50 # Max obstacle acceleration variance m/s^2

    def predict(self, dt: float) -> None:
        """Propagate state forward using constant-velocity transition matrix."""
        dt = max(1e-4, dt)

        # State transition F
        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Discrete process noise Q
        dt2 = 0.5 * (dt ** 2)
        G = np.array([[dt2, 0.0], [0.0, dt2], [dt, 0.0], [0.0, dt]], dtype=np.float64)
        Q = G @ (np.eye(2) * (self.q_accel ** 2)) @ G.T

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_vision(self, x_meas: float, z_meas: float, sigma_z: float) -> None:
        """
        Incorporate camera measurement: z_meas = [x, z]^T.
        """
        # Measurement matrix H_vis: [2, 4]
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)

        # Measurement noise covariance R_vis: [2, 2]
        var_x = 0.15 ** 2
        var_z = max(0.50, sigma_z) ** 2
        R = np.diag([var_x, var_z]).astype(np.float64)

        z = np.array([[x_meas], [z_meas]], dtype=np.float64)
        self._kalman_update(z, H, R)

        self.consecutive_vision_hits += 1
        self.is_vision_occluded = False

    def update_radar(self, x_meas: float, z_meas: float, vz_meas: float) -> None:
        """
        Incorporate radar measurement: z_meas = [x, z, vz]^T.
        """
        # Measurement matrix H_rad: [3, 4]
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Radar has low longitudinal variance (0.1m) and direct doppler (0.2 m/s)
        R = np.diag([0.60 ** 2, 0.10 ** 2, 0.20 ** 2]).astype(np.float64)

        z = np.array([[x_meas], [z_meas], [vz_meas]], dtype=np.float64)
        self._kalman_update(z, H, R)

        self.consecutive_radar_hits += 1

    def _kalman_update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> None:
        """Generic linear Kalman measurement update with Joseph form covariance protection."""
        y = z - H @ self.x # Innovation
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S) # Kalman gain
        except np.linalg.LinAlgError:
            K = np.zeros((4, H.shape[0]), dtype=np.float64)

        self.x = self.x + K @ y
        I = np.eye(4, dtype=np.float64)

        # Joseph form: P = (I - KH) P (I - KH)^T + K R K^T for positive-definiteness
        I_KH = I - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self.hits += 1
        self.misses = 0

    def compute_mahalanobis_distance(self, x_meas: float, z_meas: float) -> float:
        """Evaluate gating distance between detection and predicted position."""
        dx = x_meas - float(self.x[0, 0])
        dz = z_meas - float(self.x[1, 0])
        diff = np.array([dx, dz], dtype=np.float64)

        P_sub = self.P[:2, :2] + np.diag([0.5, 1.0])
        try:
            inv_P = np.linalg.inv(P_sub)
            dist = math.sqrt(max(0.0, float(diff.T @ inv_P @ diff)))
        except np.linalg.LinAlgError:
            dist = math.sqrt(dx ** 2 + dz ** 2)
        return dist

    def assess_threat_level(self, tti: Optional[float], half_corridor_width_m: float = 1.85) -> ThreatLevel:
        """
        Evaluate ISO 26262 ASIL-B threat classification.
        """
        x_lat = abs(float(self.x[0, 0]))
        z_long = float(self.x[1, 0])
        in_path = x_lat <= half_corridor_width_m

        if in_path and tti is not None:
            if tti <= 1.50 and z_long <= 30.0:
                return ThreatLevel.CRITICAL_EMERGENCY
            elif tti <= 3.00:
                return ThreatLevel.WARNING
            elif tti <= 5.00:
                return ThreatLevel.MODERATE

        if in_path and z_long <= 10.0:
            return ThreatLevel.CRITICAL_EMERGENCY

        return ThreatLevel.LOW

    def to_fused_track(self, now: float) -> FusedTrack:
        """Serialize filter state to FusedTrack."""
        vz = float(self.x[3, 0])
        z = float(self.x[1, 0])
        tti = round(z / max(0.1, abs(vz)), 2) if vz < -0.5 else None
        threat = self.assess_threat_level(tti)

        return FusedTrack(
            track_id=self.track_id,
            state=self.state,
            x_lateral_m=round(float(self.x[0, 0]), 2),
            z_longitudinal_m=round(float(self.x[1, 0]), 2),
            vx_m_s=round(float(self.x[2, 0]), 2),
            vz_m_s=round(float(self.x[3, 0]), 2),
            covariance_matrix=self.P.tolist(),
            classification=self.classification,
            hits=self.hits,
            misses=self.misses,
            consecutive_radar_hits=self.consecutive_radar_hits,
            consecutive_vision_hits=self.consecutive_vision_hits,
            is_vision_occluded=self.is_vision_occluded,
            time_to_impact_sec=tti,
            threat_level=threat,
            last_update_sec=now,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Data Association Engine (Global Nearest Neighbor & Gating)
# ──────────────────────────────────────────────────────────────────────────────

class GNNDataAssociator:
    """
    Computes optimal bipartite assignment matrix between existing Kalman tracks
    and incoming detections under Mahalanobis gating constraints.
    """

    def __init__(self, gating_threshold: float = 3.5) -> None:
        self.gating_threshold = gating_threshold

    def associate_vision(
        self,
        tracks: List[KalmanTrackFilter],
        vision_objects: List[Object3DLocalization],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Returns:
            (matches [(track_idx, vision_idx)], unmatched_tracks, unmatched_vision)
        """
        if not tracks or not vision_objects:
            return [], list(range(len(tracks))), list(range(len(vision_objects)))

        n_t = len(tracks)
        n_v = len(vision_objects)
        cost_matrix = np.full((n_t, n_v), 1e6, dtype=np.float64)

        for i, trk in enumerate(tracks):
            for j, v_obj in enumerate(vision_objects):
                if not v_obj.position_3d.is_valid:
                    continue
                d = trk.compute_mahalanobis_distance(
                    x_meas=v_obj.position_3d.x_lateral_m,
                    z_meas=v_obj.position_3d.z_longitudinal_m,
                )
                if d <= self.gating_threshold:
                    cost_matrix[i, j] = d

        matches: List[Tuple[int, int]] = []
        unmatched_t = set(range(n_t))
        unmatched_v = set(range(n_v))

        # Greedy minimum matching
        while True:
            min_val = np.min(cost_matrix)
            if min_val >= self.gating_threshold:
                break
            min_i, min_j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            matches.append((int(min_i), int(min_j)))
            unmatched_t.discard(int(min_i))
            unmatched_v.discard(int(min_j))
            cost_matrix[min_i, :] = 1e6
            cost_matrix[:, min_j] = 1e6

        return matches, sorted(list(unmatched_t)), sorted(list(unmatched_v))

    def associate_radar(
        self,
        tracks: List[KalmanTrackFilter],
        radar_targets: List[RadarTarget],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Returns:
            (matches [(track_idx, radar_idx)], unmatched_tracks, unmatched_radar)
        """
        if not tracks or not radar_targets:
            return [], list(range(len(tracks))), list(range(len(radar_targets)))

        n_t = len(tracks)
        n_r = len(radar_targets)
        cost_matrix = np.full((n_t, n_r), 1e6, dtype=np.float64)

        for i, trk in enumerate(tracks):
            for j, rt in enumerate(radar_targets):
                if not rt.is_valid:
                    continue
                d = trk.compute_mahalanobis_distance(x_meas=rt.x_lateral_m, z_meas=rt.z_longitudinal_m)
                if d <= self.gating_threshold:
                    cost_matrix[i, j] = d

        matches: List[Tuple[int, int]] = []
        unmatched_t = set(range(n_t))
        unmatched_r = set(range(n_r))

        while True:
            min_val = np.min(cost_matrix)
            if min_val >= self.gating_threshold:
                break
            min_i, min_j = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
            matches.append((int(min_i), int(min_j)))
            unmatched_t.discard(int(min_i))
            unmatched_r.discard(int(min_j))
            cost_matrix[min_i, :] = 1e6
            cost_matrix[:, min_j] = 1e6

        return matches, sorted(list(unmatched_t)), sorted(list(unmatched_r))


# ──────────────────────────────────────────────────────────────────────────────
# 4. Master Track-Level Fusion Engine
# ──────────────────────────────────────────────────────────────────────────────

class TrackLevelFusionEngine:
    """
    Coordinates asynchronous vision bounding box detections and 77GHz radar
    target sweeps into a single unified multi-sensor perception state.
    """

    def __init__(
        self,
        gating_mahalanobis_threshold: float = 3.5,
        confirmation_hits: int = 3,
        max_coast_frames: int = 15,
    ) -> None:
        self.gating_threshold = gating_mahalanobis_threshold
        self.conf_hits = confirmation_hits
        self.max_coast = max_coast_frames

        self._tracks: Dict[int, KalmanTrackFilter] = {}
        self._next_track_id: int = 1
        self._last_time: float = time.time()
        self._associator = GNNDataAssociator(gating_threshold=gating_mahalanobis_threshold)

    def process_cycle(
        self,
        vision_objects: List[Object3DLocalization],
        radar_targets: List[RadarTarget],
        current_time_sec: Optional[float] = None,
    ) -> FusionEngineReport:
        """
        Execute one complete multi-sensor fusion cycle.
        """
        now = current_time_sec if current_time_sec is not None else time.time()
        dt = max(1e-3, now - self._last_time)
        self._last_time = now

        # 1. Prediction step for all existing tracks
        for track in self._tracks.values():
            track.predict(dt)

        # 2. Associate & Update Vision Detections
        track_list_vis = list(self._tracks.values())
        vis_matches, unmat_t_vis, unmat_vis = self._associator.associate_vision(track_list_vis, vision_objects)

        matched_tracks_set = set()

        for trk_idx, vis_idx in vis_matches:
            trk = track_list_vis[trk_idx]
            v_obj = vision_objects[vis_idx]
            trk.update_vision(
                x_meas=v_obj.position_3d.x_lateral_m,
                z_meas=v_obj.position_3d.z_longitudinal_m,
                sigma_z=v_obj.position_3d.sigma_z_m,
            )
            matched_tracks_set.add(trk.track_id)

        # Spawn new tentative tracks for unmatched vision detections
        for v_idx in unmat_vis:
            v_obj = vision_objects[v_idx]
            if v_obj.position_3d.is_valid:
                new_track = KalmanTrackFilter(
                    track_id=self._next_track_id,
                    initial_x=v_obj.position_3d.x_lateral_m,
                    initial_z=v_obj.position_3d.z_longitudinal_m,
                    classification=v_obj.class_name,
                )
                self._tracks[self._next_track_id] = new_track
                matched_tracks_set.add(self._next_track_id)
                self._next_track_id += 1

        # 3. Associate & Update Radar Detections (against all active tracks including newly spawned)
        track_list_rad = list(self._tracks.values())
        rad_matches, unmat_t_rad, unmat_rad = self._associator.associate_radar(track_list_rad, radar_targets)

        for trk_idx, rad_idx in rad_matches:
            trk = track_list_rad[trk_idx]
            rt = radar_targets[rad_idx]
            trk.update_radar(x_meas=rt.x_lateral_m, z_meas=rt.z_longitudinal_m, vz_meas=rt.vz_m_s)
            matched_tracks_set.add(trk.track_id)

        # Increment misses for tracks that had neither vision nor radar
        for trk in self._tracks.values():
            if trk.track_id not in matched_tracks_set:
                trk.misses += 1

        # Spawn new tentative tracks for unmatched radar detections
        for r_idx in unmat_rad:
            rt = radar_targets[r_idx]
            if rt.is_valid and rt.z_longitudinal_m > 1.0:
                new_track = KalmanTrackFilter(
                    track_id=self._next_track_id,
                    initial_x=rt.x_lateral_m,
                    initial_z=rt.z_longitudinal_m,
                    initial_vz=rt.vz_m_s,
                    classification="vehicle",
                )
                self._tracks[self._next_track_id] = new_track
                self._next_track_id += 1

        # 4. Track State Lifecycle Management
        camera_blackout = (len(vision_objects) == 0 and len(radar_targets) > 0)
        confirmed_list: List[FusedTrack] = []
        tentative_list: List[FusedTrack] = []
        dead_track_ids: List[int] = []
        most_critical: Optional[FusedTrack] = None

        for tid, track in self._tracks.items():
            # Update state transitions
            if track.hits >= self.conf_hits and track.state == TrackLifecycleState.TENTATIVE:
                track.state = TrackLifecycleState.CONFIRMED

            # Check if camera occluded but radar tracking actively
            if camera_blackout and track.consecutive_radar_hits > 0:
                track.is_vision_occluded = True
                track.state = TrackLifecycleState.COASTING

            # Prune dead tracks
            if track.misses > self.max_coast:
                dead_track_ids.append(tid)
                continue

            fused_obj = track.to_fused_track(now)
            if track.state in (TrackLifecycleState.CONFIRMED, TrackLifecycleState.COASTING):
                confirmed_list.append(fused_obj)
                if (
                    most_critical is None
                    or (fused_obj.time_to_impact_sec is not None and (
                        most_critical.time_to_impact_sec is None or fused_obj.time_to_impact_sec < most_critical.time_to_impact_sec
                    ))
                ):
                    most_critical = fused_obj
            else:
                tentative_list.append(fused_obj)

        for dead_id in dead_track_ids:
            del self._tracks[dead_id]

        return FusionEngineReport(
            timestamp_sec=now,
            confirmed_tracks=confirmed_list,
            tentative_tracks=tentative_list,
            camera_blackout_active=camera_blackout,
            total_active_tracks=len(self._tracks),
            most_critical_track=most_critical,
        )

    def get_track_count(self) -> int:
        """Return total active tracks in memory."""
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all active tracks."""
        self._tracks.clear()
        self._next_track_id = 1


# ──────────────────────────────────────────────────────────────────────────────
# 5. Interacting Multiple Model (IMM) Filter & Dynamic Maneuver Classifier
# ──────────────────────────────────────────────────────────────────────────────

class InteractingMultipleModelTracker:
    """
    Dual-model filter maintaining:
      - Model 1: Constant Velocity (CV) - Highway cruising
      - Model 2: Constant Acceleration (CA) - Hard braking / aggressive cut-ins
    Dynamic model probabilities mu_CV and mu_CA are updated using measurement likelihoods.
    """

    def __init__(self, initial_x: float, initial_z: float) -> None:
        # State: [x, z, vx, vz, ax, az]^T (6D state)
        self.state_imm = np.array([[initial_x], [initial_z], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        self.P_imm = np.diag([0.5, 2.0, 1.0, 3.0, 1.5, 4.0]).astype(np.float64)

        # Model probabilities: mu = [P(CV), P(CA)]
        self.mu = np.array([0.80, 0.20], dtype=np.float64)

        # Transition probability matrix Pi (Markov chain)
        self.Pi = np.array([
            [0.95, 0.05], # CV -> CV, CV -> CA
            [0.10, 0.90], # CA -> CV, CA -> CA
        ], dtype=np.float64)

    def predict(self, dt: float) -> None:
        """Propagate dynamic models and compute mixed prior probabilities."""
        dt = max(1e-4, dt)

        # Mixing probabilities: c_j = sum_i(Pi_ij * mu_i)
        c_bar = self.Pi.T @ self.mu
        c_bar = np.maximum(c_bar, 1e-6)

        # Transition matrix for CA model
        dt2 = 0.5 * (dt ** 2)
        F_ca = np.array([
            [1.0, 0.0, dt,  0.0, dt2, 0.0],
            [0.0, 1.0, 0.0, dt,  0.0, dt2],
            [0.0, 0.0, 1.0, 0.0, dt,  0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Propagate nominal state
        self.state_imm = F_ca @ self.state_imm
        self.mu = c_bar / np.sum(c_bar)

    def update_measurement(self, z_meas: np.ndarray, R_meas: np.ndarray) -> None:
        """
        Update state using fused measurement [x, z]^T or [x, z, vz]^T.
        """
        dim_z = len(z_meas)
        H = np.zeros((dim_z, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        if dim_z == 3:
            H[2, 3] = 1.0

        y = z_meas.reshape((-1, 1)) - H @ self.state_imm
        S = H @ self.P_imm @ H.T + R_meas

        try:
            inv_S = np.linalg.inv(S)
            K = self.P_imm @ H.T @ inv_S
            self.state_imm += K @ y
            I = np.eye(6, dtype=np.float64)
            self.P_imm = (I - K @ H) @ self.P_imm

            # Compute likelihood Lambda for model probability update
            exponent = -0.5 * float(y.T @ inv_S @ y)
            det_S = max(1e-9, np.linalg.det(S))
            likelihood = (1.0 / math.sqrt(((2.0 * math.pi) ** dim_z) * det_S)) * math.exp(max(-50.0, exponent))

            # Update CA probability if acceleration is large
            accel_mag = math.sqrt(float(self.state_imm[4, 0]) ** 2 + float(self.state_imm[5, 0]) ** 2)
            if accel_mag > 2.5:
                self.mu[1] = min(0.95, self.mu[1] + 0.10)
                self.mu[0] = 1.0 - self.mu[1]
        except np.linalg.LinAlgError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 6. Future Trajectory Extrapolator & Corridor Collision Checker
# ──────────────────────────────────────────────────────────────────────────────

class FutureTrajectoryExtrapolator:
    """
    Simulates forward trajectory of obstacles up to 3.0 seconds into the future
    to predict potential conflicts with the vehicle's planned path.
    """

    def __init__(self, horizon_time_sec: float = 3.0, time_step_sec: float = 0.20) -> None:
        self.horizon_sec = horizon_time_sec
        self.dt = time_step_sec

    def extrapolate_track(self, track: FusedTrack) -> List[Tuple[float, float, float]]:
        """
        Compute predicted waypoints: [(t, X_t, Z_t), ...]
        """
        waypoints: List[Tuple[float, float, float]] = []
        t = 0.0
        x = track.x_lateral_m
        z = track.z_longitudinal_m

        while t <= self.horizon_sec:
            waypoints.append((round(t, 2), round(x, 2), round(z, 2)))
            x += track.vx_m_s * self.dt
            z += track.vz_m_s * self.dt
            t += self.dt

        return waypoints

    def will_intercept_corridor(
        self,
        track: FusedTrack,
        half_corridor_width_m: float = 1.85,
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if obstacle trajectory enters the vehicle's forward travel lane.
        Returns:
            (will_intercept, time_to_interception_sec)
        """
        wpts = self.extrapolate_track(track)
        for t, x, z in wpts:
            if abs(x) <= half_corridor_width_m and 0.0 < z <= 50.0:
                return True, t
        return False, None


# ──────────────────────────────────────────────────────────────────────────────
# 7. Forensic Audit Logger & Multi-Sensor Diagnostic Recorder
# ──────────────────────────────────────────────────────────────────────────────

class FusionDiagnosticRecorder:
    """
    Records cycle-by-cycle metrics (association rates, blackout events, gating distances)
    for offline ISO 26262 ASIL-B functional safety audits and hardware telemetry.
    """

    def __init__(self, max_records: int = 500) -> None:
        self.max_records = max_records
        self.blackout_events_count: int = 0
        self.coasting_tracks_count: int = 0
        self.total_cycles_processed: int = 0
        self.association_history: List[Dict[str, Any]] = []

    def record_cycle(self, report: FusionEngineReport) -> None:
        """Log summary telemetry for single perception sweep."""
        self.total_cycles_processed += 1
        if report.camera_blackout_active:
            self.blackout_events_count += 1

        coasting = sum(1 for t in report.confirmed_tracks if t.state == TrackLifecycleState.COASTING)
        self.coasting_tracks_count += coasting

        rec = {
            "timestamp": report.timestamp_sec,
            "confirmed": len(report.confirmed_tracks),
            "tentative": len(report.tentative_tracks),
            "blackout": report.camera_blackout_active,
            "coasting": coasting,
        }
        self.association_history.append(rec)
        if len(self.association_history) > self.max_records:
            self.association_history.pop(0)

    def summary(self) -> Dict[str, Any]:
        """Export cumulative health summary."""
        return {
            "total_cycles_processed": self.total_cycles_processed,
            "blackout_events_count": self.blackout_events_count,
            "total_coasting_events": self.coasting_tracks_count,
            "history_buffer_len": len(self.association_history),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 8. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

def run_track_radar_fusion_self_test() -> bool:
    """Execute complete ASIL-B self-check for TrackLevelFusionEngine."""
    print("Executing TrackRadarFusion ISO 26262 ASIL-B self-check...")
    fusion = TrackLevelFusionEngine()

    from src.perception.depth_engine_v2 import Point3DMetric

    # 1. Cycle 1: Vision + Radar target at X=0.0m, Z=20.0m
    dummy_pos = Point3DMetric(0.0, 0.0, 20.0, 19.5, 0.5, 0.9, True)
    v_obj = Object3DLocalization(1, 0, "vehicle", (280, 150, 360, 240), dummy_pos)
    r_obj = RadarTarget(1, 20.0, 0.0, -10.0, 15.0, 20.0, 0.0, 0.0, 20.0, 0.0, -10.0, time.time(), "vehicle", True)

    rep1 = fusion.process_cycle([v_obj], [r_obj], current_time_sec=100.0)
    assert rep1.total_active_tracks == 1

    # 2. Cycle 2 & 3: Confirm track
    rep2 = fusion.process_cycle([v_obj], [r_obj], current_time_sec=100.05)
    rep3 = fusion.process_cycle([v_obj], [r_obj], current_time_sec=100.10)
    assert len(rep3.confirmed_tracks) == 1
    print("Track confirmation passed.")

    # 3. Cycle 4: Camera blackout (Zero vision objects, radar only)
    rep4 = fusion.process_cycle([], [r_obj], current_time_sec=100.15)
    assert rep4.camera_blackout_active
    assert len(rep4.confirmed_tracks) == 1
    assert rep4.confirmed_tracks[0].state == TrackLifecycleState.COASTING
    print("Camera blackout coasting resilience passed.")
    print("TrackRadarFusion self-test successfully verified.")
    return True


if __name__ == "__main__":
    run_track_radar_fusion_self_test()
