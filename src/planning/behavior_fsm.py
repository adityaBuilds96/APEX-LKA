"""
src/planning/behavior_fsm.py
============================
Hierarchical Behavior State Machine for APEX-LKA Autonomous Navigation:
- States:
  0 = LANE_FOLLOWING
  1 = ADAPTIVE_CRUISE
  2 = YIELD_OBSTACLE
  3 = OVERTAKE_OBSTACLE
  4 = EMERGENCY_BRAKE
  5 = SAFE_STOP
- Manages high-level driving tactics, safety transitions, speed ceilings,
  and lateral corridor bias based on RSS Shield, TTI, obstacle tracks, and lane retention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("apex_behavior_fsm")


class BehaviorState(IntEnum):
    LANE_FOLLOWING = 0
    LANE_KEEPING = 0
    ADAPTIVE_CRUISE = 1
    OBSTACLE_AVOIDANCE = 2
    YIELD_OBSTACLE = 2
    OVERTAKE_OBSTACLE = 3
    EMERGENCY_BRAKE = 4
    EMERGENCY_STOP = 4
    SEARCHING = 5
    SAFE_STOP = 6


@dataclass
class BehaviorConfig:
    """Configurable thresholds for behavior state transitions."""
    default_speed_limit_mps: float = 4.17      # 15 km/h chassis top speed
    standstill_distance_m: float = 2.0         # Minimum following distance at rest
    time_gap_seconds: float = 1.5              # Time headway to lead vehicle (s)
    max_accel_mpss: float = 1.5                # Comfort acceleration limit
    max_decel_mpss: float = 3.0                # Comfort deceleration limit
    emergency_decel_mpss: float = 4.5          # Hard braking deceleration
    overtake_lateral_offset_m: float = 1.2     # Lateral nudge distance for overtake
    yield_distance_m: float = 8.0              # Distance to begin yielding
    tti_danger_threshold_s: float = 1.8        # Critical TTI trigger for emergency brake
    corridor_half_width_m: float = 1.6         # Drivable lane corridor boundary
    min_state_dwell_frames: int = 5            # Anti-hunting debouncing dwell frames


@dataclass
class PlannerBehaviorContext:
    """Active behavior tactical plan dispatched to path planner and ACC."""
    state: BehaviorState = BehaviorState.LANE_FOLLOWING
    state_name: str = "LANE_FOLLOWING"
    target_speed_mps: float = 4.17
    speed_ceiling_mps: float = 4.17
    obstacle_avoidance_active: bool = False
    emergency_brake_active: bool = False
    priority_score: float = 1.0
    transition_reason: str = "Nominal lane tracking"
    lateral_bias_m: float = 0.0                # Metric lateral nudge offset (+ = right, - = left)
    lead_distance_m: Optional[float] = None
    lead_relative_speed_mps: Optional[float] = None
    tti_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": int(self.state),
            "state_name": self.state_name,
            "target_speed_mps": round(float(self.target_speed_mps), 2),
            "speed_ceiling_mps": round(float(self.speed_ceiling_mps), 2),
            "obstacle_avoidance_active": self.obstacle_avoidance_active,
            "emergency_brake_active": self.emergency_brake_active,
            "priority_score": round(float(self.priority_score), 2),
            "transition_reason": self.transition_reason,
            "lateral_bias_m": round(float(self.lateral_bias_m), 2),
            "lead_distance_m": round(float(self.lead_distance_m), 2) if self.lead_distance_m is not None else None,
            "tti_seconds": round(float(self.tti_seconds), 2) if self.tti_seconds is not None else None,
        }


class BehaviorStateMachine:
    """
    Context-aware finite state machine governing autonomous driving modes.
    Ensures safe, deterministic, priority-ordered behavioral transitions.
    """

    def __init__(self, config: Optional[BehaviorConfig] = None) -> None:
        self.config = config or BehaviorConfig()
        self.current_state = BehaviorState.LANE_FOLLOWING
        self._dwell_frames: int = 0
        self._last_reason: str = "System boot"

    def reset(self) -> None:
        """Reset state machine to initial lane following mode."""
        self.current_state = BehaviorState.LANE_FOLLOWING
        self._dwell_frames = 0
        self._last_reason = "System reset"

    def update(
        self,
        ego_speed_mps: float,
        tracked_objects: Optional[List[Any]] = None,
        rss_override: bool = False,
        lane_detected: bool = True,
        lane_retained: bool = False,
        dt: float = 0.033,
    ) -> PlannerBehaviorContext:
        """
        Evaluate sensor inputs and execute deterministic state transitions.

        Transition Priority Hierarchy:
        1. RSS Override / Critical TTI -> EMERGENCY_BRAKE
        2. Stationary / blocked corridor at near stop -> SAFE_STOP
        3. Static obstacle in path with passable adjacent space -> OVERTAKE_OBSTACLE
        4. Close obstacle / constriction requiring slowdown -> YIELD_OBSTACLE
        5. Moving lead vehicle inside corridor -> ADAPTIVE_CRUISE
        6. Clear road + lane valid/retained -> LANE_FOLLOWING
        """
        self._dwell_frames += 1

        # ── 1. Find Closest In-Corridor Lead Obstacle ──────────────────────────
        lead_dist: Optional[float] = None
        lead_rel_vz: Optional[float] = None
        lead_lat_x: Optional[float] = None
        lead_is_static: bool = True
        tti: Optional[float] = None

        if tracked_objects:
            in_corridor_objs = []
            for obj in tracked_objects:
                x_m = getattr(obj, "x_m", 0.0)
                z_m = getattr(obj, "z_m", 0.0)
                # Object must be ahead (z > 0.5m) and within lateral corridor
                if z_m > 0.5 and abs(x_m) <= self.config.corridor_half_width_m:
                    in_corridor_objs.append(obj)

            if in_corridor_objs:
                # Sort by longitudinal distance
                in_corridor_objs.sort(key=lambda o: getattr(o, "z_m", 999.0))
                closest = in_corridor_objs[0]
                lead_dist = float(getattr(closest, "z_m", 0.0))
                lead_lat_x = float(getattr(closest, "x_m", 0.0))

                vel_3d = getattr(closest, "velocity_3d", None)
                if vel_3d is not None and len(vel_3d) >= 3:
                    # vz > 0 means moving away, vz < 0 closing
                    lead_rel_vz = float(vel_3d[2])
                    speed_mag = float(np.linalg.norm(vel_3d))
                    lead_is_static = (speed_mag < 0.35)
                else:
                    lead_rel_vz = 0.0
                    lead_is_static = True

                # Compute Time-To-Impact (TTI)
                closing_speed = ego_speed_mps - lead_rel_vz
                if closing_speed > 0.2 and lead_dist > 0.1:
                    tti = lead_dist / closing_speed
                else:
                    tti = None

        # ── 2. Determine Candidate Next State ─────────────────────────────────
        next_state = self.current_state
        reason = self._last_reason

        # Priority 1: RSS Safety Shield Override
        if rss_override:
            next_state = BehaviorState.EMERGENCY_BRAKE
            reason = "Mobileye RSS Mathematical Safety Violation"

        # Priority 2: Critical TTI or imminent impact
        elif tti is not None and tti <= self.config.tti_danger_threshold_s and lead_dist is not None and lead_dist < 6.0:
            next_state = BehaviorState.EMERGENCY_BRAKE
            reason = f"Imminent collision risk (TTI={tti:.2f}s, dist={lead_dist:.2f}m)"

        # Priority 3: Safe Stop if stopped and path is blocked
        elif (
            ego_speed_mps < 0.5
            and lead_dist is not None
            and lead_dist <= (self.config.standstill_distance_m + 0.5)
            and lead_is_static
        ):
            next_state = BehaviorState.SAFE_STOP
            reason = f"Vehicle stopped behind obstacle (dist={lead_dist:.2f}m)"

        # Priority 4: Stationary Obstacle in Corridor (Overtake vs Yield)
        elif lead_dist is not None and lead_dist <= self.config.yield_distance_m and lead_is_static:
            # If obstacle is biased towards one side of corridor, nudge towards other
            if abs(lead_lat_x or 0.0) > 0.3:
                next_state = BehaviorState.OVERTAKE_OBSTACLE
                reason = f"Overtaking static obstacle at {lead_dist:.1f}m (offset {lead_lat_x:+.2f}m)"
            else:
                next_state = BehaviorState.YIELD_OBSTACLE
                reason = f"Yielding to static obstacle directly centered at {lead_dist:.1f}m"

        # Priority 5: Dynamic Lead Vehicle -> Adaptive Cruise Control
        elif lead_dist is not None and lead_dist <= 25.0:
            next_state = BehaviorState.ADAPTIVE_CRUISE
            reason = f"Following lead vehicle at {lead_dist:.1f}m"

        # Priority 6: Nominal Lane Following
        else:
            if not lane_detected and not lane_retained:
                # If lanes completely lost and no retention, slow down to safe stop
                next_state = BehaviorState.SAFE_STOP
                reason = "Lane vision lost with expired retention"
            else:
                next_state = BehaviorState.LANE_FOLLOWING
                reason = "Clear lane corridor tracking"

        # ── 3. Apply Debounce / Dwell Filter ───────────────────────────────────
        # EMERGENCY_BRAKE is instantaneous (zero dwell time delay)
        is_emergency = (next_state == BehaviorState.EMERGENCY_BRAKE)
        can_transition = is_emergency or (self._dwell_frames >= self.config.min_state_dwell_frames)

        if can_transition and next_state != self.current_state:
            log.info("FSM Transition: %s -> %s (Reason: %s)", self.current_state.name, next_state.name, reason)
            self.current_state = next_state
            self._dwell_frames = 0
            self._last_reason = reason
        elif not can_transition and next_state != self.current_state:
            # Suppress jitter until dwell time expires
            pass

        # ── 4. Generate Planner Behavioral Context ────────────────────────────
        context = PlannerBehaviorContext(
            state=self.current_state,
            state_name=self.current_state.name,
            transition_reason=self._last_reason,
            lead_distance_m=lead_dist,
            lead_relative_speed_mps=lead_rel_vz,
            tti_seconds=tti,
        )

        # Tactical parameters per state
        if self.current_state == BehaviorState.EMERGENCY_BRAKE:
            context.target_speed_mps = 0.0
            context.speed_ceiling_mps = 0.0
            context.emergency_brake_active = True
            context.priority_score = 10.0

        elif self.current_state == BehaviorState.SAFE_STOP:
            context.target_speed_mps = 0.0
            context.speed_ceiling_mps = 0.0
            context.priority_score = 8.0

        elif self.current_state == BehaviorState.YIELD_OBSTACLE:
            context.target_speed_mps = min(1.5, self.config.default_speed_limit_mps * 0.4)
            context.speed_ceiling_mps = 2.0
            context.obstacle_avoidance_active = True
            context.priority_score = 6.0

        elif self.current_state == BehaviorState.OVERTAKE_OBSTACLE:
            context.target_speed_mps = min(2.5, self.config.default_speed_limit_mps * 0.6)
            context.speed_ceiling_mps = 3.0
            context.obstacle_avoidance_active = True
            context.priority_score = 5.0
            # Nudge in opposite direction of obstacle lateral position
            if lead_lat_x is not None:
                context.lateral_bias_m = -np.sign(lead_lat_x) * self.config.overtake_lateral_offset_m
            else:
                context.lateral_bias_m = -self.config.overtake_lateral_offset_m

        elif self.current_state == BehaviorState.ADAPTIVE_CRUISE:
            context.speed_ceiling_mps = self.config.default_speed_limit_mps
            # Target speed computed dynamically by ACC module
            context.target_speed_mps = self.config.default_speed_limit_mps
            context.priority_score = 3.0

        else:  # LANE_FOLLOWING
            context.target_speed_mps = self.config.default_speed_limit_mps
            context.speed_ceiling_mps = self.config.default_speed_limit_mps
            context.priority_score = 1.0

        return context
