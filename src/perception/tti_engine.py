"""
src/perception/tti_engine.py
=============================
APEX-Vision Core v3.0 — Time-To-Impact (TTI) Collision Physics Engine (ISO 26262 Compliant)
- Calculates exact Time-To-Impact (TTI):
    TTI = Z_relative / (-v_relative) for closing obstacles (v_relative < 0).
- Assigns deterministic Threat Classification:
    0 = CLEAR (TTI > 4.0s or diverging)
    1 = WARNING (2.5s < TTI <= 4.0s)
    2 = CRITICAL (1.5s < TTI <= 2.5s)
    3 = EMERGENCY_BRAKE (TTI <= 1.5s)
- Formats payload metrics ready for CAN ID 0x170.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CollisionLevel(IntEnum):
    CLEAR           = 0
    WARNING         = 1
    CRITICAL        = 2
    EMERGENCY_BRAKE = 3


class EmergencyAction(IntEnum):
    NONE            = 0
    BRAKE_NOW       = 1
    SWERVE_LEFT     = 2
    SWERVE_RIGHT    = 3


@dataclass
class TTIAnalysis:
    """Collision assessment result for a dynamic target."""
    track_id: int
    tti_s: float
    tti_ms: int
    is_closing: bool
    closing_speed_m_s: float
    collision_level: CollisionLevel
    emergency_action: EmergencyAction
    intersect_x_mm: int


class TTIEngine:
    """
    Time-To-Impact collision assessment engine.
    """

    def __init__(
        self,
        warning_thresh_s: float = 4.0,
        critical_thresh_s: float = 2.5,
        emergency_thresh_s: float = 1.5,
        lane_half_width_m: float = 1.75,
    ) -> None:
        self.t_warn = float(warning_thresh_s)
        self.t_crit = float(critical_thresh_s)
        self.t_emerg = float(emergency_thresh_s)
        self.lane_half_w = float(lane_half_width_m)

    def calculate_tti(
        self,
        track_id: int,
        z_rel_m: float,
        vz_rel_m_s: float,
        x_rel_m: float = 0.0,
        vx_rel_m_s: float = 0.0,
    ) -> TTIAnalysis:
        """
        Evaluate Time-To-Impact for obstacle at longitudinal distance z_rel_m
        with relative forward velocity vz_rel_m_s.
        Closing occurs when vz_rel_m_s < 0 (obstacle approaching ego).
        """
        # Closing speed (-vz when vz < 0)
        closing_speed = -vz_rel_m_s

        if closing_speed > 0.05 and z_rel_m > 0.1:
            tti = z_rel_m / closing_speed
            is_closing = True
        else:
            tti = 99.9  # Diverging or stationary relative
            is_closing = False

        tti_ms = int(min(65535, max(0, round(tti * 1000.0))))

        # Lateral impact position
        if is_closing and tti < 10.0:
            intersect_x_m = x_rel_m + vx_rel_m_s * tti
        else:
            intersect_x_m = x_rel_m
        intersect_x_mm = int(min(32767, max(-32768, round(intersect_x_m * 1000.0))))

        # Determine threat level
        in_lane_or_intersecting = abs(intersect_x_m) <= (self.lane_half_w + 0.5)

        if not is_closing or tti > self.t_warn or not in_lane_or_intersecting:
            level = CollisionLevel.CLEAR
            action = EmergencyAction.NONE
        elif tti > self.t_crit:
            level = CollisionLevel.WARNING
            action = EmergencyAction.NONE
        elif tti > self.t_emerg:
            level = CollisionLevel.CRITICAL
            action = EmergencyAction.BRAKE_NOW
        else:
            level = CollisionLevel.EMERGENCY_BRAKE
            # Decide if evasion is possible or hard brake
            if intersect_x_m > 0:
                action = EmergencyAction.BRAKE_NOW  # default safest ISO 26262 brake
            else:
                action = EmergencyAction.BRAKE_NOW

        return TTIAnalysis(
            track_id=track_id,
            tti_s=round(tti, 2),
            tti_ms=tti_ms,
            is_closing=is_closing,
            closing_speed_m_s=round(closing_speed, 2),
            collision_level=level,
            emergency_action=action,
            intersect_x_mm=intersect_x_mm,
        )
