"""
src/utils/thermal_guard.py
==========================
Hardware protection and thermal throttling for NVIDIA Jetson Orin Nano and RTX 4060.

Thresholds:
- Temp <= 80°C : Normal operation (Full 30+ FPS)
- Temp > 80°C  : Mild throttling (Cap at 15 FPS)
- Temp > 85°C  : Heavy throttling (Cap at 10 FPS, alert user; 5s cooldown during training)
- Temp > 90°C  : Thermal Emergency Stop (ESTOP trigger, request vehicle pull-over)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

log = logging.getLogger("apex_thermal_guard")

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


class ThermalState(Enum):
    NORMAL = "NORMAL"
    THROTTLE_MILD = "THROTTLE_MILD"       # > 80°C (15 FPS)
    THROTTLE_HEAVY = "THROTTLE_HEAVY"     # > 85°C (10 FPS)
    EMERGENCY_STOP = "EMERGENCY_STOP"     # > 90°C (0 FPS / Pull over)


class ThermalGuard:
    """
    Monitors processor temperatures and applies proactive throttling and emergency stops.
    """

    def __init__(
        self,
        mild_threshold_c: float = 80.0,
        heavy_threshold_c: float = 85.0,
        estop_threshold_c: float = 90.0,
        default_target_fps: float = 30.0,
    ) -> None:
        self.mild_threshold_c = mild_threshold_c
        self.heavy_threshold_c = heavy_threshold_c
        self.estop_threshold_c = estop_threshold_c
        self.default_target_fps = default_target_fps

        self.current_state = ThermalState.NORMAL
        self.current_temp_c: float = 45.0
        self.current_power_w: float = 10.0
        self.target_fps: float = default_target_fps
        self._simulated_temp: Optional[float] = None
        self._last_alert_time: float = 0.0

        # Try to initialize NVML for desktop GPUs (RTX 4060)
        self._nvml_initialized = False
        if HAS_PYNVML:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                self._nvml_initialized = False

    def simulate_temperature(self, temp_c: Optional[float]) -> None:
        """Inject simulated temperature reading for testing and validation."""
        self._simulated_temp = temp_c

    def read_temperature(self) -> Tuple[float, float]:
        """
        Query hardware thermal sensor.
        Priority order:
        1. Simulated temperature
        2. Jetson sysfs thermal zones
        3. pynvml (NVIDIA desktop GPU)
        4. Fallback safe ambient default
        """
        if self._simulated_temp is not None:
            self.current_temp_c = self._simulated_temp
            return self.current_temp_c, self.current_power_w

        # 1. Jetson Linux sysfs
        jetson_temp = self._read_jetson_thermal_zone()
        if jetson_temp is not None:
            self.current_temp_c = jetson_temp
            return self.current_temp_c, self.current_power_w

        # 2. NVIDIA Desktop GPU (pynvml)
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                self.current_temp_c = float(temp)
                self.current_power_w = float(power)
                return self.current_temp_c, self.current_power_w
            except Exception:
                pass

        return self.current_temp_c, self.current_power_w

    def _read_jetson_thermal_zone(self) -> Optional[float]:
        """Read Jetson Orin thermal zone if running on Tegra platform."""
        try:
            # Check thermal zones
            for zone in range(4):
                zone_path = Path(f"/sys/devices/virtual/thermal/thermal_zone{zone}/temp")
                if zone_path.exists():
                    raw = zone_path.read_text().strip()
                    temp_c = float(raw) / 1000.0 if float(raw) > 1000 else float(raw)
                    return temp_c
        except Exception:
            pass
        return None

    def evaluate_thermal_state(self, temp_c: Optional[float] = None) -> Tuple[ThermalState, float, str]:
        """
        Evaluate temperature against safety thresholds and determine target FPS.
        """
        if temp_c is None:
            temp_c, _ = self.read_temperature()
        else:
            self.current_temp_c = temp_c

        if temp_c >= self.estop_threshold_c:
            self.current_state = ThermalState.EMERGENCY_STOP
            self.target_fps = 0.0
            msg = f"CRITICAL OVERHEAT: GPU at {temp_c:.1f}°C (>= {self.estop_threshold_c}°C)! EMERGENCY STOP TRIGGERED."
            log.critical(msg)
            return self.current_state, self.target_fps, msg

        if temp_c >= self.heavy_threshold_c:
            self.current_state = ThermalState.THROTTLE_HEAVY
            self.target_fps = 10.0
            msg = f"HEAVY THERMAL THROTTLING: GPU at {temp_c:.1f}°C. Capping inference at 10 FPS."
            log.warning(msg)
            return self.current_state, self.target_fps, msg

        if temp_c >= self.mild_threshold_c:
            self.current_state = ThermalState.THROTTLE_MILD
            self.target_fps = 15.0
            msg = f"MILD THERMAL THROTTLING: GPU at {temp_c:.1f}°C. Capping inference at 15 FPS."
            log.warning(msg)
            return self.current_state, self.target_fps, msg

        self.current_state = ThermalState.NORMAL
        self.target_fps = self.default_target_fps
        return self.current_state, self.target_fps, "Thermal status nominal."

    def apply_throttle_delay(self, frame_start_time: float) -> float:
        """
        Sleep if necessary to throttle frame rate to `self.target_fps`.
        Returns total frame cycle duration in milliseconds.
        """
        if self.target_fps <= 0.0:
            return 0.0

        min_cycle_time = 1.0 / self.target_fps
        elapsed = time.time() - frame_start_time
        remaining = min_cycle_time - elapsed
        if remaining > 0.0:
            time.sleep(remaining)

        total_ms = (time.time() - frame_start_time) * 1000.0
        return total_ms

    def handle_training_cooldown(self, checkpoint_callback: Optional[Callable[[], None]] = None) -> bool:
        """
        During training, if temp > heavy_threshold (85°C), save checkpoint and insert 5s cooldown.
        """
        temp_c, _ = self.read_temperature()
        if temp_c >= self.heavy_threshold_c:
            log.warning("Training Thermal Guard: GPU reached %.1f°C. Initiating 5s cooldown.", temp_c)
            if checkpoint_callback:
                try:
                    checkpoint_callback()
                except Exception as e:
                    log.error("Failed to save emergency thermal checkpoint: %s", e)
            time.sleep(5.0)
            return True
        return False
