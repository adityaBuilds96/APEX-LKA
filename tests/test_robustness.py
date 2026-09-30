"""
tests/test_robustness.py
========================
Automated verification of the Never-Crash Architecture across APEX-LKA:
1. SafeExecutor exception shielding & consecutive failure threshold
2. 6-Level Graceful Degradation hierarchy
3. MemoryGuard monotonic leak detection & automated memory reclamation
4. ThermalGuard multi-stage throttling & ESTOP handling
5. CrashLogger forensic diagnostic dumps (JSON + Markdown)
6. Preflight HealthChecker sanity validation
7. ProcessWatchdog liveness monitoring, auto-restart & safe-mode shield
8. Simulated CUDA OOM failover recovery
9. Camera disconnect / invalid frame fallback (LANE_LOST + Last Known Good)
10. Model file corruption resilience
11. BoundedHistoryBuffer capacity invariance
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from deploy.watchdog import ProcessWatchdog
from src.inference.memory_guard import BoundedHistoryBuffer, MemoryGuard
from src.inference.predictor import (
    BaseLanePredictor,
    DegradationLevel,
    DetectionStatus,
    GracefulDegradationPredictor,
    LanePrediction,
    ModelStatus,
)
from src.inference.preprocessing import PreprocessResult
from src.inference.safe_wrapper import SafeExecutor, safe_call, safe_execute
from src.utils.crash_logger import CrashLogger
from src.utils.health_check import HealthChecker, PreflightReport
from src.utils.thermal_guard import ThermalGuard, ThermalState


# ───────────────────────────────────────────────────────────────────────────
# 1. SafeExecutor Exception Shielding
# ───────────────────────────────────────────────────────────────────────────

def test_safe_executor_catches_all_exceptions():
    """Verify SafeExecutor catches arithmetic, runtime, and custom errors without crashing."""
    executor = SafeExecutor(name="test_shield", safe_default=42)

    def faulty_div():
        return 1 / 0

    def faulty_runtime():
        raise RuntimeError("GPU execution halted")

    res1 = executor.execute(faulty_div)
    assert res1 == 42
    assert executor.consecutive_failures == 1
    assert executor.total_failures == 1

    res2 = executor.execute(faulty_runtime)
    assert res2 == 42
    assert executor.consecutive_failures == 2
    assert executor.total_failures == 2

    # Success resets consecutive failures
    res3 = executor.execute(lambda: 100)
    assert res3 == 100
    assert executor.consecutive_failures == 0
    assert executor.total_successes == 1

    # Returns last valid result if available
    res4 = executor.execute(faulty_div)
    assert res4 == 100  # Last valid result retained!


def test_safe_executor_consecutive_failure_trigger():
    """Verify degradation callback fires once consecutive error threshold is met."""
    degradation_called = []

    def on_deg(consec, exc):
        degradation_called.append((consec, str(exc)))

    executor = SafeExecutor(
        name="test_cb",
        safe_default="SAFE",
        max_consecutive_failures=3,
        on_degradation=on_deg,
    )

    def bug():
        raise ValueError("Hardware bus timeout")

    for _ in range(5):
        executor.execute(bug)

    # Should trigger callback exactly once upon reaching threshold
    assert len(degradation_called) == 1
    assert degradation_called[0][0] == 3
    assert "Hardware bus timeout" in degradation_called[0][1]


# ───────────────────────────────────────────────────────────────────────────
# 2. 6-Level Graceful Degradation Stack
# ───────────────────────────────────────────────────────────────────────────

def test_graceful_degradation_hierarchy():
    """Verify predictor can transition between levels down to Level 6."""
    predictor = GracefulDegradationPredictor(initial_level=DegradationLevel.LEVEL_5_CLASSICAL_CV)
    assert predictor.current_level == DegradationLevel.LEVEL_5_CLASSICAL_CV

    # Create dummy preprocessed frame
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    prep = PreprocessResult(
        original_bgr=frame,
        resized_bgr=frame,
        input_tensor=frame.astype(np.float32) / 255.0,
        model_h=360,
        model_w=640,
    )

    pred = predictor.predict(prep)
    assert isinstance(pred, LanePrediction)
    assert pred.degradation_level in (DegradationLevel.LEVEL_5_CLASSICAL_CV, DegradationLevel.LEVEL_6_LAST_KNOWN_GOOD)


def test_cuda_oom_failover_recovery():
    """Verify that a simulated CUDA OOM exception degrades to the next level."""
    predictor = GracefulDegradationPredictor(initial_level=DegradationLevel.LEVEL_2_GPU_ONNX)

    # Mock level 2 predictor to raise CUDA out of memory
    mock_pred = MagicMock()
    mock_pred.predict.side_effect = RuntimeError("CUDA error: out of memory (allocate 2048 MB)")
    mock_pred.model_status = ModelStatus.READY
    predictor._predictors[DegradationLevel.LEVEL_2_GPU_ONNX] = mock_pred

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    prep = PreprocessResult(
        original_bgr=frame,
        resized_bgr=frame,
        input_tensor=frame.astype(np.float32) / 255.0,
        model_h=360,
        model_w=640,
    )

    pred = predictor.predict(prep)
    assert predictor.current_level.value > DegradationLevel.LEVEL_2_GPU_ONNX.value
    assert len(predictor.transitions) >= 1
    assert "CUDA OOM" in predictor.transitions[0]["reason"]


def test_camera_disconnect_fallback():
    """Verify that an invalid/dropped camera frame triggers Level 6 with LANE_LOST."""
    predictor = GracefulDegradationPredictor(initial_level=DegradationLevel.LEVEL_5_CLASSICAL_CV)

    # First cache a good prediction
    dummy_pred = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        left_mask=np.ones((360, 640), dtype=np.uint8) * 255,
        right_mask=np.ones((360, 640), dtype=np.uint8) * 255,
        left_confidence=0.9,
        right_confidence=0.9,
    )
    predictor.last_known_good = dummy_pred

    # Disconnected/empty frame
    bad_prep = PreprocessResult(error="Camera USB bus disconnected")
    pred = predictor.predict(bad_prep)

    assert pred.status == DetectionStatus.LANE_LOST
    assert pred.degradation_level == DegradationLevel.LEVEL_6_LAST_KNOWN_GOOD
    assert pred.is_degraded is True
    assert "Camera frame invalid" in pred.warning_banner


def test_model_corruption_resilience():
    """Verify that model file corruption fails over to Level 5 Classical CV."""
    predictor = GracefulDegradationPredictor(initial_level=DegradationLevel.LEVEL_2_GPU_ONNX)

    mock_onnx = MagicMock()
    mock_onnx.predict.side_effect = ValueError("Invalid model protobuf format: corrupted file header")
    mock_onnx.model_status = ModelStatus.READY
    predictor._predictors[DegradationLevel.LEVEL_2_GPU_ONNX] = mock_onnx

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    prep = PreprocessResult(
        original_bgr=frame,
        resized_bgr=frame,
        input_tensor=frame.astype(np.float32) / 255.0,
        model_h=360,
        model_w=640,
    )

    pred = predictor.predict(prep)
    # Should drop to Level 5 Classical CV directly
    assert predictor.current_level.value >= DegradationLevel.LEVEL_5_CLASSICAL_CV.value


# ───────────────────────────────────────────────────────────────────────────
# 3. MemoryGuard Monotonic Leak Detection & Reclamation
# ───────────────────────────────────────────────────────────────────────────

def test_memory_guard_leak_detection_and_cleanup(tmp_path):
    """Verify MemoryGuard detects monotonic memory growth and purges buffers."""
    csv_log = tmp_path / "test_mem.csv"
    persistent_called = []

    guard = MemoryGuard(
        check_interval_s=0.01,
        leak_window_size=4,
        leak_threshold_mb=20.0,
        persistent_leak_threshold=2,
        log_csv_path=csv_log,
        on_persistent_leak=lambda: persistent_called.append(True),
    )

    buffer = BoundedHistoryBuffer(max_size=50)
    for i in range(30):
        buffer.append(f"frame_{i}")
    guard.register_buffer(buffer)
    assert len(buffer) == 30

    # Simulate monotonic RAM growth
    with guard._lock:
        for i, rss in enumerate([100.0, 115.0, 130.0, 145.0, 160.0]):
            from src.inference.memory_guard import MemorySnapshot
            guard._history.append(MemorySnapshot(time.time() + i, rss, rss + 20))

    assert guard.detect_monotonic_growth() is True

    # Execute cleanup
    res = guard.check(force_sample=True)
    assert res["cleanup_triggered"] is True
    assert len(buffer) == 0  # Buffer cleared by cleanup action!

    # Simulate persistent leak
    guard.check(force_sample=True)
    assert len(persistent_called) >= 1


def test_bounded_history_buffer_capacity():
    """Verify BoundedHistoryBuffer strictly bounds depth to max_size."""
    buf = BoundedHistoryBuffer(max_size=10)
    for i in range(25):
        buf.append(i)

    assert len(buf) == 10
    assert buf.to_list() == list(range(15, 25))


# ───────────────────────────────────────────────────────────────────────────
# 4. ThermalGuard Throttling & ESTOP
# ───────────────────────────────────────────────────────────────────────────

def test_thermal_guard_throttling_states():
    """Verify thermal safety thresholds dynamically adapt target FPS."""
    guard = ThermalGuard(
        mild_threshold_c=80.0,
        heavy_threshold_c=85.0,
        estop_threshold_c=90.0,
        default_target_fps=30.0,
    )

    # 1. Normal temperature (45°C)
    state, fps, _ = guard.evaluate_thermal_state(temp_c=45.0)
    assert state == ThermalState.NORMAL
    assert fps == 30.0

    # 2. Mild throttling (82°C)
    state, fps, _ = guard.evaluate_thermal_state(temp_c=82.0)
    assert state == ThermalState.THROTTLE_MILD
    assert fps == 15.0

    # 3. Heavy throttling (87°C)
    state, fps, _ = guard.evaluate_thermal_state(temp_c=87.0)
    assert state == ThermalState.THROTTLE_HEAVY
    assert fps == 10.0

    # 4. Emergency Stop (92°C)
    state, fps, _ = guard.evaluate_thermal_state(temp_c=92.0)
    assert state == ThermalState.EMERGENCY_STOP
    assert fps == 0.0


def test_thermal_guard_training_cooldown():
    """Verify training cooldown trigger saves checkpoint."""
    guard = ThermalGuard(heavy_threshold_c=85.0)
    guard.simulate_temperature(88.0)

    checkpoint_saved = []

    with patch("time.sleep", return_value=None):
        triggered = guard.handle_training_cooldown(
            checkpoint_callback=lambda: checkpoint_saved.append(True)
        )

    assert triggered is True
    assert len(checkpoint_saved) == 1


# ───────────────────────────────────────────────────────────────────────────
# 5. CrashLogger Diagnostics
# ───────────────────────────────────────────────────────────────────────────

def test_crash_logger_report_generation(tmp_path):
    """Verify CrashLogger exports structured JSON and Markdown diagnostic reports."""
    crash_dir = tmp_path / "crashes"
    logger = CrashLogger(crash_dir=crash_dir)
    logger.update_model_info({"backend": "ONNX", "status": "READY"})
    logger.record_inference_event({"frame_id": 42, "inference_ms": 22.4})

    try:
        raise RuntimeError("Fatal camera sensor deserialization error")
    except Exception as exc:
        json_path, md_path = logger.generate_crash_report(exc, context="UnitTestContext")

    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["exception_type"] == "RuntimeError"
    assert "Fatal camera sensor" in data["exception_message"]
    assert data["context"] == "UnitTestContext"
    assert data["model_info"]["backend"] == "ONNX"
    assert "system_snapshot" in data

    md_content = md_path.read_text(encoding="utf-8")
    assert "Fatal camera sensor deserialization error" in md_content
    assert "## Stack Trace" in md_content


# ───────────────────────────────────────────────────────────────────────────
# 6. Startup HealthCheck
# ───────────────────────────────────────────────────────────────────────────

def test_startup_health_checker(tmp_path):
    """Verify pre-flight checks handle system validation."""
    checker = HealthChecker(min_disk_free_gb=0.1, min_ram_free_mb=50.0, project_root=tmp_path)
    report = checker.run_preflight_checks(strict=False)

    assert isinstance(report, PreflightReport)
    assert len(report.items) >= 4
    formatted = checker.format_health_report(report)
    assert "PRE-FLIGHT SYSTEM HEALTH REPORT" in formatted


# ───────────────────────────────────────────────────────────────────────────
# 7. ProcessWatchdog Liveness & Safe Mode
# ───────────────────────────────────────────────────────────────────────────

def test_process_watchdog_unresponsive_restart_and_safe_mode():
    """Verify watchdog detects unresponsive heartbeat and triggers safe mode after 3 restarts."""
    watchdog = ProcessWatchdog(
        command="python dummy.py",
        timeout_s=1.0,
        max_restarts=3,
        window_s=60.0,
        sim_mode=True,
    )
    watchdog.start_process()

    assert watchdog.is_heartbeat_fresh() is True

    # 1. Simulate process becoming unresponsive (> timeout)
    watchdog.set_sim_unresponsive()
    assert watchdog.is_heartbeat_fresh() is False

    # 2. Restart #1
    ok1 = watchdog.restart_process()
    assert ok1 is True
    assert watchdog.total_restarts == 1
    assert watchdog.is_safe_mode is False

    # 3. Restart #2
    ok2 = watchdog.restart_process()
    assert ok2 is True
    assert watchdog.total_restarts == 2
    assert watchdog.is_safe_mode is False

    # 4. Restart #3
    ok3 = watchdog.restart_process()
    assert ok3 is True
    assert watchdog.total_restarts == 3

    # 5. 4th failure within window -> Safe Mode Trigger!
    ok4 = watchdog.restart_process()
    assert ok4 is False
    assert watchdog.is_safe_mode is True
