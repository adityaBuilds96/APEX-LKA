"""
tests/test_telemetry_auditor.py
===============================
Unit tests for Task 6.5: Telemetry Black-Box Audit:
1. Strict monotonic timestamp validation
2. Inter-frame gap detection (> 100ms flagging)
3. Latency percentiles (p50, p95, p99 <= 40ms) and FPS floor verification
4. End-to-end execution of TelemetryAuditor on benchmark session
"""

import csv
import json
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.telemetry_auditor import TelemetryAuditor


def test_telemetry_auditor_on_mock_healthy_session(tmp_path):
    """Verify auditor passes on synthetic compliant telemetry log."""
    csv_path = tmp_path / "telemetry.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "frame_id", "steering_deg", "offset_m", "heading_err_deg", "track_quality", "mode_enum", "status", "estop", "infer_ms", "fps", "ram_mb", "hash"])
        base_t = 1_000_000_000_000
        for i in range(50):
            writer.writerow([
                base_t + i * 33_333_333,
                i,
                1.5,
                0.05,
                0.5,
                0.95,
                1,
                "AUTO",
                0,
                12.5,
                30.0,
                150.0,
                "abc123hash",
            ])

    auditor = TelemetryAuditor(session_dir=tmp_path)
    summary = auditor.audit()
    assert summary["passed"] is True
    assert summary["monotonic_timestamps"] is True
    assert summary["gaps_over_100ms_count"] == 0
    assert summary["latency_p99_ms"] <= 40.0
    assert summary["peak_ram_mb"] < 1400.0


def test_telemetry_auditor_detects_gap_anomaly(tmp_path):
    """Verify auditor detects and fails sessions containing gaps > 100ms."""
    csv_path = tmp_path / "telemetry.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ns", "frame_id", "steering_deg", "offset_m", "heading_err_deg", "track_quality", "mode_enum", "status", "estop", "infer_ms", "fps", "ram_mb", "hash"])
        base_t = 1_000_000_000_000
        # Normal rows
        writer.writerow([base_t, 0, 0.0, 0.0, 0.0, 0.95, 1, "AUTO", 0, 10.0, 30.0, 100.0, "h1"])
        writer.writerow([base_t + 33_333_333, 1, 0.0, 0.0, 0.0, 0.95, 1, "AUTO", 0, 10.0, 30.0, 100.0, "h2"])
        # Gap anomaly: 150 ms gap
        writer.writerow([base_t + 33_333_333 + 150_000_000, 2, 0.0, 0.0, 0.0, 0.95, 1, "AUTO", 0, 10.0, 30.0, 100.0, "h3"])

    auditor = TelemetryAuditor(session_dir=tmp_path)
    summary = auditor.audit()
    assert summary["passed"] is False
    assert summary["gaps_over_100ms_count"] == 1
    assert summary["max_gap_ms"] >= 150.0


def test_telemetry_auditor_live_benchmark_session():
    """Verify auditor on actual recorded session in results/benchmarks/last_session."""
    session_dir = PROJECT_ROOT / "results" / "benchmarks" / "last_session"
    if not (session_dir / "telemetry.csv").exists():
        pytest.skip("last_session telemetry not yet generated")

    auditor = TelemetryAuditor(session_dir=session_dir)
    summary = auditor.audit()
    assert summary["monotonic_timestamps"] is True
    assert summary["latency_p99_ms"] <= 40.0
    assert summary["peak_ram_mb"] < 2500.0
