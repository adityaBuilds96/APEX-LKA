"""
tests/test_data_collector.py
============================
Unit tests for Closed-Loop Edge Case Data Harvester:
- Trigger detection: Low confidence, RSS intervention, Driver override, Shadow divergence
- 5-frame temporal window bundling [T-2, T-1, T, T+1, T+2]
- Package serialization (frames, depth map, sensor JSON, package_manifest.json)
- Storage quota enforcement via FIFO rolling deletion (< 2.0 GB limit)
"""

import json
import time
from pathlib import Path
import numpy as np
import pytest

from src.fleet.data_collector import (
    ClosedLoopDataHarvester,
    HarvestTriggerReason,
    HarvestEvent,
)


@pytest.fixture
def harvester_fixture(tmp_path):
    """Fixture providing isolated data harvester with small storage quota."""
    output_dir = tmp_path / "harvested"
    # Small quota: 2 MB for fast FIFO purge testing
    harvester = ClosedLoopDataHarvester(
        output_dir=output_dir,
        storage_quota_mb=2,
        trigger_confidence_low=0.40,
        trigger_divergence_m=0.40,
        pre_event_frames=2,
        post_event_frames=2,
        consecutive_low_conf=3,
    )
    return {
        "harvester": harvester,
        "output_dir": output_dir,
        "tmp_path": tmp_path,
    }


def test_trigger_low_confidence_consecutive_frames(harvester_fixture):
    """Verify trigger fires on >= 3 consecutive frames with confidence < 0.40."""
    harvester = harvester_fixture["harvester"]
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Frame 0: normal
    evt0 = harvester.observe(dummy_frame, confidence=0.85, offset_m=0.0, frame_idx=0)
    assert evt0 is None

    # Frame 1: low (1st)
    evt1 = harvester.observe(dummy_frame, confidence=0.35, offset_m=0.0, frame_idx=1)
    assert evt1 is None

    # Frame 2: low (2nd)
    evt2 = harvester.observe(dummy_frame, confidence=0.32, offset_m=0.0, frame_idx=2)
    assert evt2 is None

    # Frame 3: low (3rd) -> Triggers capture, begins post-event collection
    evt3 = harvester.observe(dummy_frame, confidence=0.30, offset_m=0.0, frame_idx=3)
    assert evt3 is not None
    assert evt3.trigger_reason == HarvestTriggerReason.LOW_CONFIDENCE

    # Complete post-event frames (2 post-event frames)
    evt4 = harvester.observe(dummy_frame, confidence=0.45, offset_m=0.0, frame_idx=4)
    evt5 = harvester.observe(dummy_frame, confidence=0.50, offset_m=0.0, frame_idx=5)

    # Check packaged event folder created
    event_folders = list(harvester.output_dir.glob("event_*"))
    assert len(event_folders) == 1
    manifest_path = event_folders[0] / "package_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["trigger_reason"] == HarvestTriggerReason.LOW_CONFIDENCE.value
        assert data["total_frames"] == 5  # 2 pre + 1 trigger + 2 post


def test_trigger_rss_intervention(harvester_fixture):
    """Verify RSS shield intervention immediately triggers capture."""
    harvester = harvester_fixture["harvester"]
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    evt = harvester.observe(
        dummy_frame,
        confidence=0.80,
        offset_m=0.0,
        frame_idx=10,
        rss_override_active=True,
    )
    assert evt is not None
    assert evt.trigger_reason == HarvestTriggerReason.RSS_INTERVENTION


def test_trigger_driver_manual_override(harvester_fixture):
    """Verify manual override pin toggle immediately triggers capture."""
    harvester = harvester_fixture["harvester"]
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    evt = harvester.observe(
        dummy_frame,
        confidence=0.85,
        offset_m=0.0,
        frame_idx=20,
        driver_override_active=True,
    )
    assert evt is not None
    assert evt.trigger_reason == HarvestTriggerReason.DRIVER_OVERRIDE


def test_trigger_shadow_model_divergence(harvester_fixture):
    """Verify shadow lateral divergence > 0.40m triggers capture."""
    harvester = harvester_fixture["harvester"]
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    evt = harvester.observe(
        dummy_frame,
        confidence=0.85,
        offset_m=0.10,
        frame_idx=30,
        shadow_divergence_m=0.55,  # > 0.40m
    )
    assert evt is not None
    assert evt.trigger_reason == HarvestTriggerReason.SHADOW_DIVERGENCE


def test_package_content_serialization(harvester_fixture):
    """Verify package contains raw images, depth maps, and metadata JSON."""
    harvester = harvester_fixture["harvester"]
    dummy_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    dummy_depth = np.full((120, 160), 15.0, dtype=np.float32)

    # Populate 2 pre-event frames
    harvester.observe(dummy_frame, confidence=0.8, offset_m=0.0, frame_idx=1)
    harvester.observe(dummy_frame, confidence=0.8, offset_m=0.0, frame_idx=2)

    # Trigger
    harvester.observe(
        dummy_frame,
        confidence=0.8,
        offset_m=0.0,
        frame_idx=3,
        depth_map=dummy_depth,
        rss_override_active=True,
    )

    # 2 post-event frames
    harvester.observe(dummy_frame, confidence=0.8, offset_m=0.0, frame_idx=4)
    harvester.observe(dummy_frame, confidence=0.8, offset_m=0.0, frame_idx=5)

    event_folders = list(harvester.output_dir.glob("event_*"))
    assert len(event_folders) == 1
    pkg_dir = event_folders[0]

    # Check frame images exist
    frames = list(pkg_dir.glob("frame_*.jpg"))
    assert len(frames) == 5

    # Check depth map exists for frame with depth
    depth_maps = list(pkg_dir.glob("depth_*.npy"))
    assert len(depth_maps) >= 1

    # Check sensor telemetry JSON exists
    telemetry_files = list(pkg_dir.glob("telemetry_*.json"))
    assert len(telemetry_files) == 5


def test_rolling_fifo_storage_quota_enforcement(harvester_fixture):
    """
    CRITICAL HARD CONSTRAINT:
    Hard limit enforced using FIFO rolling deletion to prevent SD card overflow.
    """
    harvester = harvester_fixture["harvester"]
    output_dir = harvester_fixture["output_dir"]

    # Set tiny quota: 50 KB
    harvester.storage_quota_bytes = 50 * 1024

    # Create 4 synthetic packages with staggered timestamps
    pkg_dirs = []
    for i in range(4):
        d = output_dir / f"event_20261011_00000{i}"
        d.mkdir(parents=True, exist_ok=True)
        # Create a 20 KB dummy file in each package (total 80 KB > 50 KB quota)
        dummy_file = d / "data.bin"
        dummy_file.write_bytes(b"X" * (20 * 1024))
        # Ensure distinct modification times
        time.sleep(0.02)
        pkg_dirs.append(d)

    # Enforce quota
    harvester._enforce_storage_quota()

    # Remaining packages should strictly occupy <= 50 KB
    remaining = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("event_")]
    assert len(remaining) < 4
    # The oldest directory (event_20261011_000000) should have been purged first
    assert not (output_dir / "event_20261011_000000").exists()
    assert (output_dir / "event_20261011_000003").exists()

    total_bytes = harvester._get_dir_size(output_dir)
    assert total_bytes <= harvester.storage_quota_bytes
