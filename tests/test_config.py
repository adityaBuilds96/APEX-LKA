"""
tests/test_config.py
====================
Test that the project configuration loads correctly and all paths resolve.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_config_loads():
    """Config YAML must load without error."""
    from src.config import cfg
    assert isinstance(cfg, dict), "Config must be a dict"
    assert "project" in cfg
    assert "paths" in cfg
    assert "data_collection" in cfg
    assert "model" in cfg
    assert "training" in cfg


def test_paths_object():
    """PATHS object must have correct attributes."""
    from src.config import PATHS
    required = [
        "raw_videos", "raw_frames", "annotated",
        "train", "val", "test",
        "annotations", "models", "checkpoints",
        "logs", "results"
    ]
    for attr in required:
        assert hasattr(PATHS, attr), f"PATHS missing attribute: {attr}"
        path = getattr(PATHS, attr)
        assert isinstance(path, Path), f"PATHS.{attr} must be a Path"


def test_ensure_dirs():
    """ensure_dirs must create all directories without error."""
    from src.config import ensure_dirs
    ensure_dirs()
    from src.config import PATHS
    assert PATHS.raw_videos.exists()
    assert PATHS.raw_frames.exists()
    assert PATHS.annotated.exists()


def test_data_collection_config():
    """Data collection config must have required keys."""
    from src.config import cfg
    dc = cfg["data_collection"]
    assert "target_fps" in dc
    assert "duplicate_threshold" in dc
    assert dc["target_fps"] > 0
    assert 0 < dc["duplicate_threshold"] <= 1.0


def test_model_config():
    """Model config must specify num_classes correctly."""
    from src.config import cfg
    mc = cfg["model"]
    assert mc["num_classes"] == 3, "Must have 3 classes: bg, left_lane, right_lane"
    assert mc["input_channels"] == 3


def test_steering_convention():
    """
    Verify the sign convention documentation matches the config values.
    lateral_error > 0 → vehicle right of center → steer left
    """
    from src.config import cfg
    sc = cfg["steering"]
    assert sc["Kp"] > 0, "Kp must be positive"
    assert sc["Kd"] >= 0, "Kd must be non-negative"
    assert sc["max_steering_command"] > 0


if __name__ == "__main__":
    # Run tests directly without pytest
    tests = [
        test_config_loads,
        test_paths_object,
        test_ensure_dirs,
        test_data_collection_config,
        test_model_config,
        test_steering_convention,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
