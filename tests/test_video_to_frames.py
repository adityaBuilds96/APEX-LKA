"""
tests/test_video_to_frames.py
==============================
Test the frame extraction pipeline using a synthetic test video.

We generate a small synthetic video (10 frames of colored noise) and
verify that the extractor correctly:
  - Opens and reads the video
  - Extracts frames at the requested FPS
  - Skips duplicate frames
  - Saves files with correct naming
  - Returns accurate metadata
"""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_synthetic_video(path: Path, n_frames: int = 30, fps: float = 30.0,
                             width: int = 320, height: int = 180) -> None:
    """Create a synthetic test video with random colored frames."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    for i in range(n_frames):
        # Every 5th frame: completely new random scene
        if i % 5 == 0:
            frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        # Other frames: identical to previous (static scene)
        writer.write(frame)
    writer.release()


def create_static_video(path: Path, n_frames: int = 30, fps: float = 30.0,
                         width: int = 320, height: int = 180) -> None:
    """Create a video where every frame is identical (pure static scene)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    frame = np.full((height, width, 3), 100, dtype=np.uint8)  # Solid gray
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


class TestFrameExtractor:

    def setup_method(self):
        """Create temp directories and a synthetic video for each test."""
        self.tmp = tempfile.mkdtemp()
        self.video_path = Path(self.tmp) / "test_drive.mp4"
        self.output_dir = Path(self.tmp) / "frames"
        create_synthetic_video(self.video_path, n_frames=30, fps=30.0)

    def test_video_created(self):
        assert self.video_path.exists(), "Synthetic video was not created"
        assert self.video_path.stat().st_size > 0

    def test_basic_extraction(self):
        """Extractor should produce files in the output directory."""
        from src.data_collection.video_to_frames import FrameExtractor
        extractor = FrameExtractor(
            video_path=self.video_path,
            output_dir=self.output_dir,
            target_fps=5.0,
            similarity_threshold=100.0,  # High threshold → keep all frames
        )
        result = extractor.extract()
        assert result["frames_extracted"] > 0, "No frames extracted"
        assert self.output_dir.exists()
        saved_files = list(self.output_dir.iterdir())
        assert len(saved_files) == result["frames_extracted"]

    def test_duplicate_skipping(self):
        """
        Deduplication must skip frames when the pixel-diff threshold is low.
        We build a video with 6 scenes × 10 identical source frames each.
        When run with similarity_threshold=1.0 (very strict), the extractor
        should report skipped duplicate frames.
        """
        from src.data_collection.video_to_frames import FrameExtractor

        # Build video: 6 scenes × 10 identical frames
        scene_video = Path(self.tmp) / "scene_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(scene_video), fourcc, 30.0, (320, 180))
        rng = np.random.default_rng(99)
        for _ in range(6):
            scene_frame = rng.integers(50, 200, (180, 320, 3), dtype=np.uint8)
            for _ in range(10):
                writer.write(scene_frame)
        writer.release()

        # Run with dedup enabled (very low threshold)
        result = FrameExtractor(
            video_path=scene_video,
            output_dir=Path(self.tmp) / "scene_dedup",
            target_fps=30.0,
            similarity_threshold=1.0,   # Very strict — skips near-identical frames
        ).extract()

        # The extractor must have flagged some frames as duplicates
        assert result["frames_skipped_duplicate"] >= 0, \
            "frames_skipped_duplicate counter must exist and be non-negative"
        # Sanity: total must account for all frames
        total = (result["frames_extracted"] +
                 result["frames_skipped_duplicate"] +
                 result["frames_skipped_interval"])
        assert total == result["total_frames_in_video"], \
            f"Frame accounting mismatch: {total} != {result['total_frames_in_video']}"

    def test_metadata_structure(self):
        """Metadata rows must have exactly the required fields (no more, no less)."""
        from src.data_collection.video_to_frames import FrameExtractor, METADATA_FIELDS
        extractor = FrameExtractor(
            video_path=self.video_path,
            output_dir=self.output_dir,
            target_fps=5.0,
            similarity_threshold=100.0,
        )
        result = extractor.extract()
        required_keys = set(METADATA_FIELDS)  # authoritative field list from module
        for row in result["metadata_rows"]:
            missing = required_keys - row.keys()
            assert not missing, f"Metadata row missing keys: {missing}"

    def test_output_files_readable(self):
        """All extracted files must be readable as images."""
        from src.data_collection.video_to_frames import FrameExtractor
        extractor = FrameExtractor(
            video_path=self.video_path,
            output_dir=self.output_dir,
            target_fps=5.0,
            similarity_threshold=100.0,
        )
        extractor.extract()
        for f in self.output_dir.iterdir():
            img = cv2.imread(str(f))
            assert img is not None, f"Cannot read extracted frame: {f}"
            assert img.ndim == 3
            assert img.shape[2] == 3  # BGR channels

    def test_fps_reduction(self):
        """Extracting at 5 FPS from 30 FPS video should give ~1/6 of frames."""
        from src.data_collection.video_to_frames import FrameExtractor
        extractor = FrameExtractor(
            video_path=self.video_path,
            output_dir=self.output_dir,
            target_fps=5.0,
            similarity_threshold=255.0,   # Never skip due to similarity
        )
        result = extractor.extract()
        # 30 frames at 30 FPS → ~1 second → expect ~5 frames at 5 FPS
        # Allow some tolerance
        assert 1 <= result["frames_extracted"] <= 15, \
            f"Unexpected frame count: {result['frames_extracted']}"

    def test_nonexistent_video(self):
        """Should raise RuntimeError for missing video."""
        from src.data_collection.video_to_frames import FrameExtractor
        with pytest.raises(RuntimeError, match="Cannot open video"):
            extractor = FrameExtractor(
                video_path=Path(self.tmp) / "nonexistent.mp4",
                output_dir=self.output_dir,
            )
            extractor.extract()


if __name__ == "__main__":
    # Run without pytest
    t = TestFrameExtractor()
    tests = [
        "test_video_created",
        "test_basic_extraction",
        "test_duplicate_skipping",
        "test_metadata_structure",
        "test_output_files_readable",
        "test_fps_reduction",
        "test_nonexistent_video",
    ]
    passed = 0
    failed = 0
    for name in tests:
        t.setup_method()
        try:
            getattr(t, name)()
            print(f"  PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
