"""
src/data/format_adapters.py
===========================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Multi-Format Perception Annotation Adapters & Polyline Normalizers.
ISO 26262 ASIL-B Data Ingestion Standard for Autonomous Ground Vehicles.

Supported Data Formats:
-----------------------
1. TuSimple Benchmark Format:
   - Line-delimited JSON (`label_data_*.json`, `test_label.json`)
   - `h_samples`: List of Y-coordinates (height samples)
   - `lanes`: Nested list of X-coordinates corresponding to each Y sample (-2 = absent)
   - `raw_file`: Relative path to keyframe (typically 20th frame in clip)
2. Semantic PNG Masks:
   - 8-bit single-channel indexed ground truth masks (Classes 0, 1, 2, 3)
   - 24-bit RGB color-coded masks
3. COCO Format:
   - Category-indexed polygon segmentation structures
4. YOLO Format:
   - Normalized bounding box and normalized polygon coordinate sequences
5. Raw Image & Video Streams:
   - Single JPEG/PNG frames and MP4/AVI sequential video streams
"""

from __future__ import annotations

import json
import logging
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema

logger = logging.getLogger("APEX.FormatAdapters")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Coordinate & Polyline Representation Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Point2D:
    """2D floating-point coordinate representation in pixel space."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> Tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))


@dataclass
class LanePolyline:
    """
    Continuous lane boundary marking polyline with semantic role and confidence.
    """
    points: List[Point2D] = field(default_factory=list)
    semantic_class: LaneClassID = LaneClassID.BACKGROUND
    confidence: float = 1.0
    lane_id: int = -1
    is_continuous: bool = True

    @property
    def num_points(self) -> int:
        return len(self.points)

    @property
    def is_valid(self) -> bool:
        return len(self.points) >= 2

    def to_numpy(self) -> np.ndarray:
        """Convert points to [N, 2] float32 array."""
        if not self.points:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float32)

    def to_int_contour(self) -> np.ndarray:
        """Convert points to OpenCV contour shape [N, 1, 2] int32."""
        pts = [[int(round(p.x)), int(round(p.y))] for p in self.points]
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    def get_x_at_y(self, y_target: float) -> Optional[float]:
        """Interpolate X coordinate at a given Y height."""
        if len(self.points) < 2:
            return None
        # Points are typically sorted top-to-bottom (increasing Y)
        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]
            y_min, y_max = min(p1.y, p2.y), max(p1.y, p2.y)
            if y_min <= y_target <= y_max:
                if abs(p2.y - p1.y) < 1e-5:
                    return (p1.x + p2.x) / 2.0
                t = (y_target - p1.y) / (p2.y - p1.y)
                return p1.x + t * (p2.x - p1.x)
        return None


@dataclass
class NormalizedSample:
    """
    Standardized intermediate container uniting annotations across all source formats.
    """
    sample_id: str
    image_path: str
    image_width: int
    image_height: int
    polylines: List[LanePolyline] = field(default_factory=list)
    sequence_id: Optional[str] = None
    precomputed_mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_polylines(self) -> bool:
        return len(self.polylines) > 0

    @property
    def has_mask(self) -> bool:
        return self.precomputed_mask is not None


# ──────────────────────────────────────────────────────────────────────────────
# 2. Abstract Base Adapter Interface
# ──────────────────────────────────────────────────────────────────────────────

class BaseFormatAdapter(ABC):
    """
    Abstract contract for parsing domain-specific dataset annotations into
    normalized APEX-Vision polyline and mask representations.
    """

    def __init__(self, target_width: int = 640, target_height: int = 360) -> None:
        self.target_width = target_width
        self.target_height = target_height
        self.schema = get_default_label_schema()

    @abstractmethod
    def parse_source(self, raw_data: Union[str, bytes, Path, Dict[str, Any]]) -> List[NormalizedSample]:
        """Parse raw dataset payload into a list of NormalizedSample objects."""
        pass

    @abstractmethod
    def identify_format(self, candidate: Any) -> bool:
        """Determine whether candidate data matches this adapter's schema."""
        pass


# ──────────────────────────────────────────────────────────────────────────────
# 3. TuSimple Benchmark Format Adapter
# ──────────────────────────────────────────────────────────────────────────────

class TuSimpleAdapter(BaseFormatAdapter):
    """
    High-performance parser for TuSimple lane detection benchmark annotations.
    Handles JSON files where each line is:
      {"lanes": [...], "h_samples": [...], "raw_file": "..."}

    Lanes are represented as arrays of X coordinates evaluated at each Y in `h_samples`.
    A value of -2 indicates that the lane boundary is absent/unobserved at that height.
    """

    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 360,
        original_width: int = 1280,
        original_height: int = 720,
        min_points_per_lane: int = 3,
        camera_center_x_ratio: float = 0.50,
    ) -> None:
        super().__init__(target_width=target_width, target_height=target_height)
        self.orig_w = original_width
        self.orig_h = original_height
        self.min_pts = min_points_per_lane
        self.center_x_ratio = camera_center_x_ratio

    def identify_format(self, candidate: Any) -> bool:
        """Check if candidate content or path matches TuSimple schema."""
        if isinstance(candidate, (str, bytes)):
            text = candidate.decode("utf-8", errors="ignore") if isinstance(candidate, bytes) else candidate
            first_line = text.strip().split("\n")[0]
            try:
                data = json.loads(first_line)
                return "lanes" in data and "h_samples" in data and "raw_file" in data
            except Exception:
                return False
        elif isinstance(candidate, dict):
            return "lanes" in candidate and "h_samples" in candidate and "raw_file" in candidate
        return False

    def parse_source(self, raw_data: Union[str, bytes, Path, Dict[str, Any]]) -> List[NormalizedSample]:
        """
        Parse TuSimple raw JSON content (file path, raw text, or memory dict).
        """
        records: List[Dict[str, Any]] = []

        if isinstance(raw_data, Path) or (isinstance(raw_data, str) and "\n" not in raw_data and Path(raw_data).exists()):
            file_path = Path(raw_data)
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as err:
                            logger.warning(f"Skipping malformed JSON line in {file_path.name}: {err}")
        elif isinstance(raw_data, (str, bytes)):
            text = raw_data.decode("utf-8", errors="ignore") if isinstance(raw_data, bytes) else raw_data
            for line in text.strip().split("\n"):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        elif isinstance(raw_data, dict):
            records.append(raw_data)
        elif isinstance(raw_data, list):
            records = raw_data

        parsed_samples: List[NormalizedSample] = []
        for idx, rec in enumerate(records):
            sample = self._parse_single_record(rec, fallback_idx=idx)
            if sample is not None:
                parsed_samples.append(sample)

        return parsed_samples

    def _parse_single_record(self, record: Dict[str, Any], fallback_idx: int) -> Optional[NormalizedSample]:
        """
        Parse and sort lanes from a single TuSimple dictionary entry.
        """
        lanes = record.get("lanes", [])
        h_samples = record.get("h_samples", [])
        raw_file = record.get("raw_file", f"frame_{fallback_idx:06d}.jpg")

        if not lanes or not h_samples:
            return None

        # Build raw polylines
        raw_polylines: List[Tuple[float, LanePolyline]] = []
        for lane_idx, lane_x in enumerate(lanes):
            valid_pts: List[Point2D] = []
            for x_val, y_val in zip(lane_x, h_samples):
                if x_val >= 0 and y_val >= 0:
                    valid_pts.append(Point2D(x=float(x_val), y=float(y_val)))

            if len(valid_pts) >= self.min_pts:
                # Calculate mean x near bottom (ego horizon reference)
                bottom_pts = [p.x for p in valid_pts[-3:]]
                ref_x = sum(bottom_pts) / len(bottom_pts)
                poly = LanePolyline(
                    points=valid_pts,
                    semantic_class=LaneClassID.BACKGROUND, # Will be assigned below
                    lane_id=lane_idx,
                )
                raw_polylines.append((ref_x, poly))

        # Generate deterministic, unique sample_id preserving clip identity
        p_raw = Path(raw_file)
        clean_parts = [part for part in p_raw.parts if part not in ("clips", "TUSimple", "train_set", "test_set")]
        if len(clean_parts) >= 2:
            base_id = "_".join(clean_parts)
            sample_id = Path(base_id).stem
        else:
            sample_id = f"{p_raw.stem}_{fallback_idx:05d}"

        if not raw_polylines:
            return NormalizedSample(
                sample_id=sample_id,
                image_path=raw_file,
                image_width=self.orig_w,
                image_height=self.orig_h,
                polylines=[],
            )

        # Sort lanes from left to right based on bottom lateral coordinate
        raw_polylines.sort(key=lambda item: item[0])
        sorted_polylines = [item[1] for item in raw_polylines]

        # Identify ego left and ego right lanes relative to image center
        assigned_polylines = self._classify_ego_boundaries(sorted_polylines, image_width=self.orig_w)

        return NormalizedSample(
            sample_id=sample_id,
            image_path=raw_file,
            image_width=self.orig_w,
            image_height=self.orig_h,
            polylines=assigned_polylines,
            metadata={"source_format": "tusimple", "original_lane_count": len(lanes), "raw_file": raw_file},
        )

    def _classify_ego_boundaries(
        self,
        sorted_polylines: List[LanePolyline],
        image_width: int,
    ) -> List[LanePolyline]:
        """
        Assign Left Boundary (Class 2) and Right Boundary (Class 3) to the
        immediate lane pair bracketing the ego vehicle lateral centerline.
        """
        center_x = float(image_width) * self.center_x_ratio

        # Find closest lane to left of center and closest lane to right of center
        left_candidates = []
        right_candidates = []

        for poly in sorted_polylines:
            bottom_pts = [p.x for p in poly.points[-3:]]
            avg_x = sum(bottom_pts) / len(bottom_pts)
            if avg_x < center_x:
                left_candidates.append((avg_x, poly))
            else:
                right_candidates.append((avg_x, poly))

        # Ego left is the rightmost lane among left candidates (closest to center)
        if left_candidates:
            left_candidates.sort(key=lambda item: item[0])
            ego_left = left_candidates[-1][1]
            ego_left.semantic_class = LaneClassID.LEFT_BOUNDARY
        else:
            ego_left = None

        # Ego right is the leftmost lane among right candidates (closest to center)
        if right_candidates:
            right_candidates.sort(key=lambda item: item[0])
            ego_right = right_candidates[0][1]
            ego_right.semantic_class = LaneClassID.RIGHT_BOUNDARY
        else:
            ego_right = None

        # If one side is missing but multiple candidates exist, assign adjacent pair
        if ego_left is not None and ego_right is None:
            if len(left_candidates) >= 2:
                ego_left = left_candidates[-2][1]
                ego_left.semantic_class = LaneClassID.LEFT_BOUNDARY
                ego_right = left_candidates[-1][1]
                ego_right.semantic_class = LaneClassID.RIGHT_BOUNDARY
        elif ego_right is not None and ego_left is None:
            if len(right_candidates) >= 2:
                ego_left = right_candidates[0][1]
                ego_left.semantic_class = LaneClassID.LEFT_BOUNDARY
                ego_right = right_candidates[1][1]
                ego_right.semantic_class = LaneClassID.RIGHT_BOUNDARY

        # If only lanes on one side exist and count < 2 or fallback needed
        if ego_left is None and ego_right is None and sorted_polylines:
            if len(sorted_polylines) >= 2:
                sorted_polylines[0].semantic_class = LaneClassID.LEFT_BOUNDARY
                sorted_polylines[1].semantic_class = LaneClassID.RIGHT_BOUNDARY
            else:
                sorted_polylines[0].semantic_class = LaneClassID.LEFT_BOUNDARY

        return sorted_polylines


# ──────────────────────────────────────────────────────────────────────────────
# 4. Semantic PNG Mask Format Adapter
# ──────────────────────────────────────────────────────────────────────────────

class SemanticMaskAdapter(BaseFormatAdapter):
    """
    Adapter for datasets that provide pre-rasterized 8-bit or RGB PNG masks.
    """

    def identify_format(self, candidate: Any) -> bool:
        if isinstance(candidate, np.ndarray) and candidate.ndim in (2, 3):
            return True
        if isinstance(candidate, (str, Path)):
            return str(candidate).lower().endswith(".png")
        return False

    def parse_source(self, raw_data: Union[str, bytes, Path, np.ndarray]) -> List[NormalizedSample]:
        """
        Load and validate a pre-rasterized mask array or image file.
        """
        mask_arr: Optional[np.ndarray] = None
        source_path = "memory_mask.png"

        if isinstance(raw_data, np.ndarray):
            mask_arr = raw_data
        elif isinstance(raw_data, (str, Path)):
            p = Path(raw_data)
            source_path = str(p)
            mask_arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        elif isinstance(raw_data, bytes):
            nparr = np.frombuffer(raw_data, np.uint8)
            mask_arr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if mask_arr is None:
            raise ValueError("Failed to decode mask array from provided source.")

        # If RGB mask, convert color codes to class indices
        if mask_arr.ndim == 3 and mask_arr.shape[2] >= 3:
            mask_2d = self._convert_rgb_mask_to_indexed(mask_arr)
        else:
            mask_2d = mask_arr.astype(np.uint8)

        sample = NormalizedSample(
            sample_id=Path(source_path).stem,
            image_path=source_path,
            image_width=mask_2d.shape[1],
            image_height=mask_2d.shape[0],
            polylines=[],
            precomputed_mask=mask_2d,
            metadata={"source_format": "semantic_png"},
        )
        return [sample]

    def _convert_rgb_mask_to_indexed(self, rgb_mask: np.ndarray) -> np.ndarray:
        """Map RGB color coded mask to 4-class {0, 1, 2, 3} indexed mask."""
        h, w = rgb_mask.shape[:2]
        indexed = np.zeros((h, w), dtype=np.uint8)

        # Match schema colors
        for cid in LaneClassID.all_ids():
            bgr_color = self.schema.get_color_bgr(cid)
            match = np.all(rgb_mask[:, :, :3] == bgr_color, axis=-1)
            indexed[match] = cid

        return indexed


# ──────────────────────────────────────────────────────────────────────────────
# 5. COCO Polygon Annotation Adapter
# ──────────────────────────────────────────────────────────────────────────────

class COCOFormatAdapter(BaseFormatAdapter):
    """
    Parser for COCO-style JSON polygon datasets containing lane categories.
    """

    def identify_format(self, candidate: Any) -> bool:
        if isinstance(candidate, dict):
            return "images" in candidate and "annotations" in candidate and "categories" in candidate
        if isinstance(candidate, (str, bytes)):
            text = candidate.decode("utf-8", errors="ignore") if isinstance(candidate, bytes) else candidate
            try:
                data = json.loads(text)
                return "images" in data and "annotations" in data and "categories" in data
            except Exception:
                return False
        return False

    def parse_source(self, raw_data: Union[str, bytes, Path, Dict[str, Any]]) -> List[NormalizedSample]:
        data: Dict[str, Any] = {}
        if isinstance(raw_data, (str, Path)) and Path(str(raw_data)).exists():
            with open(raw_data, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(raw_data, (str, bytes)):
            text = raw_data.decode("utf-8", errors="ignore") if isinstance(raw_data, bytes) else raw_data
            data = json.loads(text)
        elif isinstance(raw_data, dict):
            data = raw_data

        images_info = {img["id"]: img for img in data.get("images", [])}
        categories = {cat["id"]: cat["name"].lower() for cat in data.get("categories", [])}

        annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)

        samples: List[NormalizedSample] = []
        for img_id, img_info in images_info.items():
            file_name = img_info.get("file_name", f"coco_{img_id}.jpg")
            w = img_info.get("width", self.target_width)
            h = img_info.get("height", self.target_height)

            polylines: List[LanePolyline] = []
            for ann in annotations_by_image.get(img_id, []):
                cat_name = categories.get(ann.get("category_id"), "")
                assigned_class = self._map_coco_category_to_schema(cat_name)

                seg = ann.get("segmentation", [])
                if isinstance(seg, list):
                    for poly_coords in seg:
                        if isinstance(poly_coords, list) and len(poly_coords) >= 4:
                            pts = [
                                Point2D(x=float(poly_coords[i]), y=float(poly_coords[i + 1]))
                                for i in range(0, len(poly_coords) - 1, 2)
                            ]
                            polylines.append(LanePolyline(points=pts, semantic_class=assigned_class))

            samples.append(
                NormalizedSample(
                    sample_id=str(img_id),
                    image_path=file_name,
                    image_width=w,
                    image_height=h,
                    polylines=polylines,
                    metadata={"source_format": "coco"},
                )
            )

        return samples

    def _map_coco_category_to_schema(self, category_name: str) -> LaneClassID:
        name = category_name.lower()
        if "left" in name:
            return LaneClassID.LEFT_BOUNDARY
        elif "right" in name:
            return LaneClassID.RIGHT_BOUNDARY
        elif "corridor" in name or "road" in name or "drivable" in name:
            return LaneClassID.DRIVABLE_CORRIDOR
        return LaneClassID.BACKGROUND


# ──────────────────────────────────────────────────────────────────────────────
# 6. YOLO Polygon / Segmentation Adapter
# ──────────────────────────────────────────────────────────────────────────────

class YOLOFormatAdapter(BaseFormatAdapter):
    """
    Parser for YOLOv8/v9 segmentation format (.txt label files).
    Format per line: <class-index> <x1> <y1> <x2> <y2> ... (normalized 0.0 - 1.0)
    """

    def identify_format(self, candidate: Any) -> bool:
        if isinstance(candidate, str) and not candidate.startswith("{"):
            lines = candidate.strip().split("\n")
            if lines and lines[0]:
                tokens = lines[0].strip().split()
                return len(tokens) >= 5 and tokens[0].isdigit()
        return False

    def parse_source(self, raw_data: Union[str, Path], image_dims: Tuple[int, int] = (640, 360)) -> List[NormalizedSample]:
        content = ""
        source_name = "yolo_sample"
        if isinstance(raw_data, Path) or (isinstance(raw_data, str) and Path(raw_data).exists()):
            p = Path(raw_data)
            source_name = p.stem
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = str(raw_data)

        w, h = image_dims
        polylines: List[LanePolyline] = []

        for line in content.strip().split("\n"):
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue

            cls_idx = int(tokens[0])
            coords = [float(v) for v in tokens[1:]]

            # Convert normalized coordinates to absolute pixels
            pts: List[Point2D] = []
            for i in range(0, len(coords) - 1, 2):
                pts.append(Point2D(x=coords[i] * w, y=coords[i + 1] * h))

            assigned_class = LaneClassID.BACKGROUND
            if cls_idx == 1:
                assigned_class = LaneClassID.DRIVABLE_CORRIDOR
            elif cls_idx == 2:
                assigned_class = LaneClassID.LEFT_BOUNDARY
            elif cls_idx == 3:
                assigned_class = LaneClassID.RIGHT_BOUNDARY

            polylines.append(LanePolyline(points=pts, semantic_class=assigned_class))

        return [
            NormalizedSample(
                sample_id=source_name,
                image_path=f"{source_name}.jpg",
                image_width=w,
                image_height=h,
                polylines=polylines,
                metadata={"source_format": "yolo"},
            )
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 7. Video Stream Extraction Adapter
# ──────────────────────────────────────────────────────────────────────────────

class VideoStreamAdapter(BaseFormatAdapter):
    """
    Sequentially decodes containerized video streams (.mp4, .avi, .mkv, .mov)
    extracting keyframes at configurable temporal strides with timestamp indexing.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov"}

    def identify_format(self, source_payload: Any) -> bool:
        if isinstance(source_payload, (str, Path)):
            p = Path(source_payload)
            return p.suffix.lower() in self.SUPPORTED_EXTENSIONS
        return False

    def parse_source(
        self,
        source_payload: Any,
        frame_stride: int = 5,
        max_frames: Optional[int] = None,
    ) -> List[NormalizedSample]:
        video_path = Path(source_payload)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video container: {video_path}")

        samples: List[NormalizedSample] = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.target_width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.target_height
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        extracted_count = 0

        while True:
            ret, _ = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                pts_sec = frame_idx / fps
                sample_id = f"{video_path.stem}_frame_{frame_idx:06d}"
                sample = NormalizedSample(
                    sample_id=sample_id,
                    image_path=str(video_path),
                    image_width=width,
                    image_height=height,
                    polylines=[],
                    metadata={
                        "source_format": "video_stream",
                        "frame_index": frame_idx,
                        "timestamp_sec": round(pts_sec, 4),
                        "fps": round(fps, 2),
                        "total_container_frames": total_frames,
                    },
                )
                samples.append(sample)
                extracted_count += 1
                if max_frames and extracted_count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        logger.info(
            "Extracted %d sample frames from %s (stride=%d)",
            len(samples),
            video_path.name,
            frame_stride,
        )
        return samples

    @classmethod
    def frame_generator(
        cls,
        video_path: Union[str, Path],
        stride: int = 1,
    ) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Yields (frame_index, bgr_array, timestamp_seconds) sequentially.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % stride == 0:
                    pts = idx / fps
                    yield idx, frame, pts
                idx += 1
        finally:
            cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# 8. Raw Image & Directory Adapter
# ──────────────────────────────────────────────────────────────────────────────

class RawImageDirectoryAdapter(BaseFormatAdapter):
    """
    Ingests unannotated raw camera frames from folder hierarchies for
    self-supervised pretraining, domain adaptation, or inference staging.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def identify_format(self, source_payload: Any) -> bool:
        if isinstance(source_payload, (str, Path)):
            p = Path(source_payload)
            if p.is_dir():
                return True
            return p.suffix.lower() in self.IMAGE_EXTENSIONS
        return False

    def parse_source(self, source_payload: Any) -> List[NormalizedSample]:
        target_path = Path(source_payload)
        image_files: List[Path] = []

        if target_path.is_file():
            image_files.append(target_path)
        elif target_path.is_dir():
            for ext in self.IMAGE_EXTENSIONS:
                image_files.extend(target_path.rglob(f"*{ext}"))
                image_files.extend(target_path.rglob(f"*{ext.upper()}"))
        else:
            raise ValueError(f"Path does not exist: {target_path}")

        samples: List[NormalizedSample] = []
        for img_p in sorted(image_files):
            # Read shape header without loading entire image payload if possible
            img_hdr = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
            if img_hdr is None:
                continue
            h, w = img_hdr.shape[:2]

            sample = NormalizedSample(
                sample_id=img_p.stem,
                image_path=str(img_p),
                image_width=w,
                image_height=h,
                polylines=[],
                metadata={
                    "source_format": "raw_image",
                    "file_size_bytes": img_p.stat().st_size,
                },
            )
            samples.append(sample)

        return samples


# ──────────────────────────────────────────────────────────────────────────────
# 9. Multi-Format Geometric Integrity Validator (ISO 26262 Gate)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    """Detailed audit metrics per ingested sample."""
    sample_id: str
    is_valid: bool
    rejection_codes: List[str] = field(default_factory=list)
    polyline_count: int = 0
    left_ego_detected: bool = False
    right_ego_detected: bool = False
    mean_span_px: float = 0.0


class MultiFormatBatchValidator:
    """
    Enforces ISO 26262 ASIL-B data hygiene across any format source:
    - Coordinates must lie within physical canvas dimensions.
    - Polylines must have at least 2 distinct control vertices.
    - Y-coordinates must decrease monotonically upward (camera perspective).
    - Left ego boundary must remain strictly laterally left of right boundary.
    """

    def __init__(
        self,
        enforce_strictly_monotonic: bool = True,
        min_polyline_vertices: int = 2,
    ) -> None:
        self.enforce_monotonic = enforce_strictly_monotonic
        self.min_vertices = min_polyline_vertices

    def validate_sample(self, sample: NormalizedSample) -> ValidationReport:
        report = ValidationReport(
            sample_id=sample.sample_id,
            is_valid=True,
            polyline_count=len(sample.polylines),
        )

        w = sample.image_width
        h = sample.image_height

        if w <= 0 or h <= 0:
            report.is_valid = False
            report.rejection_codes.append("INVALID_CANVAS_DIMENSIONS")
            return report

        left_poly: Optional[LanePolyline] = None
        right_poly: Optional[LanePolyline] = None

        for poly in sample.polylines:
            if len(poly.points) < self.min_vertices:
                report.is_valid = False
                report.rejection_codes.append(f"INSUFFICIENT_VERTICES_{len(poly.points)}")

            # Check bounds
            for pt in poly.points:
                if not (-50.0 <= pt.x <= w + 50.0 and -50.0 <= pt.y <= h + 50.0):
                    report.is_valid = False
                    report.rejection_codes.append("POINT_OUT_OF_BOUNDS")
                    break

            # Monotonicity check
            if self.enforce_monotonic and len(poly.points) >= 3:
                y_diffs = [
                    poly.points[i + 1].y - poly.points[i].y
                    for i in range(len(poly.points) - 1)
                ]
                has_pos = any(d > 2.0 for d in y_diffs)
                has_neg = any(d < -2.0 for d in y_diffs)
                if has_pos and has_neg:
                    report.is_valid = False
                    report.rejection_codes.append("NON_MONOTONIC_POLYLINE")

            if poly.semantic_class == LaneClassID.LEFT_BOUNDARY:
                left_poly = poly
                report.left_ego_detected = True
            elif poly.semantic_class == LaneClassID.RIGHT_BOUNDARY:
                right_poly = poly
                report.right_ego_detected = True

        # Check cross-over anomaly: Left boundary must not cross over right boundary
        if left_poly and right_poly:
            # Sample at common y heights
            left_dict = {round(p.y): p.x for p in left_poly.points}
            right_dict = {round(p.y): p.x for p in right_poly.points}
            common_ys = set(left_dict.keys()).intersection(set(right_dict.keys()))

            cross_overs = sum(1 for y in common_ys if left_dict[y] >= right_dict[y])
            if cross_overs > 0:
                report.is_valid = False
                report.rejection_codes.append("LANE_BOUNDARIES_CROSS_OVER")

        return report

    def filter_batch(
        self,
        samples: Sequence[NormalizedSample],
    ) -> Tuple[List[NormalizedSample], List[ValidationReport]]:
        valid_samples: List[NormalizedSample] = []
        rejected_reports: List[ValidationReport] = []

        for s in samples:
            rep = self.validate_sample(s)
            if rep.is_valid:
                valid_samples.append(s)
            else:
                rejected_reports.append(rep)

        return valid_samples, rejected_reports


# ──────────────────────────────────────────────────────────────────────────────
# 10. Unified Multi-Format Factory Engine
# ──────────────────────────────────────────────────────────────────────────────

class UnifiedFormatRouter:
    """
    Intelligent format detector and dispatcher routing arbitrary inputs to
    the appropriate specialized parser adapter.
    """

    def __init__(self, target_width: int = 640, target_height: int = 360) -> None:
        self.tusimple = TuSimpleAdapter(target_width, target_height)
        self.mask_adapter = SemanticMaskAdapter(target_width, target_height)
        self.coco_adapter = COCOFormatAdapter(target_width, target_height)
        self.yolo_adapter = YOLOFormatAdapter(target_width, target_height)
        self.video_adapter = VideoStreamAdapter(target_width, target_height)
        self.raw_image_adapter = RawImageDirectoryAdapter(target_width, target_height)
        self.validator = MultiFormatBatchValidator()

        self.adapters: List[BaseFormatAdapter] = [
            self.tusimple,
            self.mask_adapter,
            self.coco_adapter,
            self.yolo_adapter,
            self.video_adapter,
            self.raw_image_adapter,
        ]

    def route_and_parse(self, source_payload: Any) -> List[NormalizedSample]:
        """
        Auto-detect format and execute parsing.
        """
        for adapter in self.adapters:
            if adapter.identify_format(source_payload):
                logger.debug(f"Routing payload to {adapter.__class__.__name__}")
                return adapter.parse_source(source_payload)

        # Fallback to TuSimple if ambiguous JSON string
        try:
            samples = self.tusimple.parse_source(source_payload)
            if samples:
                return samples
        except Exception:
            pass

        raise ValueError("Unable to automatically detect format adapter for unrecognized payload")

    def parse_and_validate(
        self,
        source_payload: Any,
    ) -> Tuple[List[NormalizedSample], List[ValidationReport]]:
        """
        Parse and run strict geometric plausibility checks.
        """
        samples = self.route_and_parse(source_payload)
        return self.validator.filter_batch(samples)


# ──────────────────────────────────────────────────────────────────────────────
# 11. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing FormatAdapters ISO 26262 ASIL-B self-check...")
    router = UnifiedFormatRouter()

    # Synthetic TuSimple Record
    synthetic_tusimple = {
        "lanes": [
            [-2, -2, 210, 230, 250, 270, 290],
            [-2, -2, 430, 410, 390, 370, 350],
        ],
        "h_samples": [180, 200, 220, 240, 260, 280, 300],
        "raw_file": "clips/0530/test_clip_01/20.jpg",
    }

    samples = router.route_and_parse(synthetic_tusimple)
    assert len(samples) == 1, "Failed to parse TuSimple record."
    s = samples[0]
    assert s.sample_id == "20"
    assert len(s.polylines) == 2

    # Check class assignment: Left should be Class 2, Right should be Class 3
    classes = [p.semantic_class for p in s.polylines]
    assert LaneClassID.LEFT_BOUNDARY in classes
    assert LaneClassID.RIGHT_BOUNDARY in classes

    # Test Validation Engine
    validator = MultiFormatBatchValidator()
    report = validator.validate_sample(s)
    assert report.is_valid, f"Validation failed unexpectedly: {report.rejection_codes}"

    # Test synthetic cross-over failure detection
    inverted_sample = NormalizedSample(
        sample_id="bad_cross",
        image_path="test.jpg",
        image_width=1280,
        image_height=720,
        polylines=[
            LanePolyline(
                points=[Point2D(600, 300), Point2D(700, 600)],
                semantic_class=LaneClassID.LEFT_BOUNDARY,
            ),
            LanePolyline(
                points=[Point2D(400, 300), Point2D(500, 600)],
                semantic_class=LaneClassID.RIGHT_BOUNDARY,
            ),
        ],
    )
    bad_report = validator.validate_sample(inverted_sample)
    assert not bad_report.is_valid, "Cross-over validation failed to flag inverted boundaries."
    assert "LANE_BOUNDARIES_CROSS_OVER" in bad_report.rejection_codes

    print(f"TuSimple adapter and Validator verified successfully with {len(s.polylines)} lane polylines.")

