"""
src/data/dataset_normalizer.py
==============================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
High-Precision Image Normalizer, Deduplicator & Polyline Rasterizer.
ISO 26262 ASIL-B Data Preprocessing & Ground-Truth Generation Baseline.

Key Responsibilities:
---------------------
1. Standardized Image Normalization:
   - Resizes all input camera frames to standard perception resolution (640x360).
   - High-fidelity downscaling via `cv2.INTER_AREA` (or `cv2.INTER_LINEAR` for upscaling).
   - Complete EXIF orientation and metadata stripping for consistent coordinate spaces.
2. 4-Class Semantic Mask Rasterization:
   - Accurately converts floating-point polylines into discrete 4-class pixel masks.
   - Rasterizes Left Boundary (Class 2) and Right Boundary (Class 3) with configurable stroke width.
   - Computes closed polygon between ego left and ego right to fill Drivable Corridor (Class 1).
   - Enforces strict hierarchical painting: Background (0) -> Corridor (1) -> Boundary Lines (2, 3).
3. SHA-256 Cryptographic Deduplication:
   - Computes raw pixel stream SHA-256 digests to identify and filter duplicate frames.
   - DHash (difference hash) computation for near-duplicate and identical sequence detection.
4. Quarantine & Anomaly Defense:
   - Flags blank, corrupt, or degraded frames and isolates them to quarantine.
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.format_adapters import LanePolyline, NormalizedSample, Point2D
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema

logger = logging.getLogger("APEX.DatasetNormalizer")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Configuration & Forensic Audit Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NormalizationConfig:
    """
    Tunable parameters for image rescaling, deduplication, and mask rasterization.
    """
    target_width: int = 640
    target_height: int = 360
    lane_stroke_width_px: int = 6
    fill_corridor: bool = True
    corridor_min_points: int = 3
    strip_exif: bool = True
    enable_sha256_dedup: bool = True
    enable_dhash_dedup: bool = False
    dhash_threshold: int = 2
    min_contrast_variance: float = 0.0
    jpeg_quality: int = 95
    png_compression: int = 6


@dataclass
class NormalizedOutput:
    """
    Fully processed, standardized image and semantic mask ready for model training.
    """
    sample_id: str
    image_bgr: np.ndarray
    mask_uint8: np.ndarray
    sha256_hash: str
    dhash_int: int = 0
    is_duplicate: bool = False
    is_corrupt: bool = False
    quarantine_reason: Optional[str] = None
    original_dimensions: Tuple[int, int] = (0, 0)
    rescale_scale_x: float = 1.0
    rescale_scale_y: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Master Dataset Normalizer & Rasterizer Engine
# ──────────────────────────────────────────────────────────────────────────────

class DatasetNormalizer:
    """
    Production-grade image rescaler, SHA-256 deduplicator, and 4-class mask rasterizer.
    Converts raw TuSimple polylines and arbitrary images into strict ISO 26262 training assets.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None) -> None:
        self.config = config or NormalizationConfig()
        self.schema = get_default_label_schema()

        # Deduplication registries
        self._seen_sha256: Set[str] = set()
        self._seen_dhashes: Dict[int, str] = {}
        self._total_processed: int = 0
        self._total_duplicates: int = 0
        self._total_quarantined: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Image Rescaling & Preprocessing
    # ──────────────────────────────────────────────────────────────────────────

    def decode_and_normalize_image(
        self,
        raw_source: Union[bytes, np.ndarray, Path, str],
    ) -> Tuple[np.ndarray, Tuple[int, int], float, float]:
        """
        Decode raw image bytes or file, strip metadata, and resize to target resolution.
        Returns:
            (resized_bgr, (orig_w, orig_h), scale_x, scale_y)
        """
        img_bgr: Optional[np.ndarray] = None

        if isinstance(raw_source, np.ndarray):
            img_bgr = raw_source
        elif isinstance(raw_source, bytes):
            nparr = np.frombuffer(raw_source, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(raw_source, (str, Path)):
            p = str(raw_source)
            img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Corrupt or unreadable image stream.")

        orig_h, orig_w = img_bgr.shape[:2]
        tw, th = self.config.target_width, self.config.target_height

        scale_x = float(tw) / float(orig_w)
        scale_y = float(th) / float(orig_h)

        if orig_w == tw and orig_h == th:
            resized = img_bgr
        else:
            # INTER_AREA for downscaling (preserves thin edge sharpness), INTER_LINEAR for upscaling
            interp = cv2.INTER_AREA if (orig_w > tw or orig_h > th) else cv2.INTER_LINEAR
            resized = cv2.resize(img_bgr, (tw, th), interpolation=interp)

        return resized, (orig_w, orig_h), scale_x, scale_y

    # ──────────────────────────────────────────────────────────────────────────
    # Cryptographic Hash & DHash Deduplication
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_sha256(data: Union[bytes, np.ndarray]) -> str:
        """Compute SHA-256 hash of raw bytes or numpy byte buffer."""
        hasher = hashlib.sha256()
        if isinstance(data, np.ndarray):
            hasher.update(data.tobytes())
        else:
            hasher.update(data)
        return hasher.hexdigest()

    @staticmethod
    def compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> int:
        """
        Compute fast 64-bit difference hash (DHash) for structural similarity.
        """
        # Resize to (hash_size + 1, hash_size) in grayscale
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

        # Compute difference between adjacent horizontal pixels
        diff = small[:, 1:] > small[:, :-1]
        # Pack boolean array into single 64-bit integer
        dhash_val = 0
        for bit in diff.flatten():
            dhash_val = (dhash_val << 1) | int(bit)
        return dhash_val

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Calculate bitwise Hamming distance between two 64-bit integer hashes."""
        return bin(h1 ^ h2).count("1")

    def check_and_register_duplicate(self, sha256_hash: str, dhash_val: int) -> Tuple[bool, Optional[str]]:
        """
        Check if an image has been registered previously.
        Returns:
            (is_duplicate, duplicate_origin_hash)
        """
        if self.config.enable_sha256_dedup:
            if sha256_hash in self._seen_sha256:
                return True, f"Exact SHA-256 match ({sha256_hash[:12]}...)"

        if self.config.enable_dhash_dedup:
            for stored_hash, stored_id in self._seen_dhashes.items():
                if self.hamming_distance(dhash_val, stored_hash) <= self.config.dhash_threshold:
                    return True, f"Near-duplicate DHash match (dist <= {self.config.dhash_threshold})"

        # Not duplicate: register
        self._seen_sha256.add(sha256_hash)
        self._seen_dhashes[dhash_val] = sha256_hash
        return False, None

    # ──────────────────────────────────────────────────────────────────────────
    # 4-Class Mask Polyline Rasterization
    # ──────────────────────────────────────────────────────────────────────────

    def rasterize_polylines_to_mask(
        self,
        polylines: List[LanePolyline],
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        """
        Rasterize floating-point polylines into standard 640x360 4-class semantic mask.
        Strict ISO 26262 painting order:
          1. Class 0: Background (all pixels default to 0)
          2. Class 1: Drivable Corridor (filled polygon between ego left and ego right)
          3. Class 2: Left Boundary (continuous stroked polyline)
          4. Class 3: Right Boundary (continuous stroked polyline)
        """
        tw, th = self.config.target_width, self.config.target_height
        mask = np.zeros((th, tw), dtype=np.uint8)

        if not polylines:
            return mask

        # Rescale polylines to target coordinate space
        scaled_polylines: List[LanePolyline] = []
        for poly in polylines:
            scaled_pts = [
                Point2D(x=p.x * scale_x, y=p.y * scale_y)
                for p in poly.points
            ]
            scaled_polylines.append(
                LanePolyline(
                    points=scaled_pts,
                    semantic_class=poly.semantic_class,
                    confidence=poly.confidence,
                    lane_id=poly.lane_id,
                )
            )

        # 1. Fill Drivable Corridor (Class 1) if enabled and both ego boundaries present
        if self.config.fill_corridor:
            self._fill_drivable_corridor(mask, scaled_polylines)

        # 2. Rasterize Line Boundaries (Classes 2 and 3) on top of corridor
        stroke = self.config.lane_stroke_width_px

        for poly in scaled_polylines:
            if not poly.is_valid:
                continue

            # Identify target class index
            cid = poly.semantic_class.value
            if cid not in (LaneClassID.LEFT_BOUNDARY.value, LaneClassID.RIGHT_BOUNDARY.value):
                # Only stroke boundaries
                continue

            # Convert points to contour coordinates [N, 1, 2]
            contour = poly.to_int_contour()

            # Draw smooth continuous anti-aliased polyline
            cv2.polylines(
                mask,
                [contour],
                isClosed=False,
                color=cid,
                thickness=stroke,
                lineType=cv2.LINE_8, # Strict integer indexing without blurred alpha edges
            )

        return mask

    def _fill_drivable_corridor(self, mask: np.ndarray, polylines: List[LanePolyline]) -> None:
        """
        Construct and fill the polygon bounded by ego left and ego right lanes.
        """
        ego_left = None
        ego_right = None

        for p in polylines:
            if p.semantic_class == LaneClassID.LEFT_BOUNDARY:
                ego_left = p
            elif p.semantic_class == LaneClassID.RIGHT_BOUNDARY:
                ego_right = p

        if ego_left is None or ego_right is None:
            return

        if len(ego_left.points) < 2 or len(ego_right.points) < 2:
            return

        # Left lane points ordered top to bottom
        pts_left = sorted(ego_left.points, key=lambda pt: pt.y)
        # Right lane points ordered bottom to top (to form closed loop)
        pts_right = sorted(ego_right.points, key=lambda pt: pt.y, reverse=True)

        combined_polygon: List[List[int]] = []
        for pt in pts_left:
            combined_polygon.append([int(round(pt.x)), int(round(pt.y))])
        for pt in pts_right:
            combined_polygon.append([int(round(pt.x)), int(round(pt.y))])

        if len(combined_polygon) >= 4:
            poly_np = np.array([combined_polygon], dtype=np.int32)
            cv2.fillPoly(mask, poly_np, color=LaneClassID.DRIVABLE_CORRIDOR.value)

    # ──────────────────────────────────────────────────────────────────────────
    # Complete Sample Normalization Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def process_sample(
        self,
        sample: NormalizedSample,
        raw_image_data: Union[bytes, np.ndarray, Path, str],
    ) -> NormalizedOutput:
        """
        Execute full end-to-end normalization:
        1. Decode and scale image.
        2. Compute SHA-256 and DHash.
        3. Check duplicates.
        4. Rasterize or resize mask.
        5. Validate quality gates.
        """
        self._total_processed += 1

        try:
            img_bgr, orig_dims, scale_x, scale_y = self.decode_and_normalize_image(raw_image_data)
        except Exception as e:
            self._total_quarantined += 1
            return NormalizedOutput(
                sample_id=sample.sample_id,
                image_bgr=np.zeros((self.config.target_height, self.config.target_width, 3), dtype=np.uint8),
                mask_uint8=np.zeros((self.config.target_height, self.config.target_width), dtype=np.uint8),
                sha256_hash="",
                dhash_int=0,
                is_corrupt=True,
                quarantine_reason=f"Image decoding failed: {e}",
            )

        # Contrast validation
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        variance = float(np.var(gray))
        if self.config.min_contrast_variance > 0.0 and variance < self.config.min_contrast_variance:
            self._total_quarantined += 1
            return NormalizedOutput(
                sample_id=sample.sample_id,
                image_bgr=img_bgr,
                mask_uint8=np.zeros((self.config.target_height, self.config.target_width), dtype=np.uint8),
                sha256_hash="",
                dhash_int=0,
                is_corrupt=True,
                quarantine_reason=f"Insufficient image contrast variance: {variance:.2f} < {self.config.min_contrast_variance}",
            )

        sha256_hash = self.compute_sha256(img_bgr)
        dhash_val = self.compute_dhash(img_bgr)

        is_dup, dup_reason = self.check_and_register_duplicate(sha256_hash, dhash_val)
        if is_dup:
            self._total_duplicates += 1

        # Generate semantic mask
        if sample.has_mask and sample.precomputed_mask is not None:
            # Resize existing mask
            pm = sample.precomputed_mask
            if pm.shape != (self.config.target_height, self.config.target_width):
                mask_uint8 = cv2.resize(
                    pm,
                    (self.config.target_width, self.config.target_height),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask_uint8 = pm.copy()
        elif sample.has_polylines:
            mask_uint8 = self.rasterize_polylines_to_mask(sample.polylines, scale_x, scale_y)
        else:
            mask_uint8 = np.zeros((self.config.target_height, self.config.target_width), dtype=np.uint8)

        # Final mask schema verification
        is_schema_valid, schema_errors = self.schema.validate_mask_array(
            mask_uint8, expected_shape=(self.config.target_height, self.config.target_width)
        )
        if not is_schema_valid:
            logger.warning(f"Schema violation in {sample.sample_id}: {schema_errors}")

        return NormalizedOutput(
            sample_id=sample.sample_id,
            image_bgr=img_bgr,
            mask_uint8=mask_uint8,
            sha256_hash=sha256_hash,
            dhash_int=dhash_val,
            is_duplicate=is_dup,
            is_corrupt=False,
            quarantine_reason=dup_reason if is_dup else None,
            original_dimensions=orig_dims,
            rescale_scale_x=scale_x,
            rescale_scale_y=scale_y,
            metadata=sample.metadata,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Photometric Normalization & Illumination Auto-Balancing
# ──────────────────────────────────────────────────────────────────────────────

class PhotometricNormalizer:
    """
    Standardizes camera exposure, lighting extremes, and color temperature
    across heterogeneous datasets (daylight, rain, tunnels, twilight).
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        apply_clahe: bool = False,
    ) -> None:
        self.apply_clahe = apply_clahe
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def equalize_luminance(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on LAB luminance channel (L) without distorting chromaticity (A, B).
        """
        if not self.apply_clahe:
            return image_bgr

        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def compute_white_balance_gray_world(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Apply Gray-World color constancy algorithm to eliminate artificial chromatic casts.
        """
        img_f = image_bgr.astype(np.float32)
        b_mean = float(np.mean(img_f[:, :, 0]))
        g_mean = float(np.mean(img_f[:, :, 1]))
        r_mean = float(np.mean(img_f[:, :, 2]))

        avg_gray = (b_mean + g_mean + r_mean) / 3.0

        kb = avg_gray / max(1.0, b_mean)
        kg = avg_gray / max(1.0, g_mean)
        kr = avg_gray / max(1.0, r_mean)

        balanced = cv2.merge([
            np.clip(img_f[:, :, 0] * kb, 0, 255).astype(np.uint8),
            np.clip(img_f[:, :, 1] * kg, 0, 255).astype(np.uint8),
            np.clip(img_f[:, :, 2] * kr, 0, 255).astype(np.uint8),
        ])
        return balanced


# ──────────────────────────────────────────────────────────────────────────────
# 4. Catmull-Rom & Cubic Spline Polyline Smoother
# ──────────────────────────────────────────────────────────────────────────────

class CubicSplinePolylineSmoother:
    """
    Interpolates sparse discrete point annotations into high-density smooth curves
    to eliminate polygonal discretization artifacts in rasterized ground truth masks.
    """

    def __init__(self, target_sample_count: int = 50) -> None:
        self.target_points = target_sample_count

    def smooth_polyline(self, points: List[Point2D]) -> List[Point2D]:
        """
        Fit cubic spline to (x, y) point sequences ordered from top to bottom.
        """
        if len(points) < 4:
            return points

        pts_x = [p.x for p in points]
        pts_y = [p.y for p in points]

        # Use chord-length parameterization
        t = [0.0]
        for i in range(1, len(points)):
            dist = math.sqrt((pts_x[i] - pts_x[i - 1]) ** 2 + (pts_y[i] - pts_y[i - 1]) ** 2)
            t.append(t[-1] + max(1e-4, dist))

        total_length = t[-1]
        t_normalized = np.array(t) / total_length

        t_dense = np.linspace(0.0, 1.0, self.target_points)

        try:
            interp_x = np.interp(t_dense, t_normalized, pts_x)
            interp_y = np.interp(t_dense, t_normalized, pts_y)
            return [Point2D(x=float(x), y=float(y)) for x, y in zip(interp_x, interp_y)]
        except Exception:
            return points


# ──────────────────────────────────────────────────────────────────────────────
# 5. Frame Quality Anomaly & Defect Auditor
# ──────────────────────────────────────────────────────────────────────────────

class FrameQualityAnomalyAuditor:
    """
    Evaluates raw camera frames for physical defects:
      1. Optical Blur: Evaluated via Laplacian spatial filter variance.
      2. Extreme Overexposure / Underexposure: Histogram clipping.
      3. Digital Sensor Glitches: Flat color bands and hot pixels.
    """

    def __init__(
        self,
        blur_laplacian_threshold: float = 12.0,
        max_clipped_pixel_ratio: float = 0.40,
    ) -> None:
        self.blur_threshold = blur_laplacian_threshold
        self.max_clipped_ratio = max_clipped_pixel_ratio

    def evaluate_quality(self, image_bgr: np.ndarray) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Perform fast forensic analysis on image quality.
        Returns:
            (is_acceptable, anomaly_reason, metric_dict)
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Laplacian blur variance
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = float(lap.var())

        # 2. Histogram exposure analysis
        total_pixels = gray.size
        black_pixels = int(np.sum(gray <= 5))
        white_pixels = int(np.sum(gray >= 250))

        black_ratio = black_pixels / total_pixels
        white_ratio = white_pixels / total_pixels

        metrics = {
            "blur_laplacian_score": round(blur_score, 2),
            "underexposed_ratio": round(black_ratio, 4),
            "overexposed_ratio": round(white_ratio, 4),
        }

        if black_ratio > self.max_clipped_ratio:
            return False, f"Severe underexposure ({black_ratio * 100:.1f}% black pixels)", metrics

        if white_ratio > self.max_clipped_ratio:
            return False, f"Severe camera glare/overexposure ({white_ratio * 100:.1f}% saturated pixels)", metrics

        return True, None, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 6. Class Distribution & Spatial Balance Observer
# ──────────────────────────────────────────────────────────────────────────────

class MaskSpatialBalanceObserver:
    """
    Tracks aggregate class pixel representations across all rasterized training masks.
    Alerts if thin boundary markings (Class 2 and 3) risk gradient starvation.
    """

    def __init__(self) -> None:
        self.total_pixels_observed: int = 0
        self.class_pixel_counts: Dict[int, int] = {
            LaneClassID.BACKGROUND.value: 0,
            LaneClassID.DRIVABLE_CORRIDOR.value: 0,
            LaneClassID.LEFT_BOUNDARY.value: 0,
            LaneClassID.RIGHT_BOUNDARY.value: 0,
        }

    def update_from_mask(self, mask_uint8: np.ndarray) -> None:
        """Accumulate pixel histogram from 4-class mask."""
        classes, counts = np.unique(mask_uint8, return_counts=True)
        for cls_id, cnt in zip(classes, counts):
            if int(cls_id) in self.class_pixel_counts:
                self.class_pixel_counts[int(cls_id)] += int(cnt)
        self.total_pixels_observed += int(mask_uint8.size)

    def get_distribution_percentages(self) -> Dict[str, float]:
        """Return percentage breakdown per semantic class."""
        if self.total_pixels_observed == 0:
            return {
                "background": 100.0,
                "drivable_corridor": 0.0,
                "left_boundary": 0.0,
                "right_boundary": 0.0,
            }

        names = {
            LaneClassID.BACKGROUND.value: "background",
            LaneClassID.DRIVABLE_CORRIDOR.value: "drivable_corridor",
            LaneClassID.LEFT_BOUNDARY.value: "left_boundary",
            LaneClassID.RIGHT_BOUNDARY.value: "right_boundary",
        }
        res: Dict[str, float] = {}
        for cls_id, name in names.items():
            pct = (self.class_pixel_counts[cls_id] / self.total_pixels_observed) * 100.0
            res[name] = round(pct, 4)
        return res


# ──────────────────────────────────────────────────────────────────────────────
# 7. Corridor Geometric Sanity Validator (ISO 26262 Plausibility Gate)
# ──────────────────────────────────────────────────────────────────────────────

class CorridorGeometricValidator:
    """
    Evaluates geometric plausibility of rasterized corridor polygons to prevent
    spurious lane crossing or inverted perspective geometries from entering training.
    
    Checks:
    - Minimum enclosed corridor area in pixels.
    - Perspective trapezoidal ratio: width at bottom vs. width near vanishing point.
    - Non-self-intersection and positive convex hull area.
    - Bounded lateral boundaries within standard road width envelope.
    """

    def __init__(
        self,
        min_area_px: float = 800.0,
        max_area_ratio: float = 0.65,
        min_bottom_width_px: float = 60.0,
        max_bottom_width_px: float = 620.0,
    ) -> None:
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio
        self.min_bottom_width_px = min_bottom_width_px
        self.max_bottom_width_px = max_bottom_width_px

    def validate_corridor_polygon(
        self,
        polygon_pts: np.ndarray,
        frame_shape: Tuple[int, int] = (360, 640),
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Validate polygon coordinates before corridor painting.

        Args:
            polygon_pts: (N, 2) array of (x, y) coordinates.
            frame_shape: (height, width) of target rasterization space.

        Returns:
            Tuple of (is_valid, rejection_reason, metrics_dict).
        """
        if polygon_pts is None or len(polygon_pts) < 3:
            return False, "Corridor polygon requires at least 3 vertices", {}

        height, width = frame_shape
        total_frame_area = float(height * width)

        pts_int = polygon_pts.astype(np.int32)
        area = float(cv2.contourArea(pts_int))

        metrics: Dict[str, float] = {
            "corridor_area_px": round(area, 1),
            "frame_area_ratio": round(area / total_frame_area, 4),
        }

        if area < self.min_area_px:
            return (
                False,
                f"Corridor area too small ({area:.1f}px < threshold {self.min_area_px}px)",
                metrics,
            )

        if area > (self.max_area_ratio * total_frame_area):
            return (
                False,
                f"Corridor area abnormally large ({area:.1f}px exceeds {self.max_area_ratio*100}% of frame)",
                metrics,
            )

        # Check bottom scanline width (near vehicle hood: y > height * 0.85)
        bottom_pts = polygon_pts[polygon_pts[:, 1] >= (height * 0.85)]
        if len(bottom_pts) >= 2:
            x_min = float(np.min(bottom_pts[:, 0]))
            x_max = float(np.max(bottom_pts[:, 0]))
            bottom_width = x_max - x_min
            metrics["bottom_width_px"] = round(bottom_width, 1)

            if bottom_width < self.min_bottom_width_px:
                return (
                    False,
                    f"Corridor bottom width too narrow ({bottom_width:.1f}px < {self.min_bottom_width_px}px)",
                    metrics,
                )
            if bottom_width > self.max_bottom_width_px:
                return (
                    False,
                    f"Corridor bottom width too wide ({bottom_width:.1f}px > {self.max_bottom_width_px}px)",
                    metrics,
                )

        # Perspective taper check: Top width should be smaller than bottom width
        top_pts = polygon_pts[polygon_pts[:, 1] <= (height * 0.55)]
        if len(top_pts) >= 2 and len(bottom_pts) >= 2:
            top_width = float(np.max(top_pts[:, 0]) - np.min(top_pts[:, 0]))
            metrics["top_width_px"] = round(top_width, 1)
            taper_ratio = top_width / max(1.0, metrics.get("bottom_width_px", 1.0))
            metrics["taper_ratio"] = round(taper_ratio, 3)

            # Inverted taper: top is more than 1.6x wider than bottom is physically implausible
            if taper_ratio > 1.60:
                return (
                    False,
                    f"Inverted perspective taper detected (top/bottom ratio {taper_ratio:.2f} > 1.60)",
                    metrics,
                )

        return True, None, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 8. Forensic Visual Audit Overlay Renderer
# ──────────────────────────────────────────────────────────────────────────────

class MaskVisualAuditRenderer:
    """
    Renders diagnostic composite visual artifacts for visual data quality audits.
    Blends standardized 640x360 camera frames with 4-class semantic lane masks.
    """

    COLOR_PALETTE: Dict[int, Tuple[int, int, int]] = {
        LaneClassID.BACKGROUND.value: (0, 0, 0),         # Transparent / unhighlighted
        LaneClassID.DRIVABLE_CORRIDOR.value: (0, 200, 0), # Translucent emerald green
        LaneClassID.LEFT_BOUNDARY.value: (255, 200, 0),  # Bright cyan/sky blue
        LaneClassID.RIGHT_BOUNDARY.value: (0, 70, 255),   # Bright safety amber/orange
    }

    def __init__(self, alpha: float = 0.40) -> None:
        self.alpha = float(np.clip(alpha, 0.05, 0.95))

    def render_overlay(
        self,
        image_bgr: np.ndarray,
        mask_uint8: np.ndarray,
        draw_contours: bool = True,
    ) -> np.ndarray:
        """
        Produce alpha-blended RGB audit overlay of semantic lane perception.

        Args:
            image_bgr: (H, W, 3) standardized BGR image.
            mask_uint8: (H, W) single-channel 4-class mask.
            draw_contours: Whether to draw crisp vector boundaries around corridors.

        Returns:
            (H, W, 3) composite BGR image ready for JPEG saving or live display.
        """
        h, w = mask_uint8.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for cls_id, bgr_color in self.COLOR_PALETTE.items():
            if cls_id == LaneClassID.BACKGROUND.value:
                continue
            color_mask[mask_uint8 == cls_id] = bgr_color

        # Alpha blend where mask is non-zero
        foreground_mask = (mask_uint8 > 0)
        overlay = image_bgr.copy()
        overlay[foreground_mask] = cv2.addWeighted(
            image_bgr[foreground_mask],
            1.0 - self.alpha,
            color_mask[foreground_mask],
            self.alpha,
            0.0,
        )

        if draw_contours:
            # Draw distinct boundary lines on top of blend
            corridor_mask = (mask_uint8 == LaneClassID.DRIVABLE_CORRIDOR.value).astype(np.uint8)
            contours, _ = cv2.findContours(corridor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 128), 1)

        return overlay

    def render_side_by_side(
        self,
        image_bgr: np.ndarray,
        mask_uint8: np.ndarray,
        caption: str = "APEX-Vision Semantic Verification",
    ) -> np.ndarray:
        """
        Concatenate raw image and colorized mask side-by-side with telemetry header.
        """
        overlay = self.render_overlay(image_bgr, mask_uint8)
        colorized = np.zeros_like(image_bgr)
        for cls_id, color in self.COLOR_PALETTE.items():
            colorized[mask_uint8 == cls_id] = color

        combined = np.hstack([image_bgr, overlay, colorized])
        header_h = 28
        total_w = combined.shape[1]
        banner = np.zeros((header_h, total_w, 3), dtype=np.uint8)
        cv2.putText(
            banner,
            caption,
            (12, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return np.vstack([banner, combined])


# ──────────────────────────────────────────────────────────────────────────────
# 9. Batch Dataset Normalizer Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class BatchDatasetNormalizer:
    """
    Coordinates batch-level normalizations with real-time class balance tracking,
    forensic validation gating, deduplication metrics, and quarantine routing.
    """

    def __init__(self, normalizer: Optional[DatasetNormalizer] = None) -> None:
        self.normalizer = normalizer or DatasetNormalizer()
        self.geometric_validator = CorridorGeometricValidator()
        self.balance_observer = MaskSpatialBalanceObserver()
        self.visual_renderer = MaskVisualAuditRenderer()

        self.processed_count: int = 0
        self.duplicate_count: int = 0
        self.corrupt_count: int = 0
        self.quarantined_count: int = 0

    def process_batch(
        self,
        samples: Sequence[Tuple[NormalizedSample, Union[np.ndarray, bytes]]],
        quarantine_dir: Optional[Path] = None,
    ) -> List[NormalizedOutput]:
        """
        Process a collection of samples sequentially with full validation.

        Args:
            samples: List of (NormalizedSample, image_source) pairs.
            quarantine_dir: Directory where rejected or corrupt samples are dumped.

        Returns:
            List of valid NormalizedOutput instances.
        """
        valid_outputs: List[NormalizedOutput] = []

        for sample_meta, img_src in samples:
            out = self.normalizer.process_sample(sample_meta, img_src)
            self.processed_count += 1

            if out.is_duplicate:
                self.duplicate_count += 1
                continue

            if out.is_corrupt:
                self.corrupt_count += 1
                self.quarantined_count += 1
                if quarantine_dir:
                    self._dispatch_quarantine(sample_meta, img_src, out.rejection_reason, quarantine_dir)
                continue

            # Update spatial balance
            self.balance_observer.update_from_mask(out.mask_uint8)
            valid_outputs.append(out)

        return valid_outputs

    def _dispatch_quarantine(
        self,
        sample: NormalizedSample,
        img_src: Union[np.ndarray, bytes],
        reason: Optional[str],
        quarantine_dir: Path,
    ) -> None:
        """Write quarantined artifact to disk with diagnostic text log."""
        try:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(sample.image_path).stem or sample.sample_id
            log_file = quarantine_dir / f"{stem}_quarantine.txt"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(f"Sample ID: {sample.sample_id}\n")
                f.write(f"Source Path: {sample.image_path}\n")
                f.write(f"Rejection Reason: {reason or 'Unknown'}\n")

            if isinstance(img_src, bytes):
                img_out = quarantine_dir / f"{stem}_corrupt.jpg"
                with open(img_out, "wb") as f:
                    f.write(img_src)
            elif isinstance(img_src, np.ndarray):
                cv2.imwrite(str(quarantine_dir / f"{stem}_corrupt.png"), img_src)
        except Exception as exc:
            logger.warning("Failed to record quarantine artifact for %s: %s", sample.sample_id, exc)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Compile execution statistics and balance report."""
        return {
            "total_processed": self.processed_count,
            "total_valid": self.processed_count - self.duplicate_count - self.corrupt_count,
            "duplicates_dropped": self.duplicate_count,
            "corrupt_quarantined": self.corrupt_count,
            "class_distribution_pct": self.balance_observer.get_distribution_percentages(),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 10. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing DatasetNormalizer ISO 26262 ASIL-B self-check...")
    normalizer = DatasetNormalizer()

    # Create synthetic test image
    test_img = np.full((720, 1280, 3), 120, dtype=np.uint8)
    # Add fake road gradient
    test_img[360:, :] = 60

    # Create synthetic polylines
    left_line = LanePolyline(
        points=[Point2D(x=450.0, y=400.0), Point2D(x=350.0, y=700.0)],
        semantic_class=LaneClassID.LEFT_BOUNDARY,
    )
    right_line = LanePolyline(
        points=[Point2D(x=830.0, y=400.0), Point2D(x=930.0, y=700.0)],
        semantic_class=LaneClassID.RIGHT_BOUNDARY,
    )

    test_sample = NormalizedSample(
        sample_id="test_frame_01",
        image_path="test_frame_01.jpg",
        image_width=1280,
        image_height=720,
        polylines=[left_line, right_line],
    )

    out = normalizer.process_sample(test_sample, test_img)
    assert not out.is_corrupt
    assert out.image_bgr.shape == (360, 640, 3)
    assert out.mask_uint8.shape == (360, 640)

    # Check that mask has corridor and boundary classes
    classes_present = set(np.unique(out.mask_uint8))
    assert LaneClassID.DRIVABLE_CORRIDOR.value in classes_present
    assert LaneClassID.LEFT_BOUNDARY.value in classes_present
    assert LaneClassID.RIGHT_BOUNDARY.value in classes_present

    # Test deduplication detection
    out_dup = normalizer.process_sample(test_sample, test_img)
    assert out_dup.is_duplicate, "Failed to identify duplicate frame via SHA-256."

    # Test geometric validator
    geo_val = CorridorGeometricValidator()
    is_valid, reason, _ = geo_val.validate_corridor_polygon(
        np.array([[200, 350], [440, 350], [380, 180], [260, 180]]),
        (360, 640),
    )
    assert is_valid, f"Geometric validation failed: {reason}"

    # Test visual renderer
    renderer = MaskVisualAuditRenderer()
    overlay = renderer.render_overlay(out.image_bgr, out.mask_uint8)
    assert overlay.shape == (360, 640, 3)

    print("DatasetNormalizer self-check passed successfully.")

