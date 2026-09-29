"""
src/dataset/thumbnail_generator.py
==================================
Fast thumbnail generation and caching service for UI previews.

Provides:
  - Rapid downsampling of image frames and video keyframes to 200×120px
  - Disk-backed caching in data/.thumbnails/
  - Base64 encoding for lag-free client-side HTML/CSS rendering
  - Safe error recovery on corrupt or partial frames
"""

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from src.config import PROJECT_ROOT, cfg


class ThumbnailGenerator:
    """
    Generates and caches fast, lightweight UI thumbnail previews.
    """

    def __init__(
        self,
        thumbnail_dir: Optional[Path] = None,
        default_size: Tuple[int, int] = (200, 120),
        jpeg_quality: int = 80,
    ):
        """
        Initialize the thumbnail generator.

        Args:
            thumbnail_dir: Destination folder for cached thumbnail images.
            default_size: (width, height) of generated thumbnails.
            jpeg_quality: JPEG compression quality (1-100).
        """
        upload_cfg = cfg.get("upload", {})
        dir_name = upload_cfg.get("thumbnail_dir", "data/.thumbnails")
        self.thumbnail_dir = Path(thumbnail_dir) if thumbnail_dir else PROJECT_ROOT / dir_name
        self.thumbnail_dir.mkdir(parents=True, exist_ok=True)
        self.default_size = tuple(upload_cfg.get("thumbnail_size", default_size))
        self.jpeg_quality = int(upload_cfg.get("thumbnail_quality", jpeg_quality))

    def _get_cache_path(self, identifier: str) -> Path:
        """Derive a safe cache file path based on a hash of the identifier."""
        safe_hash = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16]
        return self.thumbnail_dir / f"thumb_{safe_hash}.jpg"

    def generate_from_image(
        self,
        image_source: Union[str, Path, np.ndarray, Image.Image],
        cache_id: Optional[str] = None,
        size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Path]:
        """
        Create a 200×120px thumbnail from an image file, numpy array, or PIL Image.

        Args:
            image_source: Image file path, numpy BGR/RGB array, or PIL Image.
            cache_id: Unique string key for disk caching.
            size: Override thumbnail dimensions (width, height).

        Returns:
            Path to saved thumbnail on disk, or None if generation failed.
        """
        target_size = size or self.default_size

        if isinstance(image_source, (str, Path)):
            src_path = Path(image_source)
            if not src_path.exists():
                return None
            key = cache_id or f"{src_path.name}_{src_path.stat().st_size}_{src_path.stat().st_mtime}"
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                return cache_path

            try:
                img = cv2.imread(str(src_path))
                if img is None:
                    return None
                thumb = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(cache_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                return cache_path
            except Exception:
                return None

        elif isinstance(image_source, np.ndarray):
            key = cache_id or f"array_{id(image_source)}"
            cache_path = self._get_cache_path(key)
            try:
                thumb = cv2.resize(image_source, target_size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(cache_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                return cache_path
            except Exception:
                return None

        elif isinstance(image_source, Image.Image):
            key = cache_id or f"pil_{id(image_source)}"
            cache_path = self._get_cache_path(key)
            try:
                image_source.thumbnail(target_size)
                image_source.save(cache_path, "JPEG", quality=self.jpeg_quality)
                return cache_path
            except Exception:
                return None

        return None

    def generate_from_video(
        self,
        video_path: Union[str, Path],
        frame_number: int = 0,
        size: Optional[Tuple[int, int]] = None,
    ) -> Optional[Path]:
        """
        Extract a keyframe thumbnail from a video file.

        Args:
            video_path: Path to .mp4 / .avi / .mov video file.
            frame_number: Specific zero-indexed frame index to sample.
            size: Override thumbnail dimensions (width, height).

        Returns:
            Path to saved thumbnail on disk, or None if extraction failed.
        """
        p = Path(video_path)
        if not p.exists():
            return None

        key = f"{p.name}_{p.stat().st_size}_f{frame_number}"
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            return cache_path

        try:
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                return None
            if frame_number > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return None

            target_size = size or self.default_size
            thumb = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(cache_path), thumb, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            return cache_path
        except Exception:
            return None

    def to_base64_data_uri(self, thumb_path: Union[str, Path]) -> Optional[str]:
        """
        Convert a thumbnail image to an inline base64 data URI for HTML insertion.

        Args:
            thumb_path: Path to thumbnail image.

        Returns:
            String format: "data:image/jpeg;base64,..."
        """
        p = Path(thumb_path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception:
            return None
