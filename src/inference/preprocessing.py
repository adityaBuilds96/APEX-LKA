"""
src/inference/preprocessing.py
================================
Image preprocessing pipeline for lane detection inference.

Responsibilities
----------------
* Load an image from disk or accept a numpy array / PIL image
* Validate the image (not None, correct dtype, minimum size)
* Resize to model input dimensions
* Normalize pixel values
* Return both the preprocessed tensor AND the original image
  (the original is needed to draw overlays at full resolution)

This module is model-agnostic — it does NOT know whether the downstream
consumer is a segmentation model, object detector, or classical CV pipeline.

Usage
-----
    from src.inference.preprocessing import preprocess, PreprocessResult
    result = preprocess("path/to/image.jpg")
    # result.original_bgr  → original OpenCV image
    # result.input_tensor  → normalized float32 numpy array (H,W,3), RGB
    # result.scale_factor  → (sx, sy) to map model coords back to original
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import cfg

# ── ImageNet normalization constants (RGB order) ───────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── Model input size from config ───────────────────────────────────────────
_MODEL_W = cfg["preprocessing"]["image_width"]   # default 640
_MODEL_H = cfg["preprocessing"]["image_height"]  # default 360


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessResult:
    """
    Container returned by preprocess().

    Attributes
    ----------
    original_bgr : np.ndarray
        Original image in BGR uint8 — used for overlay drawing.
    original_rgb : np.ndarray
        Original image in RGB uint8 — used for display in Streamlit.
    input_tensor : np.ndarray
        Normalized float32 array of shape (H, W, 3) in RGB order.
        This is what gets fed to the model or classical CV pipeline.
    resized_bgr : np.ndarray
        The image resized to model input dimensions, NOT normalized.
        Useful for classical CV (which needs uint8 pixel values).
    original_h : int
    original_w : int
    model_h : int
    model_w : int
    scale_x : float
        original_w / model_w — multiply model-space x coords by this.
    scale_y : float
        original_h / model_h — multiply model-space y coords by this.
    preprocess_ms : float
        Time taken for preprocessing in milliseconds.
    error : Optional[str]
        Set if preprocessing failed. All other fields may be None.
    """
    original_bgr:  Optional[np.ndarray] = None
    original_rgb:  Optional[np.ndarray] = None
    input_tensor:  Optional[np.ndarray] = None
    resized_bgr:   Optional[np.ndarray] = None
    original_h:    int = 0
    original_w:    int = 0
    model_h:       int = _MODEL_H
    model_w:       int = _MODEL_W
    scale_x:       float = 1.0
    scale_y:       float = 1.0
    preprocess_ms: float = 0.0
    error:         Optional[str] = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.original_bgr is not None


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(
    source: Union[str, Path, np.ndarray, bytes],
    model_w: int = _MODEL_W,
    model_h: int = _MODEL_H,
) -> PreprocessResult:
    """
    Load, validate, resize and normalise an image for model inference.

    Parameters
    ----------
    source : str | Path | np.ndarray | bytes
        - str / Path  → read from disk (any format OpenCV supports)
        - np.ndarray  → BGR uint8 image already in memory
        - bytes       → raw image bytes (e.g. from Streamlit file_uploader)
    model_w, model_h : int
        Target dimensions for the model input.

    Returns
    -------
    PreprocessResult
        Check .valid and .error before using other fields.
    """
    t0 = time.perf_counter()

    # ── 1. Load image ──────────────────────────────────────────────────────
    img_bgr = _load(source)
    if img_bgr is None:
        return PreprocessResult(
            error="Could not load image. "
                  "Check the file path or that the file is a valid image."
        )

    # ── 2. Validate ────────────────────────────────────────────────────────
    err = _validate(img_bgr)
    if err:
        return PreprocessResult(error=err)

    orig_h, orig_w = img_bgr.shape[:2]

    # ── 3. Resize ──────────────────────────────────────────────────────────
    resized = cv2.resize(img_bgr, (model_w, model_h),
                         interpolation=cv2.INTER_LINEAR)

    # ── 4. Normalise (BGR → RGB → float32 → ImageNet normalise) ───────────
    rgb    = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = (rgb - _MEAN) / _STD      # shape (H, W, 3), float32

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PreprocessResult(
        original_bgr  = img_bgr,
        original_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        input_tensor  = tensor,
        resized_bgr   = resized,
        original_h    = orig_h,
        original_w    = orig_w,
        model_h       = model_h,
        model_w       = model_w,
        scale_x       = orig_w / model_w,
        scale_y       = orig_h / model_h,
        preprocess_ms = elapsed_ms,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load(source: Union[str, Path, np.ndarray, bytes]) -> Optional[np.ndarray]:
    """Return BGR uint8 image or None on failure."""
    if isinstance(source, np.ndarray):
        if source.ndim == 3 and source.dtype == np.uint8:
            return source
        return None
    if isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Path or str
    p = Path(source)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img  # None if unreadable


def _validate(img: np.ndarray) -> Optional[str]:
    """Return an error string if the image is unusable."""
    if img is None:
        return "Image is None after loading."
    if img.ndim != 3 or img.shape[2] != 3:
        return (
            f"Unexpected image shape {img.shape}. "
            "Expected a 3-channel (H, W, 3) image."
        )
    h, w = img.shape[:2]
    if h < 64 or w < 64:
        return (
            f"Image too small ({w}x{h}). "
            "Minimum supported size is 64x64 pixels."
        )
    if img.dtype != np.uint8:
        return f"Unexpected dtype {img.dtype}. Expected uint8."
    return None
