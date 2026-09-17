"""
src/inference/predictor.py
===========================
Clean model interface layer for lane detection.

Design principles
-----------------
1. The rest of the application calls predict(image) and gets a LanePrediction.
2. It does NOT know whether the backend is:
      - classical computer vision (OpenCV)
      - semantic segmentation (PyTorch)
      - object detection (YOLO)
      - any future model
3. Swapping backends = swapping one class, nothing else changes.
4. Model status is explicit and can be queried at any time.

Usage
-----
    from src.inference.predictor import get_predictor, ModelStatus

    predictor = get_predictor("classical_cv")   # works immediately
    predictor = get_predictor("ml_segmentation") # needs trained model

    result = predictor.predict(preprocess_result)
    if result.status == ModelStatus.LANE_DETECTED:
        # use result.left_mask, result.right_mask
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.inference.preprocessing import PreprocessResult


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class ModelStatus(Enum):
    """
    Describes the current status of the prediction backend.

    NOT_TRAINED   : No trained model file exists yet.
    READY         : Model file found and loaded successfully.
    NOT_LOADED    : Model file exists but has not been loaded yet.
    LOAD_ERROR    : Model file found but failed to load.
    INFERENCE_ERROR: Model loaded but inference raised an exception.
    CLASSICAL_CV  : Classical OpenCV baseline (no ML model needed).
    """
    NOT_TRAINED     = "MODEL NOT TRAINED"
    READY           = "MODEL READY"
    NOT_LOADED      = "MODEL NOT LOADED"
    LOAD_ERROR      = "MODEL LOAD ERROR"
    INFERENCE_ERROR = "INFERENCE ERROR"
    CLASSICAL_CV    = "CLASSICAL CV BASELINE"


class DetectionStatus(Enum):
    """Result of a single lane detection inference pass."""
    LANE_DETECTED         = "LANE DETECTED"
    PARTIAL_LANE_DETECTED = "PARTIAL LANE DETECTED"
    LANE_NOT_DETECTED     = "LANE NOT DETECTED"
    MODEL_UNAVAILABLE     = "MODEL UNAVAILABLE"
    INFERENCE_ERROR       = "INFERENCE ERROR"


# ═══════════════════════════════════════════════════════════════════════════
# Prediction result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LanePrediction:
    """
    Output of BaseLanePredictor.predict().

    All downstream modules (lane geometry, visualization) consume this.

    Masks
    -----
    left_mask, right_mask : np.ndarray | None
        Binary uint8 arrays (0 or 255) at MODEL resolution (model_h x model_w).
        Pixel value 255 = lane marking present.
        None = that lane was not detected.

    Confidence
    ----------
    left_confidence, right_confidence : float in [0, 1]
        Estimated detection confidence per lane.
        0.0 = not detected / unknown.

    Backend info
    ------------
    model_status : ModelStatus
    backend_name : str
    inference_ms : float
    error_message : str | None
    """
    # Detection outcome
    status:           DetectionStatus = DetectionStatus.LANE_NOT_DETECTED
    model_status:     ModelStatus     = ModelStatus.NOT_TRAINED
    backend_name:     str             = "none"

    # Per-lane binary masks (model resolution, uint8)
    left_mask:        Optional[np.ndarray] = None
    right_mask:       Optional[np.ndarray] = None

    # Confidence [0, 1]
    left_confidence:  float = 0.0
    right_confidence: float = 0.0

    # Dimensions for coordinate scaling
    model_h:          int = 360
    model_w:          int = 640

    # Timing
    inference_ms:     float = 0.0

    # Error details
    error_message:    Optional[str] = None

    @property
    def left_detected(self) -> bool:
        return (
            self.left_mask is not None
            and self.left_mask.any()
            and self.left_confidence > 0.0
        )

    @property
    def right_detected(self) -> bool:
        return (
            self.right_mask is not None
            and self.right_mask.any()
            and self.right_confidence > 0.0
        )

    @property
    def both_detected(self) -> bool:
        return self.left_detected and self.right_detected


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base class — the model interface contract
# ═══════════════════════════════════════════════════════════════════════════

class BaseLanePredictor(ABC):
    """
    All lane detection backends must implement this interface.

    The application layer never imports a concrete predictor directly;
    it uses get_predictor() and works through this ABC.
    """

    @property
    @abstractmethod
    def model_status(self) -> ModelStatus:
        """Current status of this backend."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend identifier."""

    @abstractmethod
    def predict(self, preprocessed: PreprocessResult) -> LanePrediction:
        """
        Run inference on a preprocessed image.

        Must NEVER raise an exception — catch internally and return a
        LanePrediction with status=INFERENCE_ERROR instead.
        """


# ═══════════════════════════════════════════════════════════════════════════
# ML Segmentation predictor stub
# ═══════════════════════════════════════════════════════════════════════════

class MLSegmentationPredictor(BaseLanePredictor):
    """
    Placeholder for the trained PyTorch segmentation model.

    Currently returns MODEL_NOT_TRAINED because no model file exists yet.

    When a model is trained:
    1. Save it to models/exported/best_model.pth
    2. This class will load it automatically on next instantiation.
    3. The rest of the application requires ZERO changes.
    """

    MODEL_PATH = PROJECT_ROOT / "models" / "exported" / "best_model.pth"

    def __init__(self):
        self._status = ModelStatus.NOT_TRAINED
        self._model  = None
        self._try_load()

    def _try_load(self) -> None:
        if not self.MODEL_PATH.exists():
            self._status = ModelStatus.NOT_TRAINED
            return
        try:
            import torch
            checkpoint = torch.load(
                str(self.MODEL_PATH), map_location="cpu"
            )
            # ── When the model class is implemented, instantiate it here ──
            # from src.training.model import LaneSegNet
            # self._model = LaneSegNet(...)
            # self._model.load_state_dict(checkpoint["model_state_dict"])
            # self._model.eval()
            self._status = ModelStatus.READY
        except Exception as exc:
            self._status = ModelStatus.LOAD_ERROR
            self._model  = None
            self._load_error = str(exc)

    @property
    def model_status(self) -> ModelStatus:
        return self._status

    @property
    def backend_name(self) -> str:
        return "ML Segmentation (LaneSegNet)"

    def predict(self, preprocessed: PreprocessResult) -> LanePrediction:
        if self._status == ModelStatus.NOT_TRAINED:
            return LanePrediction(
                status        = DetectionStatus.MODEL_UNAVAILABLE,
                model_status  = self._status,
                backend_name  = self.backend_name,
                error_message = (
                    "Model not trained yet.\n"
                    "Complete dataset annotation and training before "
                    "running ML inference.\n\n"
                    "Run:  python run.py train"
                ),
            )
        if self._status == ModelStatus.LOAD_ERROR:
            return LanePrediction(
                status        = DetectionStatus.MODEL_UNAVAILABLE,
                model_status  = self._status,
                backend_name  = self.backend_name,
                error_message = f"Model failed to load: {getattr(self, '_load_error', 'unknown')}",
            )
        if self._model is None:
            return LanePrediction(
                status        = DetectionStatus.MODEL_UNAVAILABLE,
                model_status  = self._status,
                backend_name  = self.backend_name,
                error_message = "Model object is None despite READY status. Internal error.",
            )
        # ── Actual inference (enabled once model class is implemented) ─────
        try:
            import torch
            t0 = time.perf_counter()
            # tensor: (H, W, 3) float32 → (1, 3, H, W)
            t = torch.from_numpy(
                preprocessed.input_tensor.transpose(2, 0, 1)
            ).unsqueeze(0)
            with torch.no_grad():
                logits = self._model(t)          # (1, num_classes, H, W)
                probs  = torch.softmax(logits, dim=1).squeeze(0).numpy()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            left_prob  = probs[1]   # class 1 = left lane
            right_prob = probs[2]   # class 2 = right lane
            thresh     = 0.5

            left_mask  = (left_prob  >= thresh).astype(np.uint8) * 255
            right_mask = (right_prob >= thresh).astype(np.uint8) * 255
            left_conf  = float(left_prob.max())
            right_conf = float(right_prob.max())

            both  = left_mask.any() and right_mask.any()
            one   = left_mask.any() or  right_mask.any()
            det_status = (
                DetectionStatus.LANE_DETECTED         if both else
                DetectionStatus.PARTIAL_LANE_DETECTED if one  else
                DetectionStatus.LANE_NOT_DETECTED
            )
            return LanePrediction(
                status           = det_status,
                model_status     = ModelStatus.READY,
                backend_name     = self.backend_name,
                left_mask        = left_mask,
                right_mask       = right_mask,
                left_confidence  = left_conf,
                right_confidence = right_conf,
                model_h          = preprocessed.model_h,
                model_w          = preprocessed.model_w,
                inference_ms     = elapsed_ms,
            )
        except Exception as exc:
            return LanePrediction(
                status        = DetectionStatus.INFERENCE_ERROR,
                model_status  = ModelStatus.INFERENCE_ERROR,
                backend_name  = self.backend_name,
                error_message = f"Inference failed: {exc}",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def get_predictor(backend: str = "classical_cv") -> BaseLanePredictor:
    """
    Return the requested lane detection backend.

    Parameters
    ----------
    backend : "classical_cv" | "ml_segmentation"

    The classical_cv backend is always functional.
    The ml_segmentation backend requires a trained model file.
    """
    from src.inference.classical_cv import ClassicalCVPredictor  # local import to avoid circular

    if backend == "classical_cv":
        return ClassicalCVPredictor()
    elif backend == "ml_segmentation":
        return MLSegmentationPredictor()
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            "Choose 'classical_cv' or 'ml_segmentation'."
        )
