"""
src/inference/onnx_predictor.py
===============================
Hardware-accelerated ONNX Runtime predictor for APEX-LKA:
- Conforms strictly to BaseLanePredictor interface
- Automated ExecutionProvider selection:
    1. TensorrtExecutionProvider (Jetson & datacenter)
    2. CUDAExecutionProvider (NVIDIA GPUs)
    3. OpenVINOExecutionProvider (Intel hardware)
    4. CPUExecutionProvider (Universal fallback)
- Resilient exception handling (predict() never raises)
- Fast vectorized postprocessing and lane mask extraction
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.inference.predictor import (
    BaseLanePredictor,
    DetectionStatus,
    LanePrediction,
    ModelStatus,
)
from src.inference.preprocessing import PreprocessResult, preprocess

log = logging.getLogger(__name__)


def select_best_execution_provider(
    requested_providers: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """
    Select available execution provider in priority order:
    TensorRT -> CUDA -> OpenVINO -> CPU
    """
    available = ort.get_available_providers()

    if requested_providers:
        valid = [p for p in requested_providers if p in available]
        if valid:
            return valid, valid[0]

    priority_order = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]

    selected: List[str] = []
    for prov in priority_order:
        if prov in available:
            selected.append(prov)

    if not selected:
        selected = ["CPUExecutionProvider"]

    return selected, selected[0]


class ONNXLanePredictor(BaseLanePredictor):
    """
    Universal ONNX Runtime Lane Prediction Backend.
    """

    DEFAULT_ONNX_PATH = PROJECT_ROOT / "models" / "exported" / "best_model.onnx"
    FP16_ONNX_PATH = PROJECT_ROOT / "models" / "exported" / "best_model_fp16.onnx"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        providers: Optional[List[str]] = None,
        confidence_threshold: float = 0.50,
    ) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self._status = ModelStatus.NOT_LOADED
        self._active_provider = "None"
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: str = "input_image"
        self._output_name: str = "segmentation_mask"
        self._input_shape: Tuple[Any, ...] = (1, 3, 360, 640)
        self._input_dtype = np.float32

        # 1. Resolve model file path
        if model_path is not None:
            self.model_path = Path(model_path)
        elif self.DEFAULT_ONNX_PATH.exists():
            self.model_path = self.DEFAULT_ONNX_PATH
        elif self.FP16_ONNX_PATH.exists():
            self.model_path = self.FP16_ONNX_PATH
        else:
            self.model_path = self.DEFAULT_ONNX_PATH

        self._load_session(providers)

    def _load_session(self, requested_providers: Optional[List[str]] = None) -> None:
        """Instantiate ONNX Runtime InferenceSession with chosen providers."""
        if not self.model_path.exists():
            self._status = ModelStatus.NOT_TRAINED
            log.warning("ONNX model file not found at %s", self.model_path)
            return

        try:
            chosen_providers, primary = select_best_execution_provider(requested_providers)
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            log.info("Loading ONNX session: %s on %s", self.model_path.name, chosen_providers)
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=chosen_providers,
            )

            # Retrieve active provider
            self._active_provider = self._session.get_providers()[0]

            # Inspect input metadata
            inputs = self._session.get_inputs()
            if inputs:
                self._input_name = inputs[0].name
                self._input_shape = tuple(inputs[0].shape)
                if inputs[0].type == "tensor(float16)":
                    self._input_dtype = np.float16
                else:
                    self._input_dtype = np.float32

            # Inspect output metadata
            outputs = self._session.get_outputs()
            if outputs:
                self._output_name = outputs[0].name

            self._status = ModelStatus.READY
            log.info("ONNX Lane Predictor initialized successfully (%s).", self._active_provider)

        except Exception as exc:
            self._status = ModelStatus.LOAD_ERROR
            self._session = None
            self._load_error = str(exc)
            log.error("Failed to load ONNX model %s: %s", self.model_path, exc)

    @property
    def model_status(self) -> ModelStatus:
        return self._status

    @property
    def backend_name(self) -> str:
        return f"ONNX Runtime [{self._active_provider}]"

    @property
    def active_provider(self) -> str:
        return self._active_provider

    def predict(self, preprocessed: PreprocessResult) -> LanePrediction:
        """
        Run inference on preprocessed image input tensor.
        Must NEVER raise an exception.
        """
        if self._status == ModelStatus.NOT_TRAINED:
            return LanePrediction(
                status=DetectionStatus.MODEL_UNAVAILABLE,
                model_status=self._status,
                backend_name=self.backend_name,
                error_message=f"ONNX model file not found at {self.model_path}. Run 'python run.py export' first.",
            )

        if self._status == ModelStatus.LOAD_ERROR or self._session is None:
            return LanePrediction(
                status=DetectionStatus.MODEL_UNAVAILABLE,
                model_status=self._status,
                backend_name=self.backend_name,
                error_message=f"ONNX session unavailable: {getattr(self, '_load_error', 'Unknown load error')}",
            )

        try:
            t0 = time.perf_counter()

            # Input tensor from PreprocessResult is (H, W, 3) in RGB, normalized
            raw_input = preprocessed.input_tensor
            if raw_input is None:
                raise ValueError("PreprocessResult.input_tensor is None")

            # Convert (H, W, 3) -> (1, 3, H, W)
            tensor_nchw = np.transpose(raw_input, (2, 0, 1))[np.newaxis, ...]
            tensor_nchw = tensor_nchw.astype(self._input_dtype)

            # Execute ONNX Runtime inference
            feed_dict = {self._input_name: tensor_nchw}
            outputs = self._session.run([self._output_name], feed_dict)
            logits = outputs[0]  # Shape: (1, 4, H, W) or (4, H, W)

            if logits.ndim == 4:
                logits = logits[0]

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Compute Softmax probabilities: (4, H, W)
            # Subtract max along channel axis for numerical stability
            e_x = np.exp(logits - np.max(logits, axis=0, keepdims=True))
            probs = e_x / np.sum(e_x, axis=0, keepdims=True)

            # Class mapping:
            # 0: Background, 1: Road, 2: Left Lane, 3: Right Lane
            road_prob = probs[1]
            left_prob = probs[2]
            right_prob = probs[3]

            thresh = self.confidence_threshold
            left_mask = ((left_prob >= thresh) * 255).astype(np.uint8)
            right_mask = ((right_prob >= thresh) * 255).astype(np.uint8)
            road_mask = ((road_prob >= 0.40) * 255).astype(np.uint8)

            left_conf = float(np.max(left_prob)) if left_mask.any() else float(np.mean(left_prob))
            right_conf = float(np.max(right_prob)) if right_mask.any() else float(np.mean(right_prob))

            left_det = bool(left_mask.any() and left_conf >= thresh)
            right_det = bool(right_mask.any() and right_conf >= thresh)

            if left_det and right_det:
                status = DetectionStatus.LANE_DETECTED
            elif left_det or right_det:
                status = DetectionStatus.PARTIAL_LANE_DETECTED
            else:
                status = DetectionStatus.LANE_NOT_DETECTED

            # Accuracy score proxy from confidence
            accuracy_score = float((left_conf + right_conf) / 2.0) if (left_det or right_det) else 0.0

            return LanePrediction(
                status=status,
                model_status=ModelStatus.READY,
                backend_name=self.backend_name,
                left_mask=left_mask,
                right_mask=right_mask,
                road_mask=road_mask,
                left_confidence=round(left_conf, 3),
                right_confidence=round(right_conf, 3),
                model_h=preprocessed.model_h,
                model_w=preprocessed.model_w,
                inference_ms=round(elapsed_ms, 2),
                accuracy_score=round(accuracy_score, 3),
                geometry_plausibility=0.90 if (left_det and right_det) else 0.50,
                detection_mode="painted",
            )

        except Exception as exc:
            log.exception("ONNX inference exception: %s", exc)
            return LanePrediction(
                status=DetectionStatus.INFERENCE_ERROR,
                model_status=ModelStatus.INFERENCE_ERROR,
                backend_name=self.backend_name,
                error_message=f"Inference error: {exc}",
            )

    def predict_image(self, image_bgr: np.ndarray) -> LanePrediction:
        """Convenience method accepting raw BGR image array."""
        prep = preprocess(image_bgr)
        return self.predict(prep)

    def benchmark(
        self,
        iterations: int = 50,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """Measure inference latency percentiles and FPS."""
        if self._session is None:
            return {"mean_ms": 0.0, "fps": 0.0}

        dummy = np.random.randn(1, 3, 360, 640).astype(self._input_dtype)
        feed = {self._input_name: dummy}

        # Warmup
        for _ in range(warmup):
            _ = self._session.run(None, feed)

        latencies: List[float] = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = self._session.run(None, feed)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(latencies, dtype=np.float64)
        mean_ms = float(np.mean(arr))
        p50_ms = float(np.percentile(arr, 50))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))
        fps = 1000.0 / max(0.1, mean_ms)

        return {
            "mean_ms": round(mean_ms, 2),
            "p50_ms": round(p50_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "fps": round(fps, 1),
        }
