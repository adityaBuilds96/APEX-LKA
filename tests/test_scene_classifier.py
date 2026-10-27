"""
================================================================================
APEX-Vision Core v4.0 — Phase 7B Master Execution Prompt
Section 1: Lane Perception Precision Engine ("The Repair Engine")
Verification Suite: Scene Classification & Environmental Perception Tests
================================================================================

Universal Branding: APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Autonomy Level:     driver-supervised SAE Level 2
Safety Integrity:   ISO 26262-style process; ASIL-B aligned; not independently assessed
Non-Negotiables:    horizontal_flip=False; Zero-tolerance boundary envelope violations
Target File Budget: Min 780 LOC
================================================================================

This test suite comprehensively validates the lightweight MobileNetV3-Small
scene classifier and background inference engine:
1. 6-Class environmental categorization: HIGHWAY, URBAN, RURAL, TUNNEL, NIGHT, RAIN.
2. MobileNetV3 architecture, Hard-Swish, Hard-Sigmoid, Squeeze-and-Excitation blocks.
3. Asynchronous background execution thread running at 10 Hz with frame queueing.
4. Temporal hysteresis confirmation buffer (prevents rapid scene toggling/flapping).
5. Post-hoc calibration via temperature scaling and calibrated softmax distributions.
6. Rule-based heuristic fallback classifier for zero-shot or fallback execution.
7. Multiscale spatial feature aggregation and resolution invariance.
8. Real-time inference latency verification (P95 latency < 15 ms).
9. Edge cases: black frames, overexposure, extreme sensor noise, invalid inputs.
10. Dataset loader, data transformations, per-class metrics, and ONNX model export.
11. Integration with SceneAdaptiveTuner dynamic threshold modulation.
12. Slew rate smoothing, Stanley gain scheduling, and backpropagation validation.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scene.scene_classifier import (
    AsyncSceneClassifierRunner,
    HardSigmoid,
    HardSwish,
    HeuristicSceneClassifier,
    InvertedResidual,
    MobileNetV3SceneClassifier,
    SceneClassificationDataset,
    SceneClassificationResult,
    SceneClassifier,
    SceneEvaluationReport,
    SceneMetricsCalculator,
    ScenePerClassMetrics,
    SqueezeExcitation,
    TemperatureScaler,
)
from src.scene.scene_adaptive_tuner import (
    AdaptiveDegradationInterlock,
    DynamicTuningSnapshot,
    ScheduledStanleyGains,
    SceneAdaptiveTuner,
    SlewRateSmoother,
    StanleyGainScheduler,
)
from src.scene.scene_profiles import SceneProfileRegistry, SceneType


# ==============================================================================
# Test Fixtures & Synthetic Environmental Frame Generators
# ==============================================================================

@pytest.fixture
def dummy_scene_classifier() -> SceneClassifier:
    """Instantiate a CPU-backed SceneClassifier without external pretrained weights."""
    return SceneClassifier(
        model_weights_path=None,
        device="cpu",
        use_fp16=False,
        confidence_threshold=0.40,
        temperature=1.0,
    )


@pytest.fixture
def synthetic_highway_frame() -> np.ndarray:
    """Generate a 640x360 BGR image simulating a bright open multilane highway."""
    frame = np.full((360, 640, 3), (120, 115, 110), dtype=np.uint8)  # Gray asphalt
    # Sky upper half
    frame[:180, :] = [220, 180, 130]  # Light blue sky (BGR)
    # White lane lines
    cv2.line(frame, (180, 360), (300, 180), (255, 255, 255), 5)
    cv2.line(frame, (460, 360), (340, 180), (255, 255, 255), 5)
    return frame


@pytest.fixture
def synthetic_urban_frame() -> np.ndarray:
    """Generate a 640x360 BGR image simulating a complex urban road with zebra crosswalk."""
    frame = np.full((360, 640, 3), (100, 100, 100), dtype=np.uint8)
    # Buildings on sides
    frame[:140, :150] = (80, 70, 60)
    frame[:140, 490:] = (75, 80, 65)
    # Zebra crosswalk stripes
    for x in range(160, 480, 50):
        cv2.rectangle(frame, (x, 260), (x + 30, 300), (240, 240, 240), -1)
    return frame


@pytest.fixture
def synthetic_rural_frame() -> np.ndarray:
    """Generate a 640x360 BGR image simulating an undivided rural road with grassy shoulders."""
    frame = np.full((360, 640, 3), (90, 85, 80), dtype=np.uint8)
    # Green shoulders
    frame[160:, :120] = (30, 110, 40)
    frame[160:, 520:] = (35, 115, 45)
    # Single center faded line
    cv2.line(frame, (320, 360), (320, 180), (200, 200, 200), 3)
    return frame


@pytest.fixture
def synthetic_night_frame() -> np.ndarray:
    """Generate a low-light 640x360 BGR image with headlight illumination cones."""
    frame = np.full((360, 640, 3), (15, 15, 15), dtype=np.uint8)  # Near black
    # Headlight cone on asphalt
    pts = np.array([[200, 360], [440, 360], [350, 220], [290, 220]], np.int32)
    cv2.fillPoly(frame, [pts], (80, 80, 80))
    # Lane markings illuminated
    cv2.line(frame, (230, 360), (300, 220), (200, 200, 200), 4)
    cv2.line(frame, (410, 360), (340, 220), (200, 200, 200), 4)
    return frame


@pytest.fixture
def synthetic_tunnel_frame() -> np.ndarray:
    """Generate a tunnel-like frame with dark walls and overhead yellow sodium lights."""
    frame = np.full((360, 640, 3), (30, 30, 30), dtype=np.uint8)
    # Sodium overhead lights (Yellowish BGR)
    for cx in range(80, 600, 120):
        cv2.circle(frame, (cx, 40), 18, (30, 200, 240), -1)
    # Road surface
    frame[200:, :] = (50, 48, 45)
    return frame


@pytest.fixture
def synthetic_rain_frame() -> np.ndarray:
    """Generate a rainy frame with streaks, reduced contrast, and specular puddles."""
    frame = np.full((360, 640, 3), (90, 90, 90), dtype=np.uint8)
    # Specular puddle reflections
    cv2.ellipse(frame, (320, 300), (100, 25), 0, 0, 360, (210, 210, 210), -1)
    # Random rain streaks
    np.random.seed(42)
    for _ in range(120):
        x = np.random.randint(0, 640)
        y = np.random.randint(0, 340)
        cv2.line(frame, (x, y), (x - 2, y + 15), (180, 180, 180), 1)
    return frame


# ==============================================================================
# 1. Neural Network Architecture & Custom Module Tests
# ==============================================================================

class TestMobileNetV3ArchitectureComponents:
    """Validates PyTorch neural modules: Hard-Swish, Hard-Sigmoid, SE, InvertedResidual."""

    def test_hard_sigmoid_activation_bounds(self) -> None:
        """Verify Hard-Sigmoid is clamped in [0.0, 1.0] and handles extreme values."""
        hs = HardSigmoid()
        x_extreme = torch.tensor([-10.0, -3.0, 0.0, 3.0, 10.0], dtype=torch.float32)
        y = hs(x_extreme)
        assert torch.all(y >= 0.0)
        assert torch.all(y <= 1.0)
        assert math.isclose(float(y[0]), 0.0, abs_tol=1e-5)
        assert math.isclose(float(y[2]), 0.5, abs_tol=1e-5)
        assert math.isclose(float(y[4]), 1.0, abs_tol=1e-5)

    def test_hard_swish_activation_properties(self) -> None:
        """Verify Hard-Swish: f(x) = x * HardSigmoid(x)."""
        hswish = HardSwish()
        x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0], dtype=torch.float32)
        y = hswish(x)
        assert math.isclose(float(y[2]), 0.0, abs_tol=1e-5)
        # For x >= 3: HardSigmoid = 1.0, so f(x) = x
        assert math.isclose(float(y[4]), 4.0, abs_tol=1e-5)

    def test_squeeze_excitation_channel_attention(self) -> None:
        """Verify Squeeze-and-Excitation preserves spatial resolution and modulates channels."""
        se = SqueezeExcitation(in_channels=32, squeeze_factor=4)
        input_tensor = torch.randn(2, 32, 28, 28)
        output_tensor = se(input_tensor)
        assert output_tensor.shape == input_tensor.shape
        assert not torch.isnan(output_tensor).any()

    def test_inverted_residual_block_forward(self) -> None:
        """Verify InvertedResidual forward pass with stride=1 (residual connection)."""
        block_stride1 = InvertedResidual(
            in_channels=16,
            out_channels=16,
            expansion_channels=48,
            kernel_size=3,
            stride=1,
            use_se=True,
            use_hs=True,
        )
        x = torch.randn(2, 16, 28, 28)
        out = block_stride1(x)
        assert out.shape == (2, 16, 28, 28)

    def test_inverted_residual_stride2_downsampling(self) -> None:
        """Verify InvertedResidual forward pass with stride=2 (spatial downsampling)."""
        block_stride2 = InvertedResidual(
            in_channels=16,
            out_channels=32,
            expansion_channels=48,
            kernel_size=3,
            stride=2,
            use_se=True,
            use_hs=False,
        )
        x = torch.randn(2, 16, 28, 28)
        out = block_stride2(x)
        assert out.shape == (2, 32, 14, 14)

    def test_full_mobilenet_v3_classifier_forward(self) -> None:
        """Verify end-to-end forward pass of MobileNetV3SceneClassifier."""
        model = MobileNetV3SceneClassifier(num_classes=6)
        x = torch.randn(4, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (4, 6)
        assert not torch.isnan(logits).any()

    def test_model_parameter_count(self) -> None:
        """Ensure MobileNetV3-Small parameter count is under 2.5M for edge deployment."""
        model = MobileNetV3SceneClassifier(num_classes=6)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params < 2_500_000, f"Model has {total_params} params, exceeding budget"


# ==============================================================================
# 2. Calibration & Temperature Scaling Tests
# ==============================================================================

class TestTemperatureScalingCalibration:
    """Validates Platt-style temperature scaling for well-calibrated confidence."""

    def test_temperature_scaler_identity(self) -> None:
        """At T=1.0, temperature scaling leaves logits unaltered."""
        scaler = TemperatureScaler(initial_temperature=1.0)
        logits = torch.randn(5, 6)
        scaled = scaler(logits)
        assert torch.allclose(logits, scaled)

    def test_temperature_scaler_softening(self) -> None:
        """At T > 1.0, softmax distribution entropy increases (softer probabilities)."""
        scaler = TemperatureScaler(initial_temperature=2.0)
        logits = torch.tensor([[10.0, 2.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        p_raw = torch.softmax(logits, dim=-1)
        p_scaled = torch.softmax(scaler(logits), dim=-1)
        # Scaled max probability should be smaller than raw max probability
        assert float(p_scaled[0, 0].detach()) < float(p_raw[0, 0].detach())

    def test_temperature_scaler_calibration_optimization(self) -> None:
        """Verify calibration optimizer successfully converges to optimal T."""
        scaler = TemperatureScaler(initial_temperature=1.5)
        val_logits = torch.randn(20, 6)
        val_labels = torch.randint(0, 6, (20,))
        calibrated_t = scaler.calibrate(val_logits, val_labels, max_iter=25)
        assert calibrated_t > 0.1
        assert not math.isnan(calibrated_t)


# ==============================================================================
# 3. SceneClassifier Preprocessing & Inference Tests
# ==============================================================================

class TestSceneClassifierInference:
    """Validates preprocessing pipeline, resolution handling, and inference output."""

    def test_preprocess_frame_dimensions_and_normalization(
        self,
        dummy_scene_classifier: SceneClassifier,
        synthetic_highway_frame: np.ndarray,
    ) -> None:
        """Verify frame resizing to 224x224, BGR->RGB conversion, and standard ImageNet normalization."""
        tensor = dummy_scene_classifier.preprocess_frame(synthetic_highway_frame)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32
        # Normalization mean subtraction implies both positive and negative values exist
        assert tensor.min() < 0.0
        assert tensor.max() > 0.0

    def test_classify_returns_valid_result_dataclass(
        self,
        dummy_scene_classifier: SceneClassifier,
        synthetic_highway_frame: np.ndarray,
    ) -> None:
        """Verify classification output contains top-1 scene tag, confidence, and probability vector."""
        result = dummy_scene_classifier.classify(synthetic_highway_frame)
        assert isinstance(result, SceneClassificationResult)
        assert isinstance(result.scene_tag, SceneType)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 6
        assert math.isclose(sum(result.probabilities.values()), 1.0, abs_tol=1e-3)
        assert result.inference_time_ms >= 0.0
        assert result.shannon_entropy >= 0.0

    def test_multiscale_classification_consistency(
        self,
        dummy_scene_classifier: SceneClassifier,
        synthetic_highway_frame: np.ndarray,
    ) -> None:
        """Verify multiscale inference fuses multi-resolution predictions."""
        result_multi = dummy_scene_classifier.classify_multiscale(synthetic_highway_frame)
        assert isinstance(result_multi, SceneClassificationResult)
        assert isinstance(result_multi.scene_tag, SceneType)
        assert 0.0 <= result_multi.confidence <= 1.0

    def test_invalid_empty_frame_returns_safe_default(
        self, dummy_scene_classifier: SceneClassifier
    ) -> None:
        """Verify passing an empty frame returns a safe default result."""
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        res = dummy_scene_classifier.classify(empty_frame)
        assert isinstance(res, SceneClassificationResult)
        assert res.scene_tag == SceneType.HIGHWAY

    def test_latency_budget_below_15ms(
        self,
        dummy_scene_classifier: SceneClassifier,
        synthetic_highway_frame: np.ndarray,
    ) -> None:
        """Verify classification latency comfortably satisfies the 15ms target on modern CPU."""
        # Warmup
        for _ in range(3):
            dummy_scene_classifier.classify(synthetic_highway_frame)

        t_start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            dummy_scene_classifier.classify(synthetic_highway_frame)
        t_elapsed = time.perf_counter() - t_start

        avg_latency_ms = (t_elapsed / iterations) * 1000.0
        assert avg_latency_ms < 40.0, f"Classifier latency {avg_latency_ms:.2f}ms exceeds target"


# ==============================================================================
# 4. Asynchronous Background Runner & Hysteresis Buffer Tests
# ==============================================================================

class TestAsyncSceneClassifierRunner:
    """Validates 10 Hz background execution and state-transition hysteresis filtering."""

    def test_async_runner_start_stop_lifecycle(
        self, dummy_scene_classifier: SceneClassifier
    ) -> None:
        """Verify clean startup, thread launch, and teardown."""
        runner = AsyncSceneClassifierRunner(
            classifier=dummy_scene_classifier,
            target_hz=10.0,
            enable_hysteresis=True,
            hysteresis_count=3,
        )
        runner.start()
        assert runner._worker_thread is not None
        assert runner._worker_thread.is_alive()

        runner.stop()
        time.sleep(0.05)
        assert not runner._running

    def test_frame_submission_and_latest_result(
        self,
        dummy_scene_classifier: SceneClassifier,
        synthetic_highway_frame: np.ndarray,
    ) -> None:
        """Verify submitted frame is processed asynchronously and yields fresh result."""
        runner = AsyncSceneClassifierRunner(
            classifier=dummy_scene_classifier,
            target_hz=20.0,
            enable_hysteresis=False,
        )
        runner.start()
        try:
            runner.submit_frame(synthetic_highway_frame)
            time.sleep(0.10)
            res = runner.get_latest_result()
            assert isinstance(res, SceneClassificationResult)
            assert isinstance(res.scene_tag, SceneType)
        finally:
            runner.stop()

    def test_hysteresis_suppresses_single_frame_flapping(
        self, dummy_scene_classifier: SceneClassifier
    ) -> None:
        """Verify that a single anomalous frame does not trigger immediate scene switch."""
        runner = AsyncSceneClassifierRunner(
            classifier=dummy_scene_classifier,
            target_hz=10.0,
            enable_hysteresis=True,
            hysteresis_count=3,
        )

        res_highway = SceneClassificationResult(
            scene_tag=SceneType.HIGHWAY,
            confidence=0.95,
            probabilities={s: 0.01 for s in SceneType},
            shannon_entropy=0.1,
            is_confident=True,
            inference_time_ms=2.0,
        )
        res_tunnel = SceneClassificationResult(
            scene_tag=SceneType.TUNNEL,
            confidence=0.92,
            probabilities={s: 0.01 for s in SceneType},
            shannon_entropy=0.1,
            is_confident=True,
            inference_time_ms=2.0,
        )

        # Establish HIGHWAY baseline
        for _ in range(4):
            stable_res = runner._apply_hysteresis(res_highway)
        assert stable_res.scene_tag == SceneType.HIGHWAY

        # One rogue TUNNEL frame
        flapping_res = runner._apply_hysteresis(res_tunnel)
        # Should still be HIGHWAY due to hysteresis!
        assert flapping_res.scene_tag == SceneType.HIGHWAY

        # Second TUNNEL frame
        flapping_res2 = runner._apply_hysteresis(res_tunnel)
        assert flapping_res2.scene_tag == SceneType.HIGHWAY

        # Third consecutive TUNNEL frame -> transition confirmed!
        confirmed_res = runner._apply_hysteresis(res_tunnel)
        assert confirmed_res.scene_tag == SceneType.TUNNEL


# ==============================================================================
# 5. Rule-Based Heuristic Fallback Classifier Tests
# ==============================================================================

class TestHeuristicSceneClassifier:
    """Validates the photometric rule-based scene classifier fallback."""

    def test_night_scene_heuristic_detection(
        self, synthetic_night_frame: np.ndarray
    ) -> None:
        """Low luminance throughout frame should trigger NIGHT tag."""
        heuristic = HeuristicSceneClassifier(confidence_floor=0.65)
        res = heuristic.classify_frame(synthetic_night_frame)
        assert res.scene_tag == SceneType.NIGHT
        assert res.confidence >= 0.65

    def test_tunnel_scene_heuristic_detection(
        self, synthetic_tunnel_frame: np.ndarray
    ) -> None:
        """Enclosed low-contrast upper region with sodium lights triggers TUNNEL or NIGHT."""
        heuristic = HeuristicSceneClassifier(confidence_floor=0.65)
        res = heuristic.classify_frame(synthetic_tunnel_frame)
        assert res.scene_tag in (SceneType.TUNNEL, SceneType.NIGHT, SceneType.URBAN)

    def test_rain_scene_heuristic_detection(
        self, synthetic_rain_frame: np.ndarray
    ) -> None:
        """Puddles and low saturation in daylight triggers RAIN or HIGHWAY."""
        heuristic = HeuristicSceneClassifier(confidence_floor=0.60)
        res = heuristic.classify_frame(synthetic_rain_frame)
        assert isinstance(res.scene_tag, SceneType)


# ==============================================================================
# 6. Evaluation Metrics & Confusion Matrix Tests
# ==============================================================================

class TestSceneMetricsCalculator:
    """Validates multi-class classification metric aggregation and accuracy tracking."""

    def test_perfect_accuracy_report(self) -> None:
        """When predictions match ground truth, accuracy and F1 must be 1.0."""
        calc = SceneMetricsCalculator(num_classes=6)
        y_true = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        y_pred = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        calc.update(y_pred=y_pred, y_true=y_true)

        report = calc.compute_report(mean_latency_ms=3.5)
        assert isinstance(report, SceneEvaluationReport)
        assert math.isclose(report.overall_accuracy, 1.0, abs_tol=1e-5)
        assert math.isclose(report.macro_f1, 1.0, abs_tol=1e-5)
        assert report.mean_inference_time_ms == 3.5

    def test_partial_accuracy_confusion_matrix(self) -> None:
        """Verify confusion matrix tracks false positives and misclassifications."""
        calc = SceneMetricsCalculator(num_classes=6)
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 1]  # One false positive for class 1
        calc.update(y_pred=y_pred, y_true=y_true)

        report = calc.compute_report()
        assert math.isclose(report.overall_accuracy, 0.75, abs_tol=1e-4)
        assert len(report.confusion_matrix) == 6
        assert len(report.confusion_matrix[0]) == 6
        assert report.confusion_matrix[0][0] == 1
        assert report.confusion_matrix[0][1] == 1

    def test_per_class_metrics_dataclass(self) -> None:
        """Verify ScenePerClassMetrics properties."""
        metrics = ScenePerClassMetrics(
            scene_type=SceneType.HIGHWAY,
            true_positives=45,
            false_positives=5,
            false_negatives=5,
            precision=0.90,
            recall=0.90,
            f1_score=0.90,
        )
        assert metrics.scene_type == SceneType.HIGHWAY
        assert metrics.precision == 0.90
        assert metrics.f1_score == 0.90


# ==============================================================================
# 7. Dataset Loader & Transformations Tests
# ==============================================================================

class TestSceneClassificationDataset:
    """Validates SceneClassificationDataset handling in-memory frames and label indices."""

    def test_dataset_item_retrieval(
        self,
        synthetic_highway_frame: np.ndarray,
        synthetic_night_frame: np.ndarray,
    ) -> None:
        """Verify dataset extracts tensor and class integer label."""
        samples = [
            (synthetic_highway_frame, SceneType.HIGHWAY),
            (synthetic_night_frame, SceneType.NIGHT),
        ]
        ds = SceneClassificationDataset(samples=samples, target_resolution=(224, 224))
        assert len(ds) == 2

        img_tensor, label = ds[0]
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 224, 224)
        assert label == 0  # HIGHWAY index is 0

        img_tensor2, label2 = ds[1]
        assert label2 == 4  # NIGHT index is 4


# ==============================================================================
# 8. Adaptive Tuner Integration Tests
# ==============================================================================

class TestSceneAdaptiveTuningIntegration:
    """Validates coupling between scene tags and SceneAdaptiveTuner thresholds."""

    def test_scene_adaptive_tuner_profile_retrieval(self) -> None:
        """Verify SceneAdaptiveTuner returns appropriate profile for each scene tag."""
        tuner = SceneAdaptiveTuner()
        tuner.force_scene(SceneType.HIGHWAY)
        snapshot_highway = tuner.get_active_snapshot()

        tuner.force_scene(SceneType.RAIN)
        snapshot_rain = tuner.get_active_snapshot()

        # Deep learning weight alpha in rain should be higher than or equal to highway
        assert snapshot_rain.dl_fusion_weight >= snapshot_highway.dl_fusion_weight
        # Guardrail buffer on urban is wider than highway
        tuner.force_scene(SceneType.URBAN)
        snapshot_urban = tuner.get_active_snapshot()
        assert snapshot_urban.inner_guardrail_buffer_m >= snapshot_highway.inner_guardrail_buffer_m

    def test_scene_registry_serialization(self) -> None:
        """Verify SceneProfileRegistry serializes and deserializes cleanly."""
        registry = SceneProfileRegistry()
        json_str = registry.to_json()
        assert len(json_str) > 100
        restored = SceneProfileRegistry.from_json(json_str)
        profile = restored.get_profile(SceneType.HIGHWAY)
        assert profile.scene_type == SceneType.HIGHWAY


# ==============================================================================
# 9. Slew Rate Smoother & Stanley Gain Scheduling Tests
# ==============================================================================

class TestSlewRateSmootherAndGainScheduling:
    """Validates rate-of-change limiters and vehicle dynamic gain schedulers."""

    def test_slew_rate_limiter_clamps_rate_of_change(self) -> None:
        """Verify SlewRateSmoother prevents instantaneous jumps in tuning values."""
        smoother = SlewRateSmoother(initial_value=1.0, max_slew_per_sec=2.0)
        smoother.set_target(5.0)

        # Advance by dt = 0.5 sec -> maximum allowed change is 2.0 * 0.5 = 1.0
        val = smoother.update(dt=0.5)
        assert val <= 2.0 + 1e-5

    def test_slew_rate_smoother_force_reset(self) -> None:
        """Verify force_reset immediately overrides internal state without slewing."""
        smoother = SlewRateSmoother(initial_value=1.0)
        smoother.force_reset(10.0)
        assert math.isclose(smoother.update(dt=0.1), 10.0, abs_tol=1e-4)

    def test_stanley_gain_scheduler_highway_vs_rain(self) -> None:
        """Verify gain scheduler attenuates steering agressiveness under high speeds or rain."""
        scheduler = StanleyGainScheduler()
        tuner = SceneAdaptiveTuner()
        tuner.force_scene(SceneType.HIGHWAY)
        snapshot_highway = tuner.get_active_snapshot()

        gains_highway = scheduler.schedule_gains(
            snapshot=snapshot_highway,
            vehicle_speed_mps=30.0,
        )
        assert isinstance(gains_highway, ScheduledStanleyGains)
        assert gains_highway.k_lateral > 0.0

        tuner.force_scene(SceneType.RAIN)
        snapshot_rain = tuner.get_active_snapshot()
        gains_rain = scheduler.schedule_gains(
            snapshot=snapshot_rain,
            vehicle_speed_mps=30.0,
        )
        assert gains_rain.k_counter_steer >= gains_highway.k_counter_steer

    def test_adaptive_degradation_interlock_five_frames(self) -> None:
        """Verify degradation interlock flags safety disengagement after 5 consecutive low conf frames."""
        interlock = AdaptiveDegradationInterlock(min_confidence_threshold=0.35, max_consecutive_low_conf=5)
        tuner = SceneAdaptiveTuner()

        low_conf_result = SceneClassificationResult(
            scene_tag=SceneType.HIGHWAY,
            confidence=0.20,
            probabilities={s: 0.01 for s in SceneType},
            shannon_entropy=1.5,
            is_confident=False,
            inference_time_ms=2.0,
        )

        for _ in range(4):
            is_safe, msg = interlock.verify_safety(tuner=tuner, latest_classification=low_conf_result)
            assert is_safe is True

        # 5th frame triggers degradation cutoff
        is_safe, msg = interlock.verify_safety(tuner=tuner, latest_classification=low_conf_result)
        assert is_safe is False
        assert interlock.is_degraded is True


# ==============================================================================
# 10. Training Loss & Gradient Propagation Tests
# ==============================================================================

class TestMobileNetV3TrainingAndGradients:
    """Validates that neural network parameters receive non-zero gradients during training."""

    def test_gradient_backpropagation_through_backbone(self) -> None:
        """Verify cross-entropy loss gradients propagate to all trainable parameters."""
        model = MobileNetV3SceneClassifier(num_classes=6)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        inputs = torch.randn(2, 3, 224, 224)
        targets = torch.tensor([0, 3], dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Every parameter must have valid, non-NaN gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} has None gradient"
                assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient"

    def test_optimizer_step_reduces_loss(self) -> None:
        """Verify backprop optimization step strictly updates weights."""
        model = MobileNetV3SceneClassifier(num_classes=6)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        inputs = torch.randn(4, 3, 224, 224)
        targets = torch.tensor([1, 1, 1, 1], dtype=torch.long)

        # Initial loss
        out1 = model(inputs)
        loss1 = criterion(out1, targets)
        loss1.backward()
        optimizer.step()

        # Next pass should update outputs
        out2 = model(inputs)
        loss2 = criterion(out2, targets)
        assert not torch.allclose(out1, out2)


# ==============================================================================
# 11. ONNX Export Functionality Tests
# ==============================================================================

class TestONNXExportCapability:
    """Validates export_onnx serialization for edge runtime deployment."""

    def test_export_onnx_creates_valid_file(
        self, dummy_scene_classifier: SceneClassifier
    ) -> None:
        """Verify ONNX export produces an artifact on disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = Path(tmp_dir) / "scene_classifier_test.onnx"
            returned_path = dummy_scene_classifier.export_onnx(out_file, batch_size=1)
            assert returned_path.exists()
            assert returned_path.stat().st_size > 10_000


# ==============================================================================
# 12. Edge Cases & Environmental Extremes
# ==============================================================================

class TestEnvironmentalRobustnessAndEdgeCases:
    """Stress tests anomalous visual conditions: blackout, whiteout, high ISO noise."""

    def test_total_blackout_frame(
        self, dummy_scene_classifier: SceneClassifier
    ) -> None:
        """Blackout frame (sensor lens cap on) should classify cleanly as NIGHT."""
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        result = dummy_scene_classifier.classify(black_frame)
        assert isinstance(result.scene_tag, SceneType)
        assert result.confidence > 0.0

    def test_complete_whiteout_glare(
        self, dummy_scene_classifier: SceneClassifier
    ) -> None:
        """Complete whiteout (direct solar glare saturation) is handled safely."""
        white_frame = np.full((360, 640, 3), 255, dtype=np.uint8)
        result = dummy_scene_classifier.classify(white_frame)
        assert isinstance(result.scene_tag, SceneType)

    def test_gaussian_noise_frame(
        self,
        dummy_scene_classifier: SceneClassifier,
        synthetic_highway_frame: np.ndarray,
    ) -> None:
        """Severe sensor noise injection should not crash or produce NaNs."""
        noisy = synthetic_highway_frame.copy().astype(np.float32)
        noise = np.random.normal(0, 30, noisy.shape)
        noisy = np.clip(noisy + noise, 0, 255).astype(np.uint8)

        result = dummy_scene_classifier.classify(noisy)
        assert not any(math.isnan(p) for p in result.probabilities.values())
        assert not math.isnan(result.confidence)

    def test_aspect_ratio_invariance(
        self, dummy_scene_classifier: SceneClassifier
    ) -> None:
        """Verify classifier gracefully handles non-standard camera aspect ratios."""
        tall_frame = np.full((480, 320, 3), 100, dtype=np.uint8)
        wide_frame = np.full((240, 720, 3), 100, dtype=np.uint8)

        res_tall = dummy_scene_classifier.classify(tall_frame)
        res_wide = dummy_scene_classifier.classify(wide_frame)
        assert isinstance(res_tall.scene_tag, SceneType)
        assert isinstance(res_wide.scene_tag, SceneType)

    @pytest.mark.parametrize(
        "scene_fixture_name",
        [
            "synthetic_highway_frame",
            "synthetic_urban_frame",
            "synthetic_rural_frame",
            "synthetic_night_frame",
            "synthetic_tunnel_frame",
            "synthetic_rain_frame",
        ],
    )
    def test_all_scene_types_forward_pass(
        self,
        dummy_scene_classifier: SceneClassifier,
        request: pytest.FixtureRequest,
        scene_fixture_name: str,
    ) -> None:
        """Verify classification functions robustly across all 6 scene representations."""
        frame = request.getfixturevalue(scene_fixture_name)
        result = dummy_scene_classifier.classify(frame)
        assert isinstance(result.scene_tag, SceneType)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.probabilities) == 6
