"""
src/scene/__init__.py
=====================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Operational Scene Understanding & Dynamic Profile Tuning Subsystem.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.
"""

from src.scene.scene_profiles import (
    SceneType,
    SceneTuningProfile,
    ClassicalCVThresholds,
    LightingCompensationProfile,
    FusionAndTrackingProfile,
    VirtualGuardrailProfile,
    AdaptiveROIProfile,
    SceneProfileRegistry,
    DEFAULT_SCENE_PROFILES,
    get_scene_profile,
    get_scene_profile_registry,
)
from src.scene.scene_classifier import (
    SceneClassificationResult,
    MobileNetV3SceneClassifier,
    HeuristicSceneClassifier,
    SceneClassifier,
    AsyncSceneClassifierRunner,
    get_scene_classifier,
    render_scene_hud_overlay,
    SceneClassificationDataset,
    SceneMetricsCalculator,
    SceneEvaluationReport,
)
from src.scene.scene_adaptive_tuner import (
    DynamicTuningSnapshot,
    TransitionEventRecord,
    SlewRateSmoother,
    SceneAdaptiveTuner,
    ScheduledStanleyGains,
    StanleyGainScheduler,
    AdaptiveDegradationInterlock,
    ParameterDeviationMonitor,
    TuningDiagnosticVisualizer,
    get_scene_adaptive_tuner,
    get_stanley_gain_scheduler,
    get_adaptive_degradation_interlock,
    get_parameter_deviation_monitor,
)

__all__ = [
    "SceneType",
    "SceneTuningProfile",
    "ClassicalCVThresholds",
    "LightingCompensationProfile",
    "FusionAndTrackingProfile",
    "VirtualGuardrailProfile",
    "AdaptiveROIProfile",
    "SceneProfileRegistry",
    "DEFAULT_SCENE_PROFILES",
    "get_scene_profile",
    "get_scene_profile_registry",
    "SceneClassificationResult",
    "MobileNetV3SceneClassifier",
    "HeuristicSceneClassifier",
    "SceneClassifier",
    "AsyncSceneClassifierRunner",
    "get_scene_classifier",
    "render_scene_hud_overlay",
    "SceneClassificationDataset",
    "SceneMetricsCalculator",
    "SceneEvaluationReport",
    "DynamicTuningSnapshot",
    "TransitionEventRecord",
    "SlewRateSmoother",
    "SceneAdaptiveTuner",
    "ScheduledStanleyGains",
    "StanleyGainScheduler",
    "AdaptiveDegradationInterlock",
    "ParameterDeviationMonitor",
    "TuningDiagnosticVisualizer",
    "get_scene_adaptive_tuner",
    "get_stanley_gain_scheduler",
    "get_adaptive_degradation_interlock",
    "get_parameter_deviation_monitor",
]
