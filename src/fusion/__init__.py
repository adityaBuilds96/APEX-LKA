"""
src/fusion/__init__.py
======================
Phase 5 Section 4: Sensor Fusion, State Estimation, VIO & Kinematic Lane Retention.
"""

from src.fusion.eskf import (
    ErrorStateKalmanFilter15DOF,
    ESKFConfig,
    VehiclePose,
    LocalizationStatus,
)
from src.fusion.vio_engine import (
    VisualInertialOdometryEngine,
    EgoMotionMeasurement,
    VIOConfig,
)
from src.fusion.sensor_interface import (
    SensorInterfaceManager,
    IMUReading,
    GNSSReading,
    WheelEncoderReading,
    VehicleKinematics,
)
from src.fusion.lane_retention import (
    KinematicLaneRetainer,
    RetainedLaneResult,
)

__all__ = [
    "ErrorStateKalmanFilter15DOF",
    "ESKFConfig",
    "VehiclePose",
    "LocalizationStatus",
    "VisualInertialOdometryEngine",
    "EgoMotionMeasurement",
    "VIOConfig",
    "SensorInterfaceManager",
    "IMUReading",
    "GNSSReading",
    "WheelEncoderReading",
    "VehicleKinematics",
    "KinematicLaneRetainer",
    "RetainedLaneResult",
]
