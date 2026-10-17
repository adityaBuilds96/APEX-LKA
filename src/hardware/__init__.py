"""
APEX LKA — Universal Hardware Abstraction & Network Adapter Package
====================================================================
Provides multi-hardware HAL (Jetson TensorRT, Coral Edge TPU, Intel OpenVINO, CPU/GPU)
and multi-protocol networking (CAN, CAN-FD, Serial JSON, ROS 2 DDS).
"""

from src.hardware.hal import (
    HardwareAbstractionLayer,
    HardwareBackend,
    get_hal,
)
from src.hardware.net_adapter import (
    NetworkAdapterManager,
    NetworkProtocol,
)
from src.hardware.can_manager import (
    CANManager,
)
from src.hardware.ros2_publisher import (
    APEXPerceptionNode,
)

__all__ = [
    "HardwareAbstractionLayer",
    "HardwareBackend",
    "get_hal",
    "NetworkAdapterManager",
    "NetworkProtocol",
    "APEXPerceptionNode",
    "CANManager",
]
