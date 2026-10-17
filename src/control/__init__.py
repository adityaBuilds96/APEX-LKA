"""
src/control
===========
Autonomous Lateral Kinematics and Steering Control Module:
- Stanley cross-track tracking controller
- Curvature-adaptive Pure Pursuit controller
- ControllerManager: Hysteresis switching, perception attenuation, global rate limiting
"""

from src.control.stanley import StanleyController
from src.control.pure_pursuit import PurePursuitController
from src.control.controller_manager import ControllerManager
from src.control.ekf_tracker import EKFPolynomialTracker

__all__ = [
    "StanleyController",
    "PurePursuitController",
    "ControllerManager",
    "EKFPolynomialTracker",
]
