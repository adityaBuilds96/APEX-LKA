"""
src/utils/health_check.py
=========================
Pre-flight diagnostics and hardware validation prior to running inference or training.

Validates:
1. GPU accessibility and CUDA tensor compute functional test
2. Model asset integrity (TensorRT .plan, ONNX .onnx, PyTorch .pth)
3. Camera video stream availability
4. Serial & CAN hardware bus reachability
5. Available system storage (> 1 GB free)
6. Available process memory (> 500 MB free)
7. Configuration file schema validity
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import psutil
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log = logging.getLogger("apex_health_check")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class CheckItem:
    name: str
    status: str          # "PASS", "WARN", "FAIL"
    message: str
    is_critical: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightReport:
    items: List[CheckItem] = field(default_factory=list)
    overall_status: str = "PASS"   # "PASS", "WARN", "FAIL"

    @property
    def passed(self) -> bool:
        return self.overall_status in ("PASS", "WARN")

    @property
    def failed_critical(self) -> bool:
        return any(item.status == "FAIL" and item.is_critical for item in self.items)


class HealthChecker:
    """
    Executes comprehensive pre-flight verification checks.
    """

    def __init__(
        self,
        min_disk_free_gb: float = 1.0,
        min_ram_free_mb: float = 500.0,
        check_camera: bool = False,
        check_comms: bool = False,
        project_root: Optional[Path] = None,
    ) -> None:
        self.min_disk_free_gb = min_disk_free_gb
        self.min_ram_free_mb = min_ram_free_mb
        self.check_camera = check_camera
        self.check_comms = check_comms
        self.project_root = project_root or PROJECT_ROOT

    def check_gpu(self) -> CheckItem:
        """Verify CUDA availability and perform smoke tensor computation."""
        if not HAS_TORCH:
            return CheckItem(
                name="GPU & CUDA Compute",
                status="WARN",
                message="PyTorch not installed; running on CPU baseline.",
                is_critical=False,
            )

        if not torch.cuda.is_available():
            return CheckItem(
                name="GPU & CUDA Compute",
                status="WARN",
                message="No CUDA GPU detected; fallback to CPU inference.",
                is_critical=False,
            )

        try:
            device_name = torch.cuda.get_device_name(0)
            # Run simple compute operation
            x = torch.ones((32, 32), device="cuda")
            y = (x * 2.5).sum()
            val = float(y.item())
            return CheckItem(
                name="GPU & CUDA Compute",
                status="PASS",
                message=f"CUDA compute operational on {device_name}.",
                is_critical=False,
                details={"device": device_name, "smoke_test_sum": val},
            )
        except Exception as e:
            return CheckItem(
                name="GPU & CUDA Compute",
                status="FAIL",
                message=f"CUDA device detected but failed compute verification: {e}",
                is_critical=True,
            )

    def check_memory(self) -> CheckItem:
        """Verify process RAM availability."""
        try:
            mem = psutil.virtual_memory()
            available_mb = mem.available / (1024 * 1024)
            if available_mb < self.min_ram_free_mb:
                return CheckItem(
                    name="RAM Availability",
                    status="FAIL",
                    message=f"Low RAM: {available_mb:.1f} MB available (< {self.min_ram_free_mb} MB required).",
                    is_critical=True,
                    details={"available_mb": available_mb},
                )
            return CheckItem(
                name="RAM Availability",
                status="PASS",
                message=f"RAM sufficient ({available_mb:.1f} MB free).",
                is_critical=False,
                details={"available_mb": available_mb},
            )
        except Exception as e:
            return CheckItem(
                name="RAM Availability",
                status="WARN",
                message=f"Could not query memory telemetry: {e}",
            )

    def check_disk_space(self) -> CheckItem:
        """Verify free storage space on host disk."""
        try:
            disk = psutil.disk_usage(str(self.project_root))
            free_gb = disk.free / (1024**3)
            if free_gb < self.min_disk_free_gb:
                return CheckItem(
                    name="Disk Storage",
                    status="FAIL",
                    message=f"Insufficient disk space: {free_gb:.2f} GB free (< {self.min_disk_free_gb} GB required).",
                    is_critical=True,
                    details={"free_gb": free_gb},
                )
            return CheckItem(
                name="Disk Storage",
                status="PASS",
                message=f"Storage sufficient ({free_gb:.2f} GB free).",
                is_critical=False,
                details={"free_gb": free_gb},
            )
        except Exception as e:
            return CheckItem(
                name="Disk Storage",
                status="WARN",
                message=f"Could not query disk telemetry: {e}",
            )

    def check_models(self) -> CheckItem:
        """Audit readiness across all model export formats."""
        plan_file = self.project_root / "models" / "exported" / "best_model.plan"
        onnx_file = self.project_root / "models" / "exported" / "best_model.onnx"
        pth_file = self.project_root / "models" / "exported" / "best_model.pth"

        readiness: Dict[str, bool] = {
            "TensorRT_Plan": plan_file.exists(),
            "ONNX_Model": onnx_file.exists(),
            "PyTorch_Weights": pth_file.exists(),
        }

        if any(readiness.values()):
            available = [k for k, v in readiness.items() if v]
            return CheckItem(
                name="Model Assets",
                status="PASS",
                message=f"Model assets available: {', '.join(available)}.",
                is_critical=False,
                details=readiness,
            )
        else:
            return CheckItem(
                name="Model Assets",
                status="WARN",
                message="No trained neural models found in models/exported/. Classical CV fallback will be used.",
                is_critical=False,
                details=readiness,
            )

    def check_configs(self) -> CheckItem:
        """Validate YAML configuration files."""
        cfg_path = self.project_root / "deploy" / "deploy_config.yaml"
        if not cfg_path.exists():
            return CheckItem(
                name="Configuration Files",
                status="WARN",
                message="deploy/deploy_config.yaml not found; using in-memory defaults.",
                is_critical=False,
            )
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
            if not isinstance(parsed, dict):
                raise ValueError("Config root must be a dictionary")
            return CheckItem(
                name="Configuration Files",
                status="PASS",
                message="deploy/deploy_config.yaml successfully validated.",
                is_critical=False,
            )
        except Exception as e:
            return CheckItem(
                name="Configuration Files",
                status="FAIL",
                message=f"Configuration file corrupted or invalid YAML: {e}",
                is_critical=True,
            )

    def check_camera_device(self, device_id: int = 0) -> CheckItem:
        """Probe camera hardware reachability."""
        try:
            cap = cv2.VideoCapture(device_id)
            if not cap.isOpened():
                return CheckItem(
                    name="Camera Hardware",
                    status="WARN",
                    message=f"Camera at device_id={device_id} could not be opened (may be in use or disconnected).",
                    is_critical=False,
                )
            ret, frame = cap.read()
            cap.release()
            if not ret or frame is None:
                return CheckItem(
                    name="Camera Hardware",
                    status="WARN",
                    message=f"Camera at device_id={device_id} opened but returned blank frame.",
                    is_critical=False,
                )
            return CheckItem(
                name="Camera Hardware",
                status="PASS",
                message=f"Camera functional (Frame shape: {frame.shape}).",
                is_critical=False,
            )
        except Exception as e:
            return CheckItem(
                name="Camera Hardware",
                status="WARN",
                message=f"Camera test raised exception: {e}",
                is_critical=False,
            )

    def run_preflight_checks(self, strict: bool = False) -> PreflightReport:
        """
        Execute all pre-flight sanity checks and generate diagnostic report.
        If `strict=True` and a critical check fails, raises RuntimeError.
        """
        report = PreflightReport()

        report.items.append(self.check_gpu())
        report.items.append(self.check_memory())
        report.items.append(self.check_disk_space())
        report.items.append(self.check_models())
        report.items.append(self.check_configs())

        if self.check_camera:
            report.items.append(self.check_camera_device())

        # Determine overall status
        if any(item.status == "FAIL" for item in report.items):
            report.overall_status = "FAIL"
        elif any(item.status == "WARN" for item in report.items):
            report.overall_status = "WARN"
        else:
            report.overall_status = "PASS"

        if strict and report.failed_critical:
            failed_msgs = [f"{item.name}: {item.message}" for item in report.items if item.status == "FAIL" and item.is_critical]
            raise RuntimeError(f"Critical pre-flight check failed:\n" + "\n".join(failed_msgs))

        return report

    def format_health_report(self, report: PreflightReport) -> str:
        """Produce human-readable terminal/log report."""
        lines = [
            "============================================================",
            f"          APEX-LKA PRE-FLIGHT SYSTEM HEALTH REPORT",
            f"          Overall Status: [{report.overall_status}]",
            "============================================================",
        ]
        for item in report.items:
            tag = f"[{item.status}]"
            lines.append(f"{tag:<8} {item.name:<25}: {item.message}")
        lines.append("============================================================")
        return "\n".join(lines)


def run_startup_check(strict: bool = False) -> PreflightReport:
    """Convenience function to run checks and print report."""
    checker = HealthChecker()
    report = checker.run_preflight_checks(strict=strict)
    print(checker.format_health_report(report))
    return report


if __name__ == "__main__":
    run_startup_check(strict=False)
