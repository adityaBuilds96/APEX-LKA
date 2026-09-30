"""
src/utils/crash_logger.py
=========================
Automated forensic crash reporting and system diagnostics for APEX-LKA.

Collects:
- Full exception stack traces
- Hardware & memory snapshots (CPU, RAM, GPU, Disk)
- System and platform environment details
- Recent log entries (last 100 lines)
- Active configuration and model status
- Historical inference metrics
- Exports Markdown and JSON diagnostic dumps into logs/crashes/
"""

from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log = logging.getLogger("apex_crash_logger")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class CrashLogger:
    """
    Forensic diagnostics collector for fatal runtime exceptions.
    """

    def __init__(
        self,
        crash_dir: Optional[Path] = None,
        remote_endpoint: Optional[str] = None,
    ) -> None:
        self.crash_dir = crash_dir or (PROJECT_ROOT / "logs" / "crashes")
        self.crash_dir.mkdir(parents=True, exist_ok=True)
        self.remote_endpoint = remote_endpoint
        self._recent_logs: List[str] = []
        self._inference_history: List[Dict[str, Any]] = []
        self._active_model_info: Dict[str, Any] = {"status": "UNKNOWN"}
        self._active_config: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def update_model_info(self, info: Dict[str, Any]) -> None:
        with self._lock:
            self._active_model_info = dict(info)

    def update_config(self, config_dict: Dict[str, Any]) -> None:
        with self._lock:
            self._active_config = dict(config_dict)

    def record_inference_event(self, event: Dict[str, Any]) -> None:
        with self._lock:
            self._inference_history.append(event)
            if len(self._inference_history) > 100:
                self._inference_history.pop(0)

    def append_log_line(self, line: str) -> None:
        with self._lock:
            self._recent_logs.append(line.rstrip())
            if len(self._recent_logs) > 100:
                self._recent_logs.pop(0)

    def capture_system_snapshot(self) -> Dict[str, Any]:
        """Gather hardware and environment telemetry."""
        snapshot: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "os": f"{platform.system()} {platform.release()} ({platform.version()})",
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=None),
        }

        # RAM telemetry
        try:
            mem = psutil.virtual_memory()
            snapshot["ram_total_gb"] = round(mem.total / (1024**3), 2)
            snapshot["ram_available_gb"] = round(mem.available / (1024**3), 2)
            snapshot["ram_used_percent"] = mem.percent
        except Exception as e:
            snapshot["ram_error"] = str(e)

        # Disk telemetry
        try:
            disk = psutil.disk_usage(str(PROJECT_ROOT))
            snapshot["disk_free_gb"] = round(disk.free / (1024**3), 2)
            snapshot["disk_total_gb"] = round(disk.total / (1024**3), 2)
        except Exception as e:
            snapshot["disk_error"] = str(e)

        # GPU telemetry
        if HAS_TORCH and torch.cuda.is_available():
            try:
                snapshot["gpu_available"] = True
                snapshot["gpu_device_name"] = torch.cuda.get_device_name(0)
                snapshot["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
                snapshot["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 2)
            except Exception as e:
                snapshot["gpu_error"] = str(e)
        else:
            snapshot["gpu_available"] = False

        return snapshot

    def _read_recent_log_file(self, max_lines: int = 100) -> List[str]:
        """Read last N lines from persistent log file if available."""
        main_log = PROJECT_ROOT / "logs" / "apex_lka.log"
        if main_log.exists():
            try:
                with open(main_log, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    return [line.rstrip() for line in lines[-max_lines:]]
            except Exception:
                pass
        return list(self._recent_logs)

    def generate_crash_report(
        self,
        exc: BaseException,
        tb_str: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Tuple[Path, Path]:
        """
        Produce structured JSON and formatted Markdown crash reports.
        """
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        crash_id = f"crash_{ts_str}_{int(time.time() * 1000) % 10000:04d}"

        if tb_str is None:
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        sys_snap = self.capture_system_snapshot()
        recent_logs = self._read_recent_log_file()

        with self._lock:
            model_info = dict(self._active_model_info)
            cfg_info = dict(self._active_config)
            inf_hist = list(self._inference_history[-20:])

        report_data = {
            "crash_id": crash_id,
            "context": context or "Uncaught exception",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": tb_str,
            "system_snapshot": sys_snap,
            "model_info": model_info,
            "config": cfg_info,
            "recent_inference_history": inf_hist,
            "recent_logs": recent_logs,
        }

        # 1. Write JSON Dump
        json_path = self.crash_dir / f"{crash_id}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=str)
        except Exception as e:
            log.error("Failed to write JSON crash report: %s", e)

        # 2. Write Markdown Report
        md_path = self.crash_dir / f"{crash_id}.md"
        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# APEX-LKA Forensic Crash Report: {crash_id}\n\n")
                f.write(f"**Timestamp:** {sys_snap.get('timestamp')}\n")
                f.write(f"**Exception:** `{type(exc).__name__}: {exc}`\n")
                f.write(f"**Context:** {context or 'Uncaught exception'}\n\n")

                f.write("## Stack Trace\n```\n" + tb_str + "\n```\n\n")

                f.write("## System Snapshot\n")
                for k, v in sys_snap.items():
                    f.write(f"- **{k}**: {v}\n")
                f.write("\n")

                f.write("## Model & Config Telemetry\n")
                f.write(f"```json\n{json.dumps(model_info, indent=2)}\n```\n\n")

                f.write("## Recent Log Excerpt\n```text\n")
                for line in recent_logs[-30:]:
                    f.write(line + "\n")
                f.write("```\n")
        except Exception as e:
            log.error("Failed to write Markdown crash report: %s", e)

        # 3. Optional remote reporting
        if self.remote_endpoint:
            self._send_to_remote(report_data)

        log.critical("CRASH REPORT GENERATED: %s", md_path)
        return json_path, md_path

    def _send_to_remote(self, data: Dict[str, Any]) -> None:
        """Post crash report payload to remote monitoring endpoint."""
        try:
            req = urllib.request.Request(
                self.remote_endpoint,  # type: ignore[arg-type]
                data=json.dumps(data, default=str).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                log.info("Crash report dispatched to %s (Status %d)", self.remote_endpoint, resp.status)
        except Exception as e:
            log.warning("Could not dispatch crash report to remote endpoint: %s", e)


_GLOBAL_CRASH_LOGGER: Optional[CrashLogger] = None


def get_crash_logger() -> CrashLogger:
    global _GLOBAL_CRASH_LOGGER
    if _GLOBAL_CRASH_LOGGER is None:
        _GLOBAL_CRASH_LOGGER = CrashLogger()
    return _GLOBAL_CRASH_LOGGER


def install_crash_handler(
    crash_dir: Optional[Path] = None,
    remote_endpoint: Optional[str] = None,
) -> CrashLogger:
    """
    Hook sys.excepthook and threading.excepthook to capture all unhandled exceptions.
    """
    global _GLOBAL_CRASH_LOGGER
    _GLOBAL_CRASH_LOGGER = CrashLogger(crash_dir=crash_dir, remote_endpoint=remote_endpoint)

    def _sys_excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        _GLOBAL_CRASH_LOGGER.generate_crash_report(exc_value, tb_str=tb_str, context="sys.excepthook")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def _threading_excepthook(args):
        if issubclass(args.exc_type, (KeyboardInterrupt, SystemExit)):
            return
        tb_str = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        context = f"Thread: {args.thread.name if args.thread else 'unknown'}"
        _GLOBAL_CRASH_LOGGER.generate_crash_report(args.exc_value, tb_str=tb_str, context=context)

    sys.excepthook = _sys_excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _threading_excepthook

    return _GLOBAL_CRASH_LOGGER
