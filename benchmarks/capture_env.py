#!/usr/bin/env python3
"""
APEX-Vision Core v4.0 • Environment & Hardware Capture Engine
============================================================
Captures comprehensive system hardware, OS, compiler, Python, GPU, and
Jetson Tegra platform parameters into a reproducible `env.json` artifact.

Operating Rule (Finding F-06 / F-23):
- Strictly sets target_hardware to "TARGET" ONLY if running on NVIDIA Jetson Orin Nano.
- Otherwise, sets target_hardware to "NON-TARGET" (e.g. x86_64 workstation, laptop, CI VM).
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    """Capture current Git repository state, commit SHA, branch, and dirty status."""
    info: Dict[str, Any] = {
        "commit_sha": "unknown",
        "commit_short_sha": "unknown",
        "branch": "unknown",
        "is_dirty": False,
        "commit_timestamp": "unknown",
        "commit_author": "unknown",
        "remote_url": "unknown",
    }
    try:
        # Commit SHA
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            info["commit_sha"] = res.stdout.strip()
            info["commit_short_sha"] = info["commit_sha"][:7]

        # Branch
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            info["branch"] = res.stdout.strip()

        # Dirty status
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            info["is_dirty"] = len(res.stdout.strip()) > 0

        # Author and date
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cI|%an"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and "|" in res.stdout:
            parts = res.stdout.strip().split("|")
            info["commit_timestamp"] = parts[0]
            info["commit_author"] = parts[1]

        # Remote URL
        res = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            info["remote_url"] = res.stdout.strip()

    except Exception as e:
        info["error"] = str(e)

    return info


def get_cpu_info() -> Dict[str, Any]:
    """Capture detailed CPU model, core counts, and architecture."""
    cpu_info: Dict[str, Any] = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "logical_cores": os.cpu_count() or 1,
        "physical_cores": None,
        "model_string": "Unknown CPU",
    }

    try:
        if platform.system() == "Windows":
            # Windows wmic or registry
            cmd = "wmic cpu get name,NumberOfCores,NumberOfLogicalProcessors /format:list"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if "Name=" in line:
                        cpu_info["model_string"] = line.split("Name=")[-1].strip()
                    elif "NumberOfCores=" in line:
                        try:
                            cpu_info["physical_cores"] = int(line.split("=")[-1].strip())
                        except ValueError:
                            pass
        elif platform.system() == "Linux":
            if Path("/proc/cpuinfo").exists():
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_info["model_string"] = line.split(":", 1)[1].strip()
                            break
                        elif "Hardware" in line:
                            cpu_info["model_string"] = line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            if res.returncode == 0:
                cpu_info["model_string"] = res.stdout.strip()
    except Exception:
        pass

    if not cpu_info["model_string"] or cpu_info["model_string"] == "Unknown CPU":
        cpu_info["model_string"] = platform.processor() or "Generic Processor"

    return cpu_info


def get_ram_info() -> Dict[str, Any]:
    """Capture total and available system memory."""
    ram_info: Dict[str, Any] = {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "percent_used": 0.0,
    }
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        ram_info["total_gb"] = round(vm.total / (1024 ** 3), 2)
        ram_info["available_gb"] = round(vm.available / (1024 ** 3), 2)
        ram_info["percent_used"] = float(vm.percent)
    except ImportError:
        try:
            if platform.system() == "Windows":
                cmd = "wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /format:list"
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    tot, free = 0, 0
                    for line in res.stdout.splitlines():
                        if "TotalVisibleMemorySize=" in line:
                            tot = int(line.split("=")[-1].strip())
                        elif "FreePhysicalMemory=" in line:
                            free = int(line.split("=")[-1].strip())
                    if tot > 0:
                        ram_info["total_gb"] = round(tot / (1024 * 1024), 2)
                        ram_info["available_gb"] = round(free / (1024 * 1024), 2)
                        ram_info["percent_used"] = round(((tot - free) / tot) * 100.0, 1)
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    mem: Dict[str, int] = {}
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            val = parts[1].strip().split()[0]
                            mem[parts[0].strip()] = int(val)
                    if "MemTotal" in mem:
                        tot = mem["MemTotal"]
                        free = mem.get("MemAvailable", mem.get("MemFree", 0))
                        ram_info["total_gb"] = round(tot / (1024 * 1024), 2)
                        ram_info["available_gb"] = round(free / (1024 * 1024), 2)
                        ram_info["percent_used"] = round(((tot - free) / tot) * 100.0, 1)
        except Exception:
            pass
    return ram_info


def get_python_packages() -> Dict[str, Any]:
    """Capture key AV and machine learning dependency versions."""
    packages: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "python_compiler": platform.python_compiler(),
        "torch_version": None,
        "torch_cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "tensorrt_version": None,
        "onnxruntime_version": None,
        "opencv_version": None,
        "numpy_version": None,
        "scipy_version": None,
        "torchvision_version": None,
    }

    try:
        import numpy  # type: ignore
        packages["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    try:
        import scipy  # type: ignore
        packages["scipy_version"] = scipy.__version__
    except ImportError:
        pass

    try:
        import cv2  # type: ignore
        packages["opencv_version"] = cv2.__version__
    except ImportError:
        pass

    try:
        import onnxruntime  # type: ignore
        packages["onnxruntime_version"] = onnxruntime.__version__
    except ImportError:
        pass

    try:
        import torch  # type: ignore
        packages["torch_version"] = torch.__version__
        packages["torch_cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            packages["cuda_version"] = str(torch.version.cuda)
            packages["cudnn_version"] = str(torch.backends.cudnn.version())
    except ImportError:
        pass

    try:
        import tensorrt  # type: ignore
        packages["tensorrt_version"] = tensorrt.__version__
    except ImportError:
        packages["tensorrt_version"] = "not_installed"

    try:
        import torchvision  # type: ignore
        packages["torchvision_version"] = torchvision.__version__
    except ImportError:
        pass

    return packages


def get_gpu_info() -> Dict[str, Any]:
    """Capture discrete GPU / integrated Tegra GPU parameters."""
    gpu_info: Dict[str, Any] = {
        "has_cuda_gpu": False,
        "gpu_count": 0,
        "devices": [],
        "driver_version": "N/A",
    }

    # Try PyTorch CUDA inspection
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpu_info["has_cuda_gpu"] = True
            gpu_info["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "device_id": i,
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024 ** 2), 1),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": getattr(props, "multi_processor_count", None),
                })
    except Exception:
        pass

    # Try nvidia-smi for driver version
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            if lines:
                parts = lines[0].split(",")
                gpu_info["driver_version"] = parts[0].strip()
                if not gpu_info["devices"]:
                    for i, l in enumerate(lines):
                        p = l.split(",")
                        gpu_info["devices"].append({
                            "device_id": i,
                            "name": p[1].strip() if len(p) > 1 else "Unknown GPU",
                            "driver_version": parts[0].strip(),
                            "total_memory_raw": p[2].strip() if len(p) > 2 else "N/A",
                        })
                    gpu_info["has_cuda_gpu"] = True
                    gpu_info["gpu_count"] = len(lines)
    except Exception:
        pass

    return gpu_info


def get_jetson_info() -> Dict[str, Any]:
    """
    Inspect Jetson Orin Nano / Tegra SoC parameters.
    Extracts nvpmodel mode, jetson_clocks state, and hardware model string.
    """
    jetson_info: Dict[str, Any] = {
        "is_tegra": False,
        "tegra_model": "None",
        "l4t_release": "None",
        "nvpmodel_mode": "N/A",
        "jetson_clocks_status": "N/A",
    }

    # Check Linux /proc/device-tree/model
    dt_model_path = Path("/proc/device-tree/model")
    if dt_model_path.exists():
        try:
            with open(dt_model_path, "r") as f:
                model_str = f.read().strip("\x00 \n\r")
                jetson_info["tegra_model"] = model_str
                if "Jetson" in model_str or "Tegra" in model_str:
                    jetson_info["is_tegra"] = True
        except Exception:
            pass

    # Check /etc/nv_tegra_release
    tegra_rel_path = Path("/etc/nv_tegra_release")
    if tegra_rel_path.exists():
        jetson_info["is_tegra"] = True
        try:
            with open(tegra_rel_path, "r") as f:
                jetson_info["l4t_release"] = f.readline().strip()
        except Exception:
            pass

    # Query nvpmodel if command exists
    try:
        res = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "NV Power Mode:" in line or "NV Mode:" in line:
                    jetson_info["nvpmodel_mode"] = line.strip()
                    break
    except Exception:
        pass

    # Query jetson_clocks
    try:
        res = subprocess.run(["jetson_clocks", "--show"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            jetson_info["jetson_clocks_status"] = "Active"
    except Exception:
        pass

    return jetson_info


def classify_target_hardware(cpu_info: Dict[str, Any], gpu_info: Dict[str, Any], jetson_info: Dict[str, Any]) -> str:
    """
    Strict Classification:
    Returns 'TARGET' ONLY if running on NVIDIA Jetson Orin Nano.
    Otherwise returns 'NON-TARGET'.
    """
    # Check if Tegra model indicates Jetson Orin Nano
    model_str = jetson_info.get("tegra_model", "").lower()
    if jetson_info.get("is_tegra", False):
        if "orin nano" in model_str or "orin-nano" in model_str or "p3767" in model_str:
            return "TARGET"

    # Also check device name in CUDA devices if available
    for dev in gpu_info.get("devices", []):
        dev_name = dev.get("name", "").lower()
        if "orin nano" in dev_name or "orin-nano" in dev_name:
            return "TARGET"

    return "NON-TARGET"


def capture_environment(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Capture all system, git, hardware, and environment metrics into a single dictionary."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    now = datetime.datetime.now(datetime.timezone.utc)
    utc_timestamp = now.isoformat()

    git_info = get_git_info(repo_root)
    cpu_info = get_cpu_info()
    ram_info = get_ram_info()
    pkg_info = get_python_packages()
    gpu_info = get_gpu_info()
    jetson_info = get_jetson_info()

    target_hw = classify_target_hardware(cpu_info, gpu_info, jetson_info)

    env_data: Dict[str, Any] = {
        "metadata": {
            "schema_version": "4.0.0",
            "captured_utc": utc_timestamp,
            "generator": "benchmarks/capture_env.py",
            "operating_rule": "Finding F-06 / F-23 Zero-Fabrication Hardware Tagging",
        },
        "target_classification": {
            "target_hardware": target_hw,
            "is_target_hardware": bool(target_hw == "TARGET"),
            "target_device_profile": "NVIDIA Jetson Orin Nano 4GB (Sub-15W TDP)",
            "classification_reason": (
                "Verified physical Jetson Orin Nano platform" if target_hw == "TARGET"
                else "Host environment is not an NVIDIA Jetson Orin Nano; execution classified as NON-TARGET"
            ),
        },
        "git": git_info,
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "node_name": platform.node(),
        },
        "cpu": cpu_info,
        "ram": ram_info,
        "python_environment": pkg_info,
        "gpu": gpu_info,
        "jetson_tegra": jetson_info,
    }

    return env_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="APEX-Vision Core v4.0 • Environment & Hardware Capture Tool"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="benchmarks/env.json",
        help="Target output JSON path (default: benchmarks/env.json)",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        default=True,
        help="Print human-readable summary to stdout",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    env_data = capture_environment(repo_root)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_data, f, indent=2)

    if args.print_summary:
        print("=" * 80)
        print(" APEX-VISION CORE v4.0 • ENVIRONMENT CAPTURE COMPLETE")
        print("=" * 80)
        print(f" Output File       : {out_path.resolve()}")
        print(f" Captured UTC      : {env_data['metadata']['captured_utc']}")
        print(f" Hardware Class    : {env_data['target_classification']['target_hardware']}")
        print(f" OS & Platform     : {env_data['operating_system']['platform']}")
        print(f" CPU Model         : {env_data['cpu']['model_string']} ({env_data['cpu']['logical_cores']} cores)")
        print(f" RAM Total         : {env_data['ram']['total_gb']} GB")
        print(f" Python Executable : {env_data['python_environment']['python_executable']}")
        print(f" Python Version    : {env_data['python_environment']['python_version']}")
        print(f" PyTorch Version   : {env_data['python_environment']['torch_version']}")
        print(f" CUDA Available    : {env_data['python_environment']['torch_cuda_available']}")
        print(f" Git Commit SHA    : {env_data['git']['commit_sha']}")
        print(f" Git Branch (Dirty): {env_data['git']['branch']} (dirty={env_data['git']['is_dirty']})")
        print("=" * 80)


if __name__ == "__main__":
    main()
