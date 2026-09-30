"""
src/training/export.py
======================
Universal Model Export & Optimization Engine for APEX-LKA:
- PyTorch -> ONNX (Opset 17, dynamic batch, onnx-simplifier)
- Numerical validation: PyTorch vs ONNX output verification (tolerance <= 1e-4)
- FP16 Half-Precision quantization for edge devices
- INT8 Post-Training Quantization for extreme edge constraints
- TensorRT engine generation (.plan) optimized for NVIDIA Jetson (512MB workspace)
- Cross-device execution provider benchmarking (CPU, CUDA, TensorRT, OpenVINO)
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Optional dependencies
try:
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False

try:
    from onnxconverter_common import float16
    HAS_FLOAT16 = True
except ImportError:
    HAS_FLOAT16 = False

try:
    import onnxruntime.quantization as ort_quant
    HAS_ORT_QUANT = True
except ImportError:
    HAS_ORT_QUANT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg
from src.training.metrics import NUM_CLASSES
from src.training.model import LaneSegNet

log = logging.getLogger("apex_export")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = Console()


def find_default_checkpoint() -> Optional[Path]:
    """Locate the best available PyTorch checkpoint."""
    best_exported = PATHS.exported / "best_model.pth"
    if best_exported.exists():
        return best_exported

    last_ckpt = PATHS.checkpoints / "last_checkpoint.pt"
    if last_ckpt.exists():
        return last_ckpt

    ckpts = sorted(PATHS.checkpoints.glob("*.pth")) + sorted(PATHS.checkpoints.glob("*.pt"))
    if ckpts:
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ckpts[0]

    return None


def load_pytorch_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    num_classes: int = NUM_CLASSES,
    device: str = "cpu",
) -> LaneSegNet:
    """Instantiate and load LaneSegNet weights for export."""
    model = LaneSegNet(num_classes=num_classes, pretrained=False)
    # Disable auxiliary training head during export
    model.with_aux_head = False

    ckpt_path = Path(checkpoint_path) if checkpoint_path else find_default_checkpoint()
    if ckpt_path and ckpt_path.exists():
        log.info("Loading PyTorch weights from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "ema_model" in state:
                log.info("Loaded Exponential Moving Average (EMA) weights.")
                model.load_state_dict(state["ema_model"])
            elif "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
        else:
            model.load_state_dict(state)
    else:
        log.warning("No checkpoint found; exporting LaneSegNet with initialized weights.")

    model.to(device)
    model.eval()
    return model


def export_to_onnx(
    model_or_checkpoint: Optional[Union[torch.nn.Module, str, Path]] = None,
    output_path: Union[str, Path] = "models/exported/best_model.onnx",
    opset_version: int = 17,
    input_shape: Tuple[int, int, int, int] = (1, 3, 360, 640),
    dynamic_batch: bool = True,
    simplify: bool = True,
    verify_tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """
    Export PyTorch model to ONNX format with Opset 17, dynamic batch size,
    simplification, and numerical verification against PyTorch outputs.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Resolve PyTorch Model
    if isinstance(model_or_checkpoint, torch.nn.Module):
        model = model_or_checkpoint
        model.eval()
    else:
        model = load_pytorch_model(model_or_checkpoint, device="cpu")

    dummy_input = torch.randn(*input_shape, dtype=torch.float32)

    # 2. Compute PyTorch reference output
    with torch.no_grad():
        torch_output = model(dummy_input, return_aux=False).detach().cpu().numpy()

    # 3. Dynamic axes setup
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input_image": {0: "batch_size"},
            "segmentation_mask": {0: "batch_size"},
        }

    # 4. Perform ONNX export
    log.info("Exporting LaneSegNet to ONNX (Opset %d)...", opset_version)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["segmentation_mask"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    # 5. Check ONNX validity
    onnx_model = onnx.load(str(out_file))
    onnx.checker.check_model(onnx_model)
    log.info("ONNX model structure check passed.")

    # 6. Simplify with onnx-simplifier if available
    is_simplified = False
    if simplify and HAS_ONNXSIM:
        try:
            log.info("Running onnx-simplifier optimization...")
            simplified_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(simplified_model, str(out_file))
                is_simplified = True
                log.info("Model successfully simplified with onnxsim.")
            else:
                log.warning("onnx-simplifier validation check failed; keeping standard export.")
        except Exception as e:
            log.warning("onnx-simplifier encountered an issue (%s); continuing with standard export.", e)

    # 7. Numerical Verification with ONNX Runtime
    ort_session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
    ort_inputs = {"input_image": dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    max_abs_diff = float(np.max(np.abs(torch_output - ort_output)))
    mean_abs_diff = float(np.mean(np.abs(torch_output - ort_output)))
    matches = max_abs_diff <= verify_tolerance

    file_size_mb = out_file.stat().st_size / (1024 * 1024)

    log.info("Verification Complete:")
    log.info("  File Size: %.2f MB", file_size_mb)
    log.info("  Max Absolute Difference: %.2e (Tolerance: %.2e)", max_abs_diff, verify_tolerance)
    log.info("  Match Status: %s", "EXACT MATCH (PASS)" if matches else "WARNING (DIFF > TOLERANCE)")

    return {
        "output_path": str(out_file),
        "file_size_mb": round(file_size_mb, 2),
        "is_simplified": is_simplified,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "verification_passed": matches,
        "opset_version": opset_version,
    }


def export_fp16(
    onnx_path: Union[str, Path] = "models/exported/best_model.onnx",
    output_path: Optional[Union[str, Path]] = None,
    keep_io_types: bool = True,
) -> Dict[str, Any]:
    """
    Convert ONNX model to FP16 half precision.
    keep_io_types=True retains float32 inputs/outputs for plug-and-play inference.
    """
    src_path = Path(onnx_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {src_path}")

    if output_path is None:
        dst_path = src_path.parent / f"{src_path.stem}_fp16{src_path.suffix}"
    else:
        dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not HAS_FLOAT16:
        raise ImportError("onnxconverter-common is required for FP16 export. Run: pip install onnxconverter-common")

    log.info("Converting %s to FP16...", src_path.name)
    model = onnx.load(str(src_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
    onnx.save(model_fp16, str(dst_path))

    orig_size = src_path.stat().st_size / (1024 * 1024)
    fp16_size = dst_path.stat().st_size / (1024 * 1024)
    reduction_pct = (1.0 - (fp16_size / max(1e-5, orig_size))) * 100.0

    # Verify loading with ORT
    sess = ort.InferenceSession(str(dst_path), providers=["CPUExecutionProvider"])
    dummy_in = np.random.randn(1, 3, 360, 640).astype(np.float32)
    _ = sess.run(None, {"input_image": dummy_in})[0]

    log.info("FP16 Conversion complete:")
    log.info("  FP32 Size: %.2f MB -> FP16 Size: %.2f MB (%.1f%% reduction)", orig_size, fp16_size, reduction_pct)
    return {
        "output_path": str(dst_path),
        "original_size_mb": round(orig_size, 2),
        "fp16_size_mb": round(fp16_size, 2),
        "reduction_percent": round(reduction_pct, 1),
    }


def export_int8(
    onnx_path: Union[str, Path] = "models/exported/best_model.onnx",
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Apply INT8 post-training dynamic quantization for ultra-constrained edge deployment.
    """
    src_path = Path(onnx_path)
    if not src_path.exists():
        raise FileNotFoundError(f"Source ONNX model not found: {src_path}")

    if output_path is None:
        dst_path = src_path.parent / f"{src_path.stem}_int8{src_path.suffix}"
    else:
        dst_path = Path(output_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not HAS_ORT_QUANT:
        raise ImportError("onnxruntime.quantization is required for INT8 export.")

    log.info("Applying INT8 dynamic quantization to %s...", src_path.name)
    ort_quant.quantize_dynamic(
        model_input=str(src_path),
        model_output=str(dst_path),
        weight_type=ort_quant.QuantType.QUInt8,
    )

    orig_size = src_path.stat().st_size / (1024 * 1024)
    int8_size = dst_path.stat().st_size / (1024 * 1024)
    reduction_pct = (1.0 - (int8_size / max(1e-5, orig_size))) * 100.0

    # Verify loading with ORT
    sess = ort.InferenceSession(str(dst_path), providers=["CPUExecutionProvider"])
    dummy_in = np.random.randn(1, 3, 360, 640).astype(np.float32)
    _ = sess.run(None, {"input_image": dummy_in})[0]

    log.info("INT8 Quantization complete:")
    log.info("  FP32 Size: %.2f MB -> INT8 Size: %.2f MB (%.1f%% reduction)", orig_size, int8_size, reduction_pct)
    return {
        "output_path": str(dst_path),
        "original_size_mb": round(orig_size, 2),
        "int8_size_mb": round(int8_size, 2),
        "reduction_percent": round(reduction_pct, 1),
    }


def build_tensorrt_engine(
    onnx_path: Union[str, Path] = "models/exported/best_model.onnx",
    output_plan_path: Union[str, Path] = "models/exported/best_model.plan",
    workspace_mb: int = 512,
    fp16: bool = True,
) -> Optional[Path]:
    """
    Build TensorRT execution engine (.plan) optimized for NVIDIA Jetson Orin Nano:
    - Fixed resolution 640x360
    - Workspace budget: 512MB (constrained for 4GB total RAM)
    - FP16 precision enabled
    """
    src_path = Path(onnx_path)
    dst_path = Path(output_plan_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        raise FileNotFoundError(f"ONNX model for TensorRT build not found: {src_path}")

    # Check if tensorrt python module is installed
    try:
        import tensorrt as trt
    except ImportError:
        # Check if trtexec CLI utility is available
        trtexec_bin = shutil.which("trtexec")
        if trtexec_bin:
            log.info("Compiling TensorRT engine via trtexec CLI...")
            cmd = [
                trtexec_bin,
                f"--onnx={src_path}",
                f"--saveEngine={dst_path}",
                f"--memPoolSize=workspace:{workspace_mb}",
            ]
            if fp16:
                cmd.append("--fp16")
            import subprocess
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                log.info("TensorRT engine created: %s", dst_path)
                return dst_path
            else:
                log.error("trtexec failed: %s", res.stderr)
                return None

        log.warning(
            "TensorRT SDK is not installed in the current environment.\n"
            "To build the engine on NVIDIA Jetson, run:\n"
            "  trtexec --onnx=%s --saveEngine=%s --fp16 --memPoolSize=workspace:%d",
            src_path, dst_path, workspace_mb
        )
        return None

    # Native TensorRT Python compilation
    log.info("Building TensorRT engine natively (Workspace: %dMB, FP16: %s)...", workspace_mb, fp16)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(src_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                log.error("TensorRT parser error: %s", parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024 * 1024))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        log.error("Failed to build TensorRT serialized network.")
        return None

    with open(dst_path, "wb") as f:
        f.write(serialized_engine)

    log.info("Saved TensorRT engine: %s (%.2f MB)", dst_path, dst_path.stat().st_size / (1024 * 1024))
    return dst_path


def cross_device_benchmark(
    onnx_path: Optional[Union[str, Path]] = None,
    num_iterations: int = 50,
    warmup: int = 10,
    input_shape: Tuple[int, int, int, int] = (1, 3, 360, 640),
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark ONNX model throughput and latency across all available execution providers.
    """
    if onnx_path is None:
        onnx_file = PATHS.exported / "best_model.onnx"
        if not onnx_file.exists():
            # Export a default model first
            export_to_onnx(output_path=onnx_file)
    else:
        onnx_file = Path(onnx_path)

    if not onnx_file.exists():
        raise FileNotFoundError(f"Model file not found for benchmark: {onnx_file}")

    file_size_mb = onnx_file.stat().st_size / (1024 * 1024)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    available_providers = ort.get_available_providers()
    log.info("Available Execution Providers: %s", available_providers)

    results: Dict[str, Dict[str, Any]] = {}
    candidate_providers = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]

    for prov in candidate_providers:
        if prov not in available_providers:
            continue

        try:
            # Measure initialization / load time
            t_load_0 = time.perf_counter()
            sess = ort.InferenceSession(str(onnx_file), providers=[prov])
            t_load_ms = (time.perf_counter() - t_load_0) * 1000.0

            input_name = sess.get_inputs()[0].name
            feed_dict = {input_name: dummy_input}

            # Warmup runs
            for _ in range(warmup):
                _ = sess.run(None, feed_dict)

            # Benchmark runs
            latencies: List[float] = []
            for _ in range(num_iterations):
                t0 = time.perf_counter()
                _ = sess.run(None, feed_dict)
                latencies.append((time.perf_counter() - t0) * 1000.0)

            lat_arr = np.array(latencies, dtype=np.float64)
            mean_ms = float(np.mean(lat_arr))
            p50_ms = float(np.percentile(lat_arr, 50))
            p95_ms = float(np.percentile(lat_arr, 95))
            p99_ms = float(np.percentile(lat_arr, 99))
            fps = 1000.0 / max(0.1, mean_ms)

            results[prov] = {
                "provider": prov,
                "model_size_mb": round(file_size_mb, 2),
                "load_time_ms": round(t_load_ms, 2),
                "mean_ms": round(mean_ms, 2),
                "p50_ms": round(p50_ms, 2),
                "p95_ms": round(p95_ms, 2),
                "p99_ms": round(p99_ms, 2),
                "fps": round(fps, 1),
            }
        except Exception as e:
            log.warning("Could not benchmark provider %s: %s", prov, e)

    # Print comparison table
    tbl = Table(title=f"APEX-LKA Inference Benchmark ({onnx_file.name}, {file_size_mb:.1f}MB)", show_lines=True)
    tbl.add_column("Execution Provider", style="cyan")
    tbl.add_column("Load Time", justify="right")
    tbl.add_column("Latency (p50)", justify="right")
    tbl.add_column("Latency (p95)", justify="right")
    tbl.add_column("Throughput (FPS)", justify="right", style="bold green")

    for prov, stats in results.items():
        tbl.add_row(
            prov,
            f"{stats['load_time_ms']:.1f} ms",
            f"{stats['p50_ms']:.2f} ms",
            f"{stats['p95_ms']:.2f} ms",
            f"{stats['fps']:.1f} FPS",
        )
    console.print(tbl)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX-LKA Universal Model Export Engine")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="models/exported/best_model.onnx", help="Target ONNX export path")
    parser.add_argument("--fp16", action="store_true", help="Export to FP16 half precision")
    parser.add_argument("--int8", action="store_true", help="Export to INT8 dynamic quantization")
    parser.add_argument("--tensorrt", action="store_true", help="Build TensorRT engine (.plan)")
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark across providers")
    parser.add_argument("--package", action="store_true", help="Build complete edge deployment package")
    args = parser.parse_args()

    # 1. Base FP32 ONNX Export
    res = export_to_onnx(model_or_checkpoint=args.checkpoint, output_path=args.output)
    onnx_file = Path(res["output_path"])

    # 2. FP16 export if requested
    if args.fp16:
        export_fp16(onnx_path=onnx_file)

    # 3. INT8 export if requested
    if args.int8:
        export_int8(onnx_path=onnx_file)

    # 4. TensorRT build if requested
    if args.tensorrt:
        plan_out = onnx_file.parent / f"{onnx_file.stem}.plan"
        build_tensorrt_engine(onnx_path=onnx_file, output_plan_path=plan_out)

    # 5. Deployment package if requested
    if args.package:
        from deploy.deployment_package_builder import build_deployment_package
        build_deployment_package(onnx_model_path=onnx_file)

    # 6. Benchmark if requested
    if args.benchmark:
        cross_device_benchmark(onnx_path=onnx_file)


if __name__ == "__main__":
    main()
