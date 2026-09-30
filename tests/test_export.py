"""
tests/test_export.py
====================
Pytest suite for universal model export, quantization, and edge deployment:
1. ONNX export produces valid file (Opset 17, dynamic batch, simplification)
2. PyTorch vs ONNX outputs match (numerical tolerance <= 1e-4)
3. FP16 model export and size reduction (~50%)
4. INT8 dynamic quantization produces functional model
5. Deployment package bundles all required edge artifacts and zip archive
6. Benchmark utility profiles hardware without errors
7. ONNXLanePredictor conforms to BaseLanePredictor interface and handles errors gracefully
8. Predictor factory loads ONNX backend via get_predictor("onnx")
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch
import onnx
import onnxruntime as ort

from src.inference.predictor import (
    BaseLanePredictor,
    DetectionStatus,
    LanePrediction,
    ModelStatus,
    get_predictor,
)
from src.inference.preprocessing import PreprocessResult
from src.inference.onnx_predictor import ONNXLanePredictor
from src.training.model import LaneSegNet
from src.training.export import (
    cross_device_benchmark,
    export_fp16,
    export_int8,
    export_to_onnx,
)
from deploy.deployment_package_builder import build_deployment_package


@pytest.fixture(scope="module")
def exported_onnx_model(tmp_path_factory) -> Path:
    """Fixture that exports a LaneSegNet model to a temporary ONNX file."""
    temp_dir = tmp_path_factory.mktemp("export_fixture")
    onnx_file = temp_dir / "test_model.onnx"

    model = LaneSegNet(num_classes=4, pretrained=False).eval()
    res = export_to_onnx(
        model_or_checkpoint=model,
        output_path=onnx_file,
        opset_version=17,
        dynamic_batch=True,
        simplify=True,
    )
    assert Path(res["output_path"]).exists()
    return onnx_file


def test_onnx_export_produces_valid_file(exported_onnx_model: Path):
    """Verify exported ONNX file is structurally valid and non-empty."""
    assert exported_onnx_model.exists()
    size_mb = exported_onnx_model.stat().st_size / (1024 * 1024)
    # MobileNetV3-Small FPN should be between 10MB and 25MB
    assert 5.0 < size_mb < 30.0, f"Unexpected model size: {size_mb:.2f} MB"

    onnx_model = onnx.load(str(exported_onnx_model))
    onnx.checker.check_model(onnx_model)
    assert onnx_model.opset_import[0].version == 17


def test_pytorch_vs_onnx_outputs_match(exported_onnx_model: Path):
    """Verify PyTorch and ONNX Runtime outputs match within 1e-4 numerical tolerance."""
    model = LaneSegNet(num_classes=4, pretrained=False).eval()
    dummy_input = torch.randn(1, 3, 360, 640, dtype=torch.float32)

    # PyTorch reference
    with torch.no_grad():
        torch_out = model(dummy_input, return_aux=False).detach().cpu().numpy()

    # Re-export specifically with this model's weights to guarantee exact weight correspondence
    temp_onnx = exported_onnx_model.parent / "exact_match.onnx"
    res = export_to_onnx(
        model_or_checkpoint=model,
        output_path=temp_onnx,
        verify_tolerance=1e-4,
    )
    assert res["verification_passed"]
    assert res["max_abs_diff"] <= 1e-4, f"Difference too high: {res['max_abs_diff']:.2e}"


def test_fp16_model_export_and_accuracy(exported_onnx_model: Path):
    """Verify FP16 conversion reduces model size by ~50% and runs with ONNX Runtime."""
    fp16_file = exported_onnx_model.parent / "test_model_fp16.onnx"
    res = export_fp16(onnx_path=exported_onnx_model, output_path=fp16_file, keep_io_types=True)

    assert fp16_file.exists()
    assert res["reduction_percent"] > 40.0  # Approx 50% reduction

    # Verify ONNX Runtime can execute FP16 model with float32 inputs
    sess = ort.InferenceSession(str(fp16_file), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 3, 360, 640).astype(np.float32)
    output = sess.run(None, {"input_image": dummy_input})[0]
    assert output.shape == (1, 4, 360, 640)
    assert not np.isnan(output).any()


def test_int8_quantization_produces_valid_model(exported_onnx_model: Path):
    """Verify INT8 dynamic quantization produces a valid model."""
    int8_file = exported_onnx_model.parent / "test_model_int8.onnx"
    res = export_int8(onnx_path=exported_onnx_model, output_path=int8_file)

    assert int8_file.exists()
    assert res["reduction_percent"] > 20.0

    sess = ort.InferenceSession(str(int8_file), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 3, 360, 640).astype(np.float32)
    output = sess.run(None, {"input_image": dummy_input})[0]
    assert output.shape == (1, 4, 360, 640)
    assert not np.isnan(output).any()


def test_deployment_package_contains_all_required_files(exported_onnx_model: Path, tmp_path: Path):
    """Verify deployment package generator creates complete folder and zip archive."""
    deploy_output = tmp_path / "deploy_out"
    zip_path = build_deployment_package(
        onnx_model_path=exported_onnx_model,
        output_base_dir=deploy_output,
        zip_package=True,
    )

    assert zip_path.exists()
    assert zip_path.suffix == ".zip"
    assert zip_path.stat().st_size > 1000

    # Check uncompressed directory contents
    pkg_dirs = [d for d in deploy_output.iterdir() if d.is_dir()]
    assert len(pkg_dirs) == 1
    pkg_dir = pkg_dirs[0]

    required_files = [
        "best_model.onnx",
        "class_legend.json",
        "inference_config.json",
        "standalone_inference.py",
        "README.md",
        "requirements_edge.txt",
    ]
    for rf in required_files:
        p = pkg_dir / rf
        assert p.exists(), f"Missing required file: {rf}"

    # Validate JSON formats
    with open(pkg_dir / "class_legend.json", "r", encoding="utf-8") as f:
        legend = json.load(f)
        assert len(legend["classes"]) == 4

    with open(pkg_dir / "inference_config.json", "r", encoding="utf-8") as f:
        inf_cfg = json.load(f)
        assert inf_cfg["num_classes"] == 4
        assert inf_cfg["input_width"] == 640
        assert inf_cfg["input_height"] == 360


def test_benchmark_utility_runs_without_errors(exported_onnx_model: Path):
    """Verify cross_device_benchmark executes profiling runs and returns metrics."""
    bench_results = cross_device_benchmark(
        onnx_path=exported_onnx_model,
        num_iterations=5,
        warmup=2,
    )
    assert isinstance(bench_results, dict)
    assert "CPUExecutionProvider" in bench_results
    cpu_stats = bench_results["CPUExecutionProvider"]
    assert "fps" in cpu_stats
    assert "p50_ms" in cpu_stats
    assert cpu_stats["fps"] > 0.0


def test_onnx_predictor_matches_base_interface(exported_onnx_model: Path):
    """Verify ONNXLanePredictor matches BaseLanePredictor interface and executes properly."""
    predictor = ONNXLanePredictor(model_path=exported_onnx_model)
    assert isinstance(predictor, BaseLanePredictor)
    assert predictor.model_status == ModelStatus.READY
    assert "ONNX Runtime" in predictor.backend_name

    # Create dummy PreprocessResult
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    dummy_tensor = np.zeros((360, 640, 3), dtype=np.float32)
    prep = PreprocessResult(
        original_bgr=dummy_img,
        original_rgb=dummy_img,
        input_tensor=dummy_tensor,
        original_h=360,
        original_w=640,
        model_h=360,
        model_w=640,
    )

    pred = predictor.predict(prep)
    assert isinstance(pred, LanePrediction)
    assert pred.model_status == ModelStatus.READY
    assert pred.left_mask is not None
    assert pred.right_mask is not None
    assert pred.road_mask is not None
    assert pred.left_mask.shape == (360, 640)
    assert pred.right_mask.shape == (360, 640)
    assert pred.road_mask.shape == (360, 640)
    assert pred.inference_ms > 0.0

    # Test error resilience: predict() must never raise
    corrupted_prep = PreprocessResult(input_tensor=None)
    err_pred = predictor.predict(corrupted_prep)
    assert err_pred.status == DetectionStatus.INFERENCE_ERROR
    assert err_pred.error_message is not None


def test_get_predictor_onnx_backend(exported_onnx_model: Path):
    """Verify get_predictor(backend='onnx') instantiates ONNXLanePredictor."""
    pred = get_predictor(backend="onnx")
    assert isinstance(pred, BaseLanePredictor)
    assert isinstance(pred, ONNXLanePredictor)
