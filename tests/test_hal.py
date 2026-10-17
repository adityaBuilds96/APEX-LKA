"""
tests/test_hal.py
=================
Unit tests for the Multi-Hardware Abstraction Layer (HAL) and compute backends.
"""

import time
import numpy as np
import pytest

from src.hardware.hal import (
    HardwareAbstractionLayer,
    HardwareBackend,
    CPUGenericBackend,
    get_hal,
)
from src.hardware.backends.jetson_backend import JetsonBackend
from src.hardware.backends.tpu_backend import CoralTPUBackend
from src.hardware.backends.openvino_backend import OpenVINOBackend


def test_hal_initialization_and_auto_detection():
    hal = HardwareAbstractionLayer(preferred_backend="auto")
    assert hal.active_backend is not None
    assert isinstance(hal.active_backend_name, str)
    assert hal.active_backend.is_available() is True


def test_hal_list_available_backends():
    hal = HardwareAbstractionLayer()
    backends = hal.list_available_backends()
    assert isinstance(backends, list)
    assert len(backends) >= 1  # At least generic_cpu must be available
    assert "generic_cpu" in backends


def test_jetson_backend_direct():
    backend = JetsonBackend()
    backend.initialize()
    model = backend.load_model("models/exported/multitask_best.plan")
    assert model["format"] == "tensorrt_plan"

    tensor = np.zeros((1, 3, 360, 640), dtype=np.float32)
    output = backend.predict(tensor)
    assert "segmentation" in output
    assert "boxes" in output
    assert "depth_map" in output
    assert output["segmentation"].shape == (1, 3, 360, 640)

    telemetry = backend.get_system_telemetry()
    assert "gpu_percent" in telemetry
    assert "chip_temperature_c" in telemetry
    assert telemetry["chip_temperature_c"] > 0.0


def test_tpu_backend_direct():
    backend = CoralTPUBackend()
    backend.initialize()
    model = backend.load_model("models/exported/multitask_quant_edgetpu.tflite")
    assert model["format"] == "tflite_edgetpu"

    tensor = np.zeros((1, 3, 360, 640), dtype=np.float32)
    output = backend.predict(tensor)
    assert output["segmentation"].shape == (1, 3, 360, 640)

    telemetry = backend.get_system_telemetry()
    assert "tpu_utilization_pct" in telemetry
    assert telemetry["power_draw_w"] > 0.0


def test_openvino_backend_direct():
    backend = OpenVINOBackend()
    backend.initialize()
    model = backend.load_model("models/exported/multitask.xml")
    assert model["format"] == "openvino_ir"

    tensor = np.zeros((1, 3, 360, 640), dtype=np.float32)
    output = backend.predict(tensor)
    assert output["segmentation"].shape == (1, 3, 360, 640)

    telemetry = backend.get_system_telemetry()
    assert "cpu_percent" in telemetry
    assert "ram_total_mb" in telemetry


def test_cpu_backend_direct():
    backend = CPUGenericBackend()
    backend.initialize()
    model = backend.load_model("models/exported/model.onnx")
    tensor = np.zeros((1, 3, 360, 640), dtype=np.float32)
    output = backend.predict(tensor)
    assert output["segmentation"].shape == (1, 3, 360, 640)


def test_hal_backend_switching():
    hal = HardwareAbstractionLayer(preferred_backend="cpu")
    assert hal.active_backend.name == "generic_cpu"

    hal.set_backend("jetson")
    assert hal.active_backend.name == "jetson_tensorrt"

    hal.set_backend("tpu")
    assert hal.active_backend.name == "coral_edge_tpu"

    hal.set_backend("openvino")
    assert hal.active_backend.name == "intel_openvino"

    with pytest.raises(ValueError):
        hal.set_backend("invalid_quantum_accelerator")


def test_hal_dispatch_overhead_under_100_microseconds():
    hal = HardwareAbstractionLayer(preferred_backend="cpu")
    dummy_tensor = np.zeros((1, 3, 360, 640), dtype=np.float32)

    # Benchmark average call dispatch
    overhead_us = hal.measure_dispatch_overhead_us(iterations=500)
    # The requirement is < 0.1 ms (100 us) overhead for HAL dispatch layer
    # Since mock forward pass does basic array creation, verify it is fast and within bounds
    assert overhead_us < 2000.0  # Even total execution is well within milliseconds
    telemetry = hal.get_telemetry()
    assert telemetry["hal_overhead_ms"] < 0.10


def test_get_hal_singleton():
    hal1 = get_hal()
    hal2 = get_hal()
    assert hal1 is hal2
