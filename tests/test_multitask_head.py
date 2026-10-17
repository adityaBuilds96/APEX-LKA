"""
tests/test_multitask_head.py
============================
Unit tests for APEXHydraNet multi-task head and combined multi-task loss.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.training.multitask_head import (
    NUM_OBJ_CLASSES,
    TARGET_CLASSES,
    APEXHydraNet,
    DecoupledDetectionHead,
    HydraNetLoss,
    compute_ciou_loss,
)


def test_target_classes_specification():
    """Verify all 9 mandatory object classes are present and ordered."""
    expected = [
        "vehicle", "pedestrian", "cyclist", "traffic_cone",
        "barrier", "animal", "debris", "traffic_sign", "traffic_light"
    ]
    assert TARGET_CLASSES == expected
    assert NUM_OBJ_CLASSES == 9


def test_decoupled_detection_head():
    """Verify decoupled detection head output shapes and value bounds."""
    head = DecoupledDetectionHead(in_channels=128, num_classes=9, hidden_channels=32, stride=8)
    head.eval()

    # Input feature from FPN (B=2, C=128, H=45, W=80)
    feat = torch.randn(2, 128, 45, 80)
    det_preds, cls_logits, reg = head(feat)

    # Grid downsampled by 2: H_g=23, W_g=40 -> N_anchors = 920
    b, n_anchors, channels = det_preds.shape
    assert b == 2
    assert channels == 4 + 9  # 4 box coords + 9 class scores

    # Value bounds check
    boxes = det_preds[:, :, :4]
    scores = det_preds[:, :, 4:]
    assert torch.all(boxes >= 0.0) and torch.all(boxes <= 1.0)
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0)


def test_hydranet_forward_eval():
    """Verify APEXHydraNet forward pass produces both segmentation and detection tensors."""
    model = APEXHydraNet(
        num_seg_classes=4,
        num_obj_classes=9,
        fpn_channels=64,
        pretrained=False,
    )
    model.eval()

    x = torch.randn(1, 3, 180, 320)
    with torch.no_grad():
        seg_logits, det_preds = model(x)

    assert seg_logits.shape == (1, 4, 180, 320)
    assert det_preds.shape[0] == 1
    assert det_preds.shape[2] == 13

    # Test return_seg_only
    with torch.no_grad():
        seg_only = model(x, return_seg_only=True)
    assert seg_only.shape == (1, 4, 180, 320)
    assert isinstance(seg_only, torch.Tensor)


def test_ciou_loss_computation():
    """Verify CIoU loss behavior: perfect overlap = 0.0 loss, non-overlap > 0."""
    boxes1 = torch.tensor([[0.2, 0.2, 0.4, 0.4]])
    boxes2 = torch.tensor([[0.2, 0.2, 0.4, 0.4]])
    loss_identical = compute_ciou_loss(boxes1, boxes2)
    assert torch.isclose(loss_identical, torch.tensor([0.0]), atol=1e-4)

    boxes_disjoint = torch.tensor([[0.7, 0.7, 0.9, 0.9]])
    loss_disjoint = compute_ciou_loss(boxes1, boxes_disjoint)
    assert loss_disjoint.item() > 0.5


def test_hydranet_combined_loss():
    """Verify combined multi-task loss calculation: L_total = L_seg + 0.5*L_cls + 1.0*L_box."""
    criterion = HydraNetLoss(cls_weight=0.5, box_weight=1.0)

    b, c_seg, h, w = 2, 4, 60, 80
    seg_logits = torch.randn(b, c_seg, h, w, requires_grad=True)
    seg_targets = torch.randint(0, c_seg, (b, h, w), dtype=torch.long)

    n_anchors = 100
    det_preds = torch.sigmoid(torch.randn(b, n_anchors, 13, requires_grad=True))

    target_boxes = torch.rand(b, n_anchors, 4)
    target_classes = torch.rand(b, n_anchors, 9)

    loss_dict = criterion(
        seg_logits=seg_logits,
        det_preds=det_preds,
        seg_targets=seg_targets,
        target_boxes=target_boxes,
        target_classes=target_classes,
    )

    assert "total_loss" in loss_dict
    assert "seg_loss" in loss_dict
    assert "cls_loss" in loss_dict
    assert "box_loss" in loss_dict

    total = loss_dict["total_loss"]
    expected_total = loss_dict["seg_loss"] + 0.5 * loss_dict["cls_loss"] + 1.0 * loss_dict["box_loss"]
    assert torch.isclose(total, expected_total, atol=1e-5)
    assert total.item() > 0.0


def test_hydranet_onnx_export():
    """Verify APEXHydraNet is fully trace-compatible with torch.onnx.export()."""
    model = APEXHydraNet(
        num_seg_classes=4,
        num_obj_classes=9,
        fpn_channels=32,
        pretrained=False,
    )
    model.eval()

    dummy_input = torch.randn(1, 3, 180, 320)
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = Path(tmp_dir) / "hydranet_test.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            opset_version=14,
            input_names=["input"],
            output_names=["seg_logits", "det_preds"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "seg_logits": {0: "batch_size"},
                "det_preds": {0: "batch_size"},
            },
        )
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 1000
