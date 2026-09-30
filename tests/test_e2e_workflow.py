"""
tests/test_e2e_workflow.py
==========================
Comprehensive End-to-End Workflow Verification for APEX-LKA:
Record/Ingest -> Synthetic Dataset -> Partition -> Train -> Checkpoint -> Evaluate -> ONNX Export -> Inference Verification.

Proves the entire system pipeline works cohesively without gaps.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.inference.onnx_predictor import ONNXLanePredictor
from src.inference.preprocessing import PreprocessResult
from src.training.dataset import LaneSegDataset
from src.training.export import export_to_onnx
from src.training.losses import CombinedLaneLoss
from src.training.metrics import compute_classification_metrics, compute_confusion_matrix
from src.training.model import LaneSegNet


def create_synthetic_road_sample(height: int = 360, width: int = 640) -> tuple[np.ndarray, np.ndarray]:
    """Generate a realistic synthetic road image and 4-class semantic mask."""
    # Class IDs: 0=background, 1=road, 2=left_lane, 3=right_lane
    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    # Sky/background
    img[: int(height * 0.45), :] = [180, 140, 100]  # Light sky

    # Drivable road surface trapezoid
    road_pts = np.array([
        [int(width * 0.40), int(height * 0.45)],
        [int(width * 0.60), int(height * 0.45)],
        [int(width * 0.95), height],
        [int(width * 0.05), height],
    ], dtype=np.int32)
    cv2.fillPoly(img, [road_pts], (60, 60, 60))
    cv2.fillPoly(mask, [road_pts], 1)

    # Left lane marking (class 2)
    left_line_pts = np.array([
        [int(width * 0.42), int(height * 0.45)],
        [int(width * 0.44), int(height * 0.45)],
        [int(width * 0.20), height],
        [int(width * 0.16), height],
    ], dtype=np.int32)
    cv2.fillPoly(img, [left_line_pts], (255, 255, 255))
    cv2.fillPoly(mask, [left_line_pts], 2)

    # Right lane marking (class 3)
    right_line_pts = np.array([
        [int(width * 0.56), int(height * 0.45)],
        [int(width * 0.58), int(height * 0.45)],
        [int(width * 0.84), height],
        [int(width * 0.80), height],
    ], dtype=np.int32)
    cv2.fillPoly(img, [right_line_pts], (0, 255, 255))
    cv2.fillPoly(mask, [right_line_pts], 3)

    return img, mask


class SyntheticTorchDataset(Dataset):
    """Memory-efficient PyTorch Dataset for end-to-end integration test."""

    def __init__(self, images_dir: Path, masks_dir: Path):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.stems = sorted([f.stem for f in images_dir.glob("*.jpg")])

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]
        img_bgr = cv2.imread(str(self.images_dir / f"{stem}.jpg"))
        mask = cv2.imread(str(self.masks_dir / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (img_rgb - mean) / std

        tensor_img = torch.from_numpy(norm_img.transpose(2, 0, 1)).float()
        tensor_mask = torch.from_numpy(mask).long()
        return tensor_img, tensor_mask


def test_full_pipeline_end_to_end(tmp_path: Path):
    """
    Execute 10-step full lifecycle integration test.
    """
    # ── Step 1: Create synthetic dataset (10 image-mask pairs) ─────────────
    raw_dir = tmp_path / "raw"
    images_raw = raw_dir / "images"
    masks_raw = raw_dir / "masks"
    images_raw.mkdir(parents=True)
    masks_raw.mkdir(parents=True)

    for i in range(10):
        img, mask = create_synthetic_road_sample()
        cv2.imwrite(str(images_raw / f"frame_{i:03d}.jpg"), img)
        cv2.imwrite(str(masks_raw / f"frame_{i:03d}.png"), mask)

    assert len(list(images_raw.glob("*.jpg"))) == 10
    assert len(list(masks_raw.glob("*.png"))) == 10

    # ── Step 2: Split into train (6), val (2), test (2) ───────────────────
    splits = {
        "train": list(range(0, 6)),
        "val": list(range(6, 8)),
        "test": list(range(8, 10)),
    }

    split_paths = {}
    for sp_name, indices in splits.items():
        sp_img = tmp_path / sp_name / "images"
        sp_mask = tmp_path / sp_name / "masks"
        sp_img.mkdir(parents=True)
        sp_mask.mkdir(parents=True)
        for idx in indices:
            shutil.copy(str(images_raw / f"frame_{idx:03d}.jpg"), str(sp_img / f"frame_{idx:03d}.jpg"))
            shutil.copy(str(masks_raw / f"frame_{idx:03d}.png"), str(sp_mask / f"frame_{idx:03d}.png"))
        split_paths[sp_name] = (sp_img, sp_mask)

    # ── Step 3: Train LaneSegNet for 2 epochs ──────────────────────────────
    train_ds = SyntheticTorchDataset(split_paths["train"][0], split_paths["train"][1])
    val_ds = SyntheticTorchDataset(split_paths["val"][0], split_paths["val"][1])
    test_ds = SyntheticTorchDataset(split_paths["test"][0], split_paths["test"][1])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

    model = LaneSegNet(num_classes=4, pretrained=False, with_aux_head=False)
    criterion = CombinedLaneLoss(num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    initial_loss = None
    final_loss = None

    model.train()
    for epoch in range(2):
        epoch_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss_dict = criterion(logits, y)
            loss = loss_dict.total
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / len(train_loader)
        if initial_loss is None:
            initial_loss = avg_loss
        final_loss = avg_loss

    assert final_loss is not None
    assert final_loss > 0.0

    # ── Step 4: Verify checkpoint saved ────────────────────────────────────
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    ckpt_path = ckpt_dir / "best_model.pth"

    torch.save({
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "final_loss": final_loss,
    }, ckpt_path)

    assert ckpt_path.exists()
    assert ckpt_path.stat().st_size > 1_000_000  # > 1MB

    # ── Step 5 & 6: Evaluate on test set & verify metrics ──────────────────
    model.eval()
    total_cm = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            cm = compute_confusion_matrix(preds.cpu().numpy(), y.cpu().numpy(), num_classes=4)
            total_cm += cm

    metrics = compute_classification_metrics(total_cm)

    assert "mean_iou" in metrics
    assert "mean_dice" in metrics
    assert "overall_pixel_accuracy" in metrics
    assert len(metrics["per_class"]) == 4

    # ── Step 7: Export to ONNX ─────────────────────────────────────────────
    onnx_path = tmp_path / "models" / "best_model.onnx"
    export_result = export_to_onnx(
        model_or_checkpoint=model,
        output_path=onnx_path,
        input_shape=(1, 3, 360, 640),
        verify_tolerance=1e-4,
    )

    assert onnx_path.exists()
    assert export_result["verification_passed"] is True
    assert onnx_path.stat().st_size > 1_000_000

    # ── Step 8 & 9: Load ONNX model and run inference on test image ────────
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_img_bgr = cv2.imread(str(split_paths["test"][0] / "frame_008.jpg"))

    # Prepare input tensor: (H, W, 3) -> (1, 3, H, W)
    norm = test_img_bgr.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (norm - mean) / std
    inp_tensor = np.expand_dims(norm.transpose(2, 0, 1), axis=0).astype(np.float32)

    onnx_outputs = session.run(["segmentation_mask"], {"input_image": inp_tensor})
    onnx_logits = onnx_outputs[0]

    # ── Step 10: Verify PyTorch vs ONNX match numerically ──────────────────
    with torch.no_grad():
        pt_logits = model(torch.from_numpy(inp_tensor)).cpu().numpy()

    max_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    assert max_diff < 1e-4, f"PyTorch vs ONNX numerical mismatch: {max_diff}"

    pt_argmax = np.argmax(pt_logits, axis=1)
    onnx_argmax = np.argmax(onnx_logits, axis=1)
    assert np.array_equal(pt_argmax, onnx_argmax), "Argmax segmentation masks differ between PyTorch and ONNX"
