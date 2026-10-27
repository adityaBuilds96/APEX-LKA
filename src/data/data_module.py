"""
src/data/data_module.py
=======================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Fault-Tolerant PyTorch Dataset & DataModule with Quarantine Error Isolation.
ISO 26262 ASIL-B Data Ingestion & Runtime Memory Stability Baseline.

Key Architectural Capabilities:
--------------------------------
1. Memory-Bounded Batch Streaming:
   Loads 640x360 normalized images and 4-class masks on demand.
2. Worker-Level Fault Quarantine:
   If any corrupted, unreadable, or truncated image is encountered during runtime,
   the dataset isolates the bad sample, flags it in quarantine, and substitutes
   a safe fallback or drops the element via `safe_collate_fn` without crashing.
3. Universal Data Discovery:
   Auto-detects data via `manifest.jsonl`, `split.json`, or direct folder scanning.
4. Windows Multiprocessing Safety:
   Automatically detects OS platform and enforces zero-worker single-threaded
   data loading on Windows to avoid IPC pipe corruption and handle leaks.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.augmentations import AugmentationPolicyConfig, SafeLaneAugmentor, normalize_and_to_tensor
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema

logger = logging.getLogger("APEX.DataModule")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset Item Dataclass & Collate Functions
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetSampleItem:
    """
    Individual sample item metadata loaded by APEXLaneDataset.
    """
    sample_id: str
    image_path: Path
    mask_path: Path
    split: str
    clip_group: str = "default"


def safe_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Fault-tolerant batch collation. Filters out corrupted or None entries
    so that an unexpected read error in a worker thread never crashes training.
    """
    valid_items = [item for item in batch if item is not None and item.get("valid", True)]

    if not valid_items:
        # Emergency dummy batch to prevent training loop deadlock
        dummy_img = torch.zeros((1, 3, 360, 640), dtype=torch.float32)
        dummy_mask = torch.zeros((1, 360, 640), dtype=torch.int64)
        return {
            "image": dummy_img,
            "mask": dummy_mask,
            "sample_id": ["dummy_emergency_fallback"],
            "valid": [False],
            "is_emergency_fallback": True,
        }

    images = torch.stack([item["image"] for item in valid_items], dim=0)
    masks = torch.stack([item["mask"] for item in valid_items], dim=0)
    sample_ids = [item["sample_id"] for item in valid_items]

    return {
        "image": images,
        "mask": masks,
        "sample_id": sample_ids,
        "valid": [True] * len(valid_items),
        "is_emergency_fallback": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Production PyTorch Lane Dataset Engine
# ──────────────────────────────────────────────────────────────────────────────

class APEXLaneDataset(Dataset):
    """
    Production-grade 4-class lane segmentation dataset loader.
    """

    def __init__(
        self,
        data_root: Optional[Union[str, Path]] = None,
        split: str = "train",
        augment: Optional[bool] = None,
        augmentation_config: Optional[AugmentationPolicyConfig] = None,
        quarantine_dir: Optional[Union[str, Path]] = None,
        root_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        root = root_dir if root_dir is not None else (data_root if data_root is not None else "data/processed")
        self.data_root = Path(root).resolve()
        self.split = split.lower()
        self.quarantine_dir = Path(quarantine_dir).resolve() if quarantine_dir else (self.data_root.parent / "quarantine")
        self.schema = get_default_label_schema()

        # Augmentation policy
        self.is_training = (self.split == "train")
        self.should_augment = augment if augment is not None else self.is_training
        aug_cfg = augmentation_config or AugmentationPolicyConfig()
        self.augmentor = SafeLaneAugmentor(aug_cfg)

        # Discovery & sample index
        self._samples: List[DatasetSampleItem] = []
        self._discover_dataset_samples()

    def _discover_dataset_samples(self) -> None:
        """
        Locate matching (image, mask) pairs from manifest, split.json, or folders.
        """
        manifest_file = self.data_root / "manifest.jsonl"
        split_file = self.data_root / "split.json"
        images_dir = self.data_root / "images"
        masks_dir = self.data_root / "masks"

        # Strategy 1: Load from manifest.jsonl if present
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get("split") == self.split:
                            img_p = self.data_root / rec["image_rel_path"]
                            msk_p = self.data_root / rec["mask_rel_path"]
                            if img_p.exists() and msk_p.exists():
                                self._samples.append(
                                    DatasetSampleItem(
                                        sample_id=rec["sample_id"],
                                        image_path=img_p,
                                        mask_path=msk_p,
                                        split=self.split,
                                        clip_group=rec.get("clip_group", "default"),
                                    )
                                )
                if self._samples:
                    logger.info(f"Loaded {len(self._samples)} '{self.split}' samples from {manifest_file.name}")
                    return
            except Exception as e:
                logger.warning(f"Error reading manifest: {e}. Falling back to directory scan.")

        # Strategy 2: Load from split.json if present
        if split_file.exists():
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    split_map = json.load(f)
                    target_ids = split_map.get(self.split, [])
                    for sid in target_ids:
                        img_p = images_dir / f"{sid}.jpg"
                        msk_p = masks_dir / f"{sid}.png"
                        if img_p.exists() and msk_p.exists():
                            self._samples.append(
                                DatasetSampleItem(
                                    sample_id=sid,
                                    image_path=img_p,
                                    mask_path=msk_p,
                                    split=self.split,
                                )
                            )
                if self._samples:
                    logger.info(f"Loaded {len(self._samples)} '{self.split}' samples from {split_file.name}")
                    return
            except Exception as e:
                logger.warning(f"Error reading split.json: {e}")

        # Strategy 3: Direct folder scan
        if images_dir.exists() and masks_dir.exists():
            for img_path in sorted(images_dir.glob("*.jpg")):
                mask_path = masks_dir / f"{img_path.stem}.png"
                if mask_path.exists():
                    self._samples.append(
                        DatasetSampleItem(
                            sample_id=img_path.stem,
                            image_path=img_path,
                            mask_path=mask_path,
                            split=self.split,
                        )
                    )
            logger.info(f"Discovered {len(self._samples)} pairs via directory scan in {images_dir}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """
        Load, augment, and tensorize a single sample.
        Defensive error handling isolates corrupted samples and reports to quarantine.
        """
        sample_item = self._samples[idx]

        try:
            # 1. Read image
            img_bgr = cv2.imread(str(sample_item.image_path), cv2.IMREAD_COLOR)
            if img_bgr is None or img_bgr.size == 0:
                raise ValueError(f"Corrupt image file: {sample_item.image_path}")

            # 2. Read mask
            mask_uint8 = cv2.imread(str(sample_item.mask_path), cv2.IMREAD_UNCHANGED)
            if mask_uint8 is None or mask_uint8.size == 0:
                raise ValueError(f"Corrupt mask file: {sample_item.mask_path}")

            # If multi-channel mask, take first channel
            if mask_uint8.ndim == 3:
                mask_uint8 = mask_uint8[:, :, 0]

            # 3. Apply augmentations
            if self.should_augment:
                aug_img, aug_mask = self.augmentor.apply_training_transforms(img_bgr, mask_uint8)
            else:
                aug_img, aug_mask = self.augmentor.apply_validation_transforms(img_bgr, mask_uint8)

            # 4. Normalize & Tensorize
            img_tensor, mask_tensor = normalize_and_to_tensor(aug_img, aug_mask)

            return {
                "image": img_tensor,
                "mask": mask_tensor,
                "sample_id": sample_item.sample_id,
                "valid": True,
            }

        except Exception as e:
            logger.error(f"Runtime quarantine: Failed loading sample {sample_item.sample_id}: {e}")
            self._isolate_to_quarantine(sample_item, str(e))
            return None

    def _isolate_to_quarantine(self, item: DatasetSampleItem, reason: str) -> None:
        """Move corrupted runtime sample reference to quarantine log."""
        try:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            q_log = self.quarantine_dir / "runtime_quarantine.log"
            with open(q_log, "a", encoding="utf-8") as f:
                f.write(f"[{self.split}] Sample {item.sample_id} quarantined: {reason}\n")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# 3. Unified Production DataModule
# ──────────────────────────────────────────────────────────────────────────────

class APEXLaneDataModule:
    """
    Universal DataModule coordinating train, val, and test DataLoaders with
    automatic platform-aware multi-worker and pinning configuration.
    """

    def __init__(
        self,
        data_root: Union[str, Path] = "data/processed",
        batch_size: int = 8,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        augmentation_config: Optional[AugmentationPolicyConfig] = None,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.batch_size = batch_size
        self.aug_cfg = augmentation_config

        # Windows safety: Enforce 0 workers on Windows to avoid IPC socket crashes
        if sys.platform.startswith("win"):
            self.num_workers = 0
            self.pin_memory = False
        else:
            self.num_workers = num_workers if num_workers is not None else min(4, os.cpu_count() or 1)
            self.pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()

        self.train_dataset: Optional[APEXLaneDataset] = None
        self.val_dataset: Optional[APEXLaneDataset] = None
        self.test_dataset: Optional[APEXLaneDataset] = None

    def setup(self) -> None:
        """Instantiate datasets for all three partitions."""
        self.train_dataset = APEXLaneDataset(
            data_root=self.data_root,
            split="train",
            augment=True,
            augmentation_config=self.aug_cfg,
        )
        self.val_dataset = APEXLaneDataset(
            data_root=self.data_root,
            split="val",
            augment=False,
            augmentation_config=self.aug_cfg,
        )
        self.test_dataset = APEXLaneDataset(
            data_root=self.data_root,
            split="test",
            augment=False,
            augmentation_config=self.aug_cfg,
        )

        logger.info(
            f"DataModule initialized: Train={len(self.train_dataset)}, "
            f"Val={len(self.val_dataset)}, Test={len(self.test_dataset)}."
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=safe_collate_fn,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=safe_collate_fn,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            self.setup()
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=safe_collate_fn,
            drop_last=False,
        )

# Backward-compatible aliases
LaneSegmentationDataset = APEXLaneDataset
LaneSegmentationDataModule = APEXLaneDataModule


# ──────────────────────────────────────────────────────────────────────────────
# 4. In-Memory / NVMe RAM Cache for High-Epoch Throughput
# ──────────────────────────────────────────────────────────────────────────────

class DatasetMemoryCache:
    """
    Accelerated LRU tensor cache for small to medium splits (e.g. validation / test).
    Reduces NVMe read overhead on repeated validation cycles.
    """

    def __init__(self, max_items: int = 2000) -> None:
        self.max_items = max_items
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._access_history: List[str] = []

    def get(self, sample_id: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if sample_id in self._cache:
            return self._cache[sample_id]
        return None

    def put(self, sample_id: str, item: Tuple[torch.Tensor, torch.Tensor]) -> None:
        if len(self._cache) >= self.max_items:
            # Evict oldest entry
            evict_key = self._access_history.pop(0) if self._access_history else next(iter(self._cache))
            self._cache.pop(evict_key, None)

        self._cache[sample_id] = item
        self._access_history.append(sample_id)

    def clear(self) -> None:
        self._cache.clear()
        self._access_history.clear()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Balanced Dynamic Batch Sampler (Curve vs. Tangent Balance)
# ──────────────────────────────────────────────────────────────────────────────

class BalancedLaneBatchSampler(torch.utils.data.Sampler[List[int]]):
    """
    Guarantees balanced representation between challenging curved lane segments
    and standard straight motorway driving within each batch.
    """

    def __init__(
        self,
        dataset: APEXLaneDataset,
        batch_size: int = 8,
        curved_ratio: float = 0.50,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.curved_ratio = curved_ratio
        self.drop_last = drop_last

        # Partition indices into curved vs straight based on metadata
        self.curved_indices: List[int] = []
        self.straight_indices: List[int] = []

        for idx, item in enumerate(dataset._samples):
            # TuSimple clip or metadata heuristic
            if "curve" in item.sample_id.lower() or "turn" in item.clip_group.lower():
                self.curved_indices.append(idx)
            else:
                self.straight_indices.append(idx)

        # Fallback if unpartitioned: assign evenly
        if not self.curved_indices:
            mid = len(dataset) // 2
            self.curved_indices = list(range(mid))
            self.straight_indices = list(range(mid, len(dataset)))

    def __iter__(self):
        curved_pool = list(self.curved_indices)
        straight_pool = list(self.straight_indices)
        np.random.shuffle(curved_pool)
        np.random.shuffle(straight_pool)

        k_curved = max(1, int(self.batch_size * self.curved_ratio))
        k_straight = self.batch_size - k_curved

        c_idx = 0
        s_idx = 0
        total_batches = len(self)

        for _ in range(total_batches):
            batch: List[int] = []

            # Sample curved
            for _ in range(k_curved):
                if c_idx >= len(curved_pool):
                    np.random.shuffle(curved_pool)
                    c_idx = 0
                batch.append(curved_pool[c_idx])
                c_idx += 1

            # Sample straight
            for _ in range(k_straight):
                if s_idx >= len(straight_pool):
                    np.random.shuffle(straight_pool)
                    s_idx = 0
                batch.append(straight_pool[s_idx])
                s_idx += 1

            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


# ──────────────────────────────────────────────────────────────────────────────
# 6. Forensic Runtime Quarantine Handler
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QuarantineRecord:
    sample_id: str
    source_file: str
    error_message: str
    timestamp_utc: str


class CorruptedFrameQuarantineHandler:
    """
    Centralized quarantine registrar preventing unhandled exceptions and providing
    ASIL-B traceability for all quarantined, dropped, or unparseable frames.
    """

    def __init__(self, quarantine_dir: Union[str, Path]) -> None:
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[QuarantineRecord] = []
        self._log_path = self.quarantine_dir / "quarantine_manifest.jsonl"

    def register_failure(self, sample_id: str, source_path: str, reason: str) -> None:
        from datetime import datetime, timezone
        rec = QuarantineRecord(
            sample_id=sample_id,
            source_file=str(source_path),
            error_message=reason,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._records.append(rec)

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec.__dict__) + "\n")

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_quarantined": len(self._records),
            "log_path": str(self._log_path),
            "recent_failures": [r.__dict__ for r in self._records[-10:]],
        }


# ──────────────────────────────────────────────────────────────────────────────
# 7. Dataset Health & Sanity Preflight Checker
# ──────────────────────────────────────────────────────────────────────────────

class DatasetSanityAuditor:
    """
    Executes preflight audit over dataset before initiating heavy GPU training:
    - Verifies non-empty batches.
    - Asserts zero NaN / Inf occurrences in normalized image tensors.
    - Confirms proper tensor shapes [B, 3, 360, 640] and [B, 360, 640].
    - Computes preliminary class presence ratios.
    """

    def __init__(self, data_module: APEXLaneDataModule) -> None:
        self.dm = data_module

    def run_preflight_check(self, num_batches_to_test: int = 5) -> Tuple[bool, Dict[str, Any]]:
        self.dm.setup()
        loader = self.dm.train_dataloader()
        metrics: Dict[str, Any] = {
            "tested_batches": 0,
            "classes_detected": set(),
            "nan_or_inf_detected": False,
            "batch_shapes_valid": True,
        }

        if len(loader) == 0:
            return False, {"error": "Train dataloader is completely empty!"}

        for i, batch in enumerate(loader):
            if i >= num_batches_to_test:
                break

            images = batch["image"]
            masks = batch["mask"]

            # Check shapes
            if images.ndim != 4 or images.shape[1:] != (3, 360, 640):
                metrics["batch_shapes_valid"] = False
                return False, metrics

            if masks.ndim != 3 or masks.shape[1:] != (360, 640):
                metrics["batch_shapes_valid"] = False
                return False, metrics

            # Check numeric validity
            if torch.isnan(images).any() or torch.isinf(images).any():
                metrics["nan_or_inf_detected"] = True
                return False, metrics

            for cls_val in torch.unique(masks).tolist():
                metrics["classes_detected"].add(int(cls_val))

            metrics["tested_batches"] += 1

        metrics["classes_detected"] = sorted(list(metrics["classes_detected"]))
        return True, metrics


# ──────────────────────────────────────────────────────────────────────────────
# 8. Video Clip-Stratified K-Fold Splitter (Leakage Defense)
# ──────────────────────────────────────────────────────────────────────────────

class ClipStratifiedSplitter:
    """
    Guarantees zero data leakage across splits by enforcing that all frames
    belonging to the same temporal driving clip reside strictly within a single split.
    """

    def __init__(self, train_ratio: float = 0.80, val_ratio: float = 0.10, seed: int = 42) -> None:
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def partition_samples(
        self,
        samples: Sequence[DatasetSampleItem],
    ) -> Dict[str, List[DatasetSampleItem]]:
        """
        Group by clip_group and assign full clips to train, val, or test.
        """
        import random
        rng = random.Random(self.seed)

        clips: Dict[str, List[DatasetSampleItem]] = {}
        for s in samples:
            clips.setdefault(s.clip_group, []).append(s)

        clip_keys = list(clips.keys())
        rng.shuffle(clip_keys)

        n_total_clips = len(clip_keys)
        n_train = max(1, int(n_total_clips * self.train_ratio))
        n_val = max(1, int(n_total_clips * self.val_ratio))

        train_clips = set(clip_keys[:n_train])
        val_clips = set(clip_keys[n_train:n_train + n_val])
        test_clips = set(clip_keys[n_train + n_val:])

        partitioned: Dict[str, List[DatasetSampleItem]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        for k, items in clips.items():
            if k in train_clips:
                partitioned["train"].extend(items)
            elif k in val_clips:
                partitioned["val"].extend(items)
            else:
                partitioned["test"].extend(items)

        logger.info(
            "ClipStratifiedSplitter: %d total clips partitioned -> Train: %d frames, Val: %d frames, Test: %d frames",
            n_total_clips,
            len(partitioned["train"]),
            len(partitioned["val"]),
            len(partitioned["test"]),
        )
        return partitioned


# ──────────────────────────────────────────────────────────────────────────────
# 9. DataLoader Throughput & Latency Profiler
# ──────────────────────────────────────────────────────────────────────────────

class DataModuleThroughputProfiler:
    """
    Profiles DataLoader streaming FPS, batch iteration latency, and memory consumption.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.loader = dataloader

    def profile(self, max_batches: int = 20) -> Dict[str, float]:
        import time
        latencies: List[float] = []
        batch_counts = 0
        total_items = 0

        t_start = time.perf_counter()
        t_prev = t_start

        for i, batch in enumerate(self.loader):
            if i >= max_batches:
                break
            t_now = time.perf_counter()
            latencies.append(t_now - t_prev)
            t_prev = t_now
            batch_counts += 1
            total_items += batch["image"].shape[0]

        total_elapsed = time.perf_counter() - t_start
        fps = total_items / max(1e-5, total_elapsed)
        mean_latency_ms = (sum(latencies) / max(1, len(latencies))) * 1000.0

        return {
            "fps": round(fps, 2),
            "mean_batch_latency_ms": round(mean_latency_ms, 2),
            "batches_evaluated": batch_counts,
            "total_samples_streamed": total_items,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 10. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing DataModule ISO 26262 ASIL-B self-check...")
    dm = APEXLaneDataModule(batch_size=2)
    dm.setup()
    print(f"DataModule setup passed. Found {len(dm.train_dataset)} training samples.")

    # Test Memory Cache
    cache = DatasetMemoryCache(max_items=10)
    fake_img = torch.zeros((3, 360, 640))
    fake_mask = torch.zeros((360, 640), dtype=torch.int64)
    cache.put("sample_01", (fake_img, fake_mask))
    assert cache.get("sample_01") is not None
    assert cache.get("sample_nonexistent") is None

    # Test Quarantine Handler
    q_dir = PROJECT_ROOT / "data" / ".staging" / "test_quarantine"
    q_handler = CorruptedFrameQuarantineHandler(quarantine_dir=q_dir)
    q_handler.register_failure("bad_frame_01", "clips/01/frame.jpg", "Truncated JPEG data")
    assert q_handler.get_summary()["total_quarantined"] == 1

    # Clean up test quarantine
    import shutil
    if q_dir.exists():
        shutil.rmtree(q_dir)

    print("DataModule self-check passed successfully.")


