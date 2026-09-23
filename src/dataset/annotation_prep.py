"""
src/dataset/annotation_prep.py
==============================
Annotation workspace setup and workflow management for unlabeled datasets (Mode B).

Features:
  - Initializes structured annotation directory: data/annotated/images & data/annotated/masks
  - Creates deterministic class legend:
      0 = Background
      1 = Drivable Road Surface
      2 = Left Lane Boundary
      3 = Right Lane Boundary
  - Generates LabelMe / CVAT compatible configuration
  - Tracks annotation progress in annotations/workspace_manifest.json
  - Optional preliminary mask generation strictly tagged: AUTO-GENERATED / NEEDS REVIEW
"""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


CLASS_DEFINITIONS = [
    {
        "id": 0,
        "name": "background",
        "color_rgb": [0, 0, 0],
        "hex": "#000000",
        "description": "Environment, sky, vegetation, vehicles, curbs, non-road surfaces",
    },
    {
        "id": 1,
        "name": "road",
        "color_rgb": [128, 64, 128],
        "hex": "#804080",
        "description": "Drivable road asphalt / surface within travel boundaries",
    },
    {
        "id": 2,
        "name": "left_lane",
        "color_rgb": [0, 255, 0],
        "hex": "#00FF00",
        "description": "Left lane boundary marking (solid or dashed)",
    },
    {
        "id": 3,
        "name": "right_lane",
        "color_rgb": [255, 100, 0],
        "hex": "#FF6400",
        "description": "Right lane boundary marking (solid or dashed)",
    },
]


@dataclass
class AnnotationWorkspace:
    """
    Manages the annotation directory, manifests, class legend, and review workflow.
    """
    workspace_root: Path
    annotated_images_dir: Path
    annotated_masks_dir: Path
    manifest_path: Path
    legend_path: Path

    @classmethod
    def create(cls, base_dir: Path) -> "AnnotationWorkspace":
        """Initialize annotation workspace directories and manifests."""
        base_dir = Path(base_dir)
        ann_root = base_dir / "annotations"
        images_dir = base_dir / "data" / "annotated" / "images"
        masks_dir = base_dir / "data" / "annotated" / "masks"

        ann_root.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        legend_file = ann_root / "class_legend.json"
        manifest_file = ann_root / "workspace_manifest.json"

        ws = cls(
            workspace_root=ann_root,
            annotated_images_dir=images_dir,
            annotated_masks_dir=masks_dir,
            manifest_path=manifest_file,
            legend_path=legend_file,
        )
        ws._ensure_legend()
        return ws

    def _ensure_legend(self) -> None:
        """Write the deterministic class legend if not already present."""
        data = {
            "version": "1.0.0",
            "model_name": "APEX-RLP",
            "classes": CLASS_DEFINITIONS,
            "deterministic_mapping": {
                "0": "background",
                "1": "road",
                "2": "left_lane",
                "3": "right_lane",
            },
        }
        with open(self.legend_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def update_manifest(self) -> dict:
        """Scan images and masks, returning current annotation progress."""
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        mask_exts = {".png", ".bmp"}

        images = {f.stem: f for f in self.annotated_images_dir.iterdir() if f.suffix.lower() in img_exts} if self.annotated_images_dir.exists() else {}
        masks = {f.stem: f for f in self.annotated_masks_dir.iterdir() if f.suffix.lower() in mask_exts} if self.annotated_masks_dir.exists() else {}

        completed = sorted(list(set(images.keys()) & set(masks.keys())))
        pending = sorted(list(set(images.keys()) - set(masks.keys())))
        orphaned_masks = sorted(list(set(masks.keys()) - set(images.keys())))

        total = len(images)
        comp_count = len(completed)
        progress_pct = (comp_count / total * 100.0) if total > 0 else 0.0

        manifest_data = {
            "last_updated": datetime.now().isoformat(),
            "total_images": total,
            "completed_masks": comp_count,
            "pending_images": len(pending),
            "orphaned_masks": len(orphaned_masks),
            "progress_percent": round(progress_pct, 1),
            "completed_stems": completed,
            "pending_stems": pending,
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        return manifest_data

    def generate_candidate_masks(
        self,
        image_paths: list[Path],
        output_dir: Optional[Path] = None,
    ) -> int:
        """
        Generate preliminary candidate masks using Classical CV baseline.
        STRICT REQUIREMENT: All candidate masks are saved with companion metadata
        explicitly stating: 'AUTO-GENERATED / NEEDS REVIEW'.
        Never silently treated as ground truth.
        """
        from src.inference.preprocessing import preprocess
        from src.inference.classical_cv import ClassicalCVPredictor
        from src.inference.postprocessing import postprocess

        predictor = ClassicalCVPredictor()
        out_dir = output_dir or (self.workspace_root / "auto_generated_candidates")
        out_dir.mkdir(parents=True, exist_ok=True)

        generated = 0
        for img_path in image_paths:
            pre = preprocess(img_path)
            if not pre.valid:
                continue

            pred = predictor.predict(pre)
            post = postprocess(pred)

            # Compose preliminary 2D class mask: 0=bg, 2=left_lane, 3=right_lane
            # (Classical CV detects lane boundaries)
            h, w = pre.model_h, pre.model_w
            candidate_mask = np.zeros((h, w), dtype=np.uint8)

            if post.clean_left_mask is not None:
                candidate_mask[post.clean_left_mask > 0] = 2

            if post.clean_right_mask is not None:
                candidate_mask[post.clean_right_mask > 0] = 3

            # Resize back to original image dimensions for annotation inspection
            orig_h, orig_w = pre.original_h, pre.original_w
            full_mask = cv2.resize(
                candidate_mask,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # Save mask
            mask_out_path = out_dir / f"{img_path.stem}.png"
            cv2.imwrite(str(mask_out_path), full_mask)

            # Save companion disclosure metadata
            meta_path = out_dir / f"{img_path.stem}_meta.json"
            meta = {
                "image": img_path.name,
                "mask": mask_out_path.name,
                "status": "AUTO-GENERATED / NEEDS REVIEW",
                "generator": "ClassicalCVPredictor",
                "generated_at": datetime.now().isoformat(),
                "verified_by_human": False,
                "classes_present": sorted(list(set(np.unique(full_mask).tolist()))),
                "notice": "DO NOT USE FOR SUPERVISED TRAINING WITHOUT HUMAN APPROVAL",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            generated += 1

        return generated
