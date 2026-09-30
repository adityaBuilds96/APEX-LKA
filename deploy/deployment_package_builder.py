"""
deploy/deployment_package_builder.py
====================================
One-click edge deployment package generator for APEX-LKA:
- Bundles all necessary artifacts into deploy_package_YYYYMMDD/:
    * best_model.onnx (universal deployment model)
    * best_model.plan (if TensorRT engine is present)
    * class_legend.json (color palettes and class mappings)
    * inference_config.json (normalization constants, input dimensions)
    * standalone_inference.py (zero-dependency standalone edge runner)
    * README.md (comprehensive edge & Jetson Orin Nano guide)
    * requirements_edge.txt (minimal dependencies for edge devices)
- Creates compressed .zip package for seamless transport.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg

log = logging.getLogger("apex_deploy_builder")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


STANDALONE_INFERENCE_SCRIPT = '''#!/usr/bin/env python3
"""
standalone_inference.py
=======================
Self-contained, zero-dependency edge inference runner for APEX-LKA.
Requires ONLY:
    pip install numpy opencv-python onnxruntime

Usage:
    python standalone_inference.py --source webcam
    python standalone_inference.py --source video.mp4
    python standalone_inference.py --source test_image.jpg
    python standalone_inference.py --benchmark
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent


def load_configs() -> Tuple[Dict, Dict]:
    config_file = SCRIPT_DIR / "inference_config.json"
    legend_file = SCRIPT_DIR / "class_legend.json"

    with open(config_file, "r", encoding="utf-8") as f:
        inf_cfg = json.load(f)
    with open(legend_file, "r", encoding="utf-8") as f:
        legend = json.load(f)

    return inf_cfg, legend


class StandaloneLaneDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.config, self.legend = load_configs()
        self.model_h = self.config.get("input_height", 360)
        self.model_w = self.config.get("input_width", 640)
        self.mean = np.array(self.config.get("mean_rgb", [0.485, 0.456, 0.406]), dtype=np.float32)
        self.std = np.array(self.config.get("std_rgb", [0.229, 0.224, 0.225]), dtype=np.float32)
        self.conf_thresh = self.config.get("confidence_threshold", 0.50)

        if model_path is None:
            # Check for FP16 first, then default
            fp16_path = SCRIPT_DIR / "best_model_fp16.onnx"
            std_path = SCRIPT_DIR / "best_model.onnx"
            model_path = str(fp16_path if fp16_path.exists() else std_path)

        available_providers = ort.get_available_providers()
        chosen_providers = [p for p in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=chosen_providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_provider = self.session.get_providers()[0]
        print(f"[INFO] Loaded model: {Path(model_path).name} using {self.active_provider}")

    def preprocess(self, bgr_img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        orig_h, orig_w = bgr_img.shape[:2]
        resized = cv2.resize(bgr_img, (self.model_w, self.model_h))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        return tensor, (orig_w, orig_h)

    def predict(self, bgr_img: np.ndarray) -> Dict:
        t0 = time.perf_counter()
        tensor, (orig_w, orig_h) = self.preprocess(bgr_img)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        logits = outputs[0][0]  # (4, H, W)
        t_infer_ms = (time.perf_counter() - t0) * 1000.0

        # Stable softmax
        e_x = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = e_x / np.sum(e_x, axis=0, keepdims=True)

        # 0: Background, 1: Road, 2: Left Lane, 3: Right Lane
        road_mask = (probs[1] >= 0.40).astype(np.uint8) * 255
        left_mask = (probs[2] >= self.conf_thresh).astype(np.uint8) * 255
        right_mask = (probs[3] >= self.conf_thresh).astype(np.uint8) * 255

        return {
            "road_mask": road_mask,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "latency_ms": t_infer_ms,
            "orig_size": (orig_w, orig_h),
        }

    def render_overlay(self, bgr_img: np.ndarray, res: Dict) -> np.ndarray:
        orig_w, orig_h = res["orig_size"]
        disp = bgr_img.copy()

        # Resize masks back to original dimensions
        road = cv2.resize(res["road_mask"], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        left = cv2.resize(res["left_mask"], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(res["right_mask"], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        overlay = np.zeros_like(disp)
        overlay[road > 0] = (34, 197, 94)    # Emerald Green (Road)
        overlay[left > 0] = (68, 68, 239)    # Crimson Red (Left Lane)
        overlay[right > 0] = (248, 189, 56)  # Electric Cyan (Right Lane)

        alpha = 0.40
        has_any = (road > 0) | (left > 0) | (right > 0)
        disp[has_any] = cv2.addWeighted(disp[has_any], 1.0 - alpha, overlay[has_any], alpha, 0)

        # HUD Text
        fps = 1000.0 / max(0.1, res["latency_ms"])
        hud_text = f"APEX-LKA | {self.active_provider} | {res['latency_ms']:.1f}ms ({fps:.1f} FPS)"
        cv2.rectangle(disp, (10, 10), (500, 42), (15, 23, 42), -1)
        cv2.putText(disp, hud_text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1, cv2.LINE_AA)

        return disp


def main():
    parser = argparse.ArgumentParser(description="Standalone APEX-LKA Inference Runner")
    parser.add_argument("--source", type=str, default="webcam", help="Source: webcam / 0 / video.mp4 / image.jpg")
    parser.add_argument("--model", type=str, default=None, help="Path to ONNX model")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark pass")
    args = parser.parse_args()

    detector = StandaloneLaneDetector(model_path=args.model)

    if args.benchmark:
        print("[INFO] Running benchmark (100 iterations)...")
        dummy = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        # Warmup
        for _ in range(10):
            _ = detector.predict(dummy)
        lats = []
        for _ in range(100):
            r = detector.predict(dummy)
            lats.append(r["latency_ms"])
        print(f"[RESULT] Median Latency: {np.median(lats):.2f} ms | FPS: {1000.0 / np.mean(lats):.1f}")
        return

    # Image source
    if Path(args.source).is_file() and args.source.lower().endswith((".jpg", ".png", ".jpeg")):
        img = cv2.imread(args.source)
        if img is None:
            print(f"[ERROR] Could not read image: {args.source}")
            sys.exit(1)
        res = detector.predict(img)
        vis = detector.render_overlay(img, res)
        out_name = f"result_{Path(args.source).name}"
        cv2.imwrite(out_name, vis)
        print(f"[SUCCESS] Saved output to {out_name} (Inference: {res['latency_ms']:.1f} ms)")
        return

    # Video or webcam
    is_cam = (args.source == "webcam" or args.source.isdigit())
    cap_src = int(args.source) if args.source.isdigit() or args.source == "webcam" else args.source
    cap = cv2.VideoCapture(0 if args.source == "webcam" else cap_src)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {args.source}")
        sys.exit(1)

    print("[INFO] Press 'q' or ESC in display window to exit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        res = detector.predict(frame)
        vis = detector.render_overlay(frame, res)
        cv2.imshow("APEX-LKA Standalone Perception", vis)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
'''

REQUIREMENTS_EDGE_CONTENT = """# APEX-LKA Minimal Edge Deployment Requirements
# Suitable for NVIDIA Jetson Orin Nano, Raspberry Pi 5, Intel NUC
numpy>=1.23.0
opencv-python-headless>=4.7.0
onnxruntime>=1.16.0
"""

CLASS_LEGEND_DATA = {
    "classes": [
        {
            "id": 0,
            "name": "Background",
            "bgr": [35, 35, 35],
            "rgb": [35, 35, 35],
            "hex": "#232323",
            "description": "Non-drivable regions, curbs, sky, guardrails, obstacles",
        },
        {
            "id": 1,
            "name": "Road Surface",
            "bgr": [34, 197, 94],
            "rgb": [94, 197, 34],
            "hex": "#22c55e",
            "description": "Drivable road asphalt, concrete, or dirt path",
        },
        {
            "id": 2,
            "name": "Left Lane Line",
            "bgr": [68, 68, 239],
            "rgb": [239, 68, 68],
            "hex": "#ef4444",
            "description": "Left lane boundary marking or physical road edge",
        },
        {
            "id": 3,
            "name": "Right Lane Line",
            "bgr": [248, 189, 56],
            "rgb": [56, 189, 248],
            "hex": "#38bdf8",
            "description": "Right lane boundary marking or physical road edge",
        },
    ]
}


def build_deployment_package(
    onnx_model_path: Optional[Union[str, Path]] = None,
    output_base_dir: Optional[Union[str, Path]] = None,
    zip_package: bool = True,
) -> Path:
    """
    Generate complete self-contained deployment bundle.
    """
    date_stamp = time.strftime("%Y%m%d")
    pkg_name = f"deploy_package_{date_stamp}"

    if output_base_dir is None:
        deploy_dir = PROJECT_ROOT / "deploy" / pkg_name
    else:
        deploy_dir = Path(output_base_dir) / pkg_name

    deploy_dir.mkdir(parents=True, exist_ok=True)
    log.info("Constructing deployment package in: %s", deploy_dir)

    # 1. Resolve ONNX model
    if onnx_model_path is not None:
        src_onnx = Path(onnx_model_path)
    else:
        src_onnx = PATHS.exported / "best_model.onnx"
        if not src_onnx.exists():
            log.info("Exporting fresh ONNX model for package...")
            from src.training.export import export_to_onnx
            export_to_onnx(output_path=src_onnx)

    dst_onnx = deploy_dir / "best_model.onnx"
    shutil.copy2(src_onnx, dst_onnx)
    log.info("Copied: %s (%.2f MB)", dst_onnx.name, dst_onnx.stat().st_size / (1024 * 1024))

    # Also copy FP16 model if available
    src_fp16 = src_onnx.parent / f"{src_onnx.stem}_fp16{src_onnx.suffix}"
    if src_fp16.exists():
        shutil.copy2(src_fp16, deploy_dir / src_fp16.name)
        log.info("Copied: %s (%.2f MB)", src_fp16.name, src_fp16.stat().st_size / (1024 * 1024))

    # 2. Check for TensorRT engine
    src_plan = PATHS.exported / "best_model.plan"
    if src_plan.exists():
        shutil.copy2(src_plan, deploy_dir / "best_model.plan")
        log.info("Copied TensorRT Engine: best_model.plan")

    # 3. Write class_legend.json
    legend_path = deploy_dir / "class_legend.json"
    with open(legend_path, "w", encoding="utf-8") as f:
        json.dump(CLASS_LEGEND_DATA, f, indent=2)

    # 4. Write inference_config.json
    inf_config = {
        "model_architecture": "LaneSegNet-MobileNetV3-Small-FPN",
        "input_width": cfg["preprocessing"]["image_width"],
        "input_height": cfg["preprocessing"]["image_height"],
        "num_classes": 4,
        "mean_rgb": [0.485, 0.456, 0.406],
        "std_rgb": [0.229, 0.224, 0.225],
        "confidence_threshold": 0.50,
        "target_device": "NVIDIA Jetson Orin Nano / Universal Edge",
        "export_date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = deploy_dir / "inference_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(inf_config, f, indent=2)

    # 5. Write requirements_edge.txt
    req_path = deploy_dir / "requirements_edge.txt"
    req_path.write_text(REQUIREMENTS_EDGE_CONTENT, encoding="utf-8")

    # 6. Write standalone_inference.py
    runner_path = deploy_dir / "standalone_inference.py"
    runner_path.write_text(STANDALONE_INFERENCE_SCRIPT, encoding="utf-8")

    # 7. Write README.md
    readme_path = deploy_dir / "README.md"
    readme_content = f"""# APEX-LKA Edge Deployment Package
**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Architecture:** LaneSegNet (MobileNetV3-Small + FPN + SE Attention)

---

## 1. Quick Start
Run standalone lane detection on any host:
```bash
# 1. Install minimal dependencies
pip install -r requirements_edge.txt

# 2. Run inference on webcam
python standalone_inference.py --source webcam

# 3. Run inference on video or image
python standalone_inference.py --source input_video.mp4
python standalone_inference.py --source test_image.jpg

# 4. Run hardware speed benchmark
python standalone_inference.py --benchmark
```

---

## 2. NVIDIA Jetson Orin Nano Optimization
To achieve maximum FPS (50-70+ FPS) on NVIDIA Jetson:

### Option A: ONNX Runtime with CUDA / TensorRT Execution Provider
```bash
pip install onnxruntime-gpu
python standalone_inference.py --source 0
```

### Option B: Native TensorRT Engine Compilation (.plan)
Run the following command on your Jetson Orin Nano terminal:
```bash
trtexec --onnx=best_model.onnx \\
        --saveEngine=best_model.plan \\
        --fp16 \\
        --memPoolSize=workspace:512
```

---

## 3. Package File Manifest
- `best_model.onnx`: Universal ONNX model (Opset 17, dynamic batch).
- `class_legend.json`: Class indices and color palette.
- `inference_config.json`: Normalization mean/std and input dimensions.
- `standalone_inference.py`: Zero-dependency, production-grade lane perception script.
- `requirements_edge.txt`: Lightweight edge dependency list.
"""
    readme_path.write_text(readme_content, encoding="utf-8")

    # 8. Create ZIP archive
    zip_dest = deploy_dir.parent / f"{pkg_name}.zip"
    if zip_package:
        log.info("Compressing package to: %s...", zip_dest)
        with zipfile.ZipFile(zip_dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(deploy_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(deploy_dir.parent)
                    zf.write(file_path, arcname)
        log.info("Saved ZIP bundle: %s (%.2f MB)", zip_dest.name, zip_dest.stat().st_size / (1024 * 1024))

    return zip_dest if zip_package else deploy_dir


def main() -> None:
    build_deployment_package()


if __name__ == "__main__":
    main()
