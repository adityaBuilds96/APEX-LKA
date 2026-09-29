"""
dashboard/components/annotation_viewer.py
=========================================
Interactive Annotation Viewer component for APEX LKA Dataset Command Center (Section 5, Tab 5).

Capabilities:
  - Side-by-side image & semantic mask display
  - Alpha-blended mask overlay with smooth opacity slider (0% - 100%)
  - Mask-only visualization & boundary edge contours overlay
  - Dynamic class visibility toggles (Road, Left Lane, Right Lane)
  - Filter by status (ALL, PENDING, AUTO-ANNOTATED, VERIFIED, LOW CONFIDENCE)
  - Frame metadata panel: resolution, class coverage, confidence badge, QA score
  - Direct human verification and approval actions
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.config import PROJECT_ROOT, cfg
from src.dataset.annotation_prep import CLASS_DEFINITIONS, AnnotationWorkspace
from src.dataset.quality_scorer import QualityScorer, QualityTier


# 4-class semantic color map (RGB)
CLASS_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (15, 23, 42),      # Background (Glassmorphic dark slate)
    1: (168, 85, 247),    # Road surface (Vibrant Violet / Purple)
    2: (34, 197, 94),     # Left Lane (Vibrant Green)
    3: (56, 189, 248),    # Right Lane (Vibrant Sky Blue / Cyan)
}

VIEWER_CSS = """
<style>
.viewer-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    padding: 16px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
}
.meta-chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}
.meta-chip-high { background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.5); color: #22c55e; }
.meta-chip-med { background: rgba(251, 191, 36, 0.15); border: 1px solid rgba(251, 191, 36, 0.5); color: #fbbf24; }
.meta-chip-low { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.5); color: #ef4444; }
.meta-chip-info { background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.4); color: #38bdf8; }
</style>
"""


def colorize_mask(mask: np.ndarray, visible_classes: Optional[List[int]] = None) -> np.ndarray:
    """Converts single-channel class ID mask to an RGB visualization image."""
    h, w = mask.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    active_classes = visible_classes if visible_classes is not None else [0, 1, 2, 3]

    for class_id, rgb_color in CLASS_COLOR_MAP.items():
        if class_id in active_classes:
            colored[mask == class_id] = rgb_color
    return colored


def generate_edge_overlay(image_rgb: np.ndarray, mask: np.ndarray, visible_classes: Optional[List[int]] = None) -> np.ndarray:
    """Draws colored boundary contour edges over the original image."""
    output = image_rgb.copy()
    active_classes = visible_classes if visible_classes is not None else [1, 2, 3]

    for class_id in active_classes:
        if class_id == 0:
            continue
        binary = (mask == class_id).astype(np.uint8) * 255
        if np.count_nonzero(binary) == 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = CLASS_COLOR_MAP.get(class_id, (255, 255, 255))
        cv2.drawContours(output, contours, -1, color, 2)
    return output


def render_annotation_viewer(base_dir: Optional[Path] = None, selected_stem: Optional[str] = None) -> None:
    """
    Renders Tab 5 Annotation Viewer with all controls and navigation.
    """
    root_dir = Path(base_dir) if base_dir else PROJECT_ROOT
    st.markdown(VIEWER_CSS, unsafe_allow_html=True)

    img_dir = root_dir / "data" / "annotated" / "images"
    mask_dir = root_dir / "data" / "annotated" / "masks"
    cand_dir = root_dir / "annotations" / "auto_generated_candidates"
    manifest_path = root_dir / "annotations" / "workspace_manifest.json"
    qa_scorer = QualityScorer()

    # Discover all candidate and annotated frames
    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_files:
        raw_dir = root_dir / "data" / "raw_frames"
        img_files = sorted(list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.png")))

    if not img_files:
        st.info("No frames currently available for annotation viewing. Please upload or ingest images first.")
        return

    # Load manifest data
    manifest_records: Dict[str, dict] = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                manifest_records = {item.get("stem"): item for item in data.get("items", [])}
        except Exception:
            pass

    # Top Controls Bar
    c_flt, c_idx = st.columns([1, 2])

    with c_flt:
        status_filter = st.selectbox(
            "Filter Frames",
            ["All Frames", "Pending Annotation", "Auto-Annotated", "Verified Only", "Low Confidence (<0.40)"],
            index=0,
            key="ann_view_filter",
        )

    # Filter frame list
    filtered_frames: List[Path] = []
    for f in img_files:
        stem = f.stem
        m_entry = manifest_records.get(stem, {})
        has_mask = (mask_dir / f"{stem}.png").exists() or (cand_dir / f"{stem}.png").exists()
        is_verified = m_entry.get("verified", False) or m_entry.get("status") == "verified"
        conf = float(m_entry.get("confidence", 0.0))

        if status_filter == "All Frames":
            filtered_frames.append(f)
        elif status_filter == "Pending Annotation" and not has_mask:
            filtered_frames.append(f)
        elif status_filter == "Auto-Annotated" and (has_mask and not is_verified):
            filtered_frames.append(f)
        elif status_filter == "Verified Only" and (has_mask and is_verified):
            filtered_frames.append(f)
        elif status_filter == "Low Confidence (<0.40)" and (has_mask and conf < 0.40):
            filtered_frames.append(f)

    if not filtered_frames:
        st.warning(f"No frames match filter criteria: '{status_filter}'")
        return

    # Handle selection
    frame_stems = [f.stem for f in filtered_frames]
    default_idx = 0
    if selected_stem and selected_stem in frame_stems:
        default_idx = frame_stems.index(selected_stem)

    with c_idx:
        current_stem = st.selectbox(
            f"Select Frame ({len(filtered_frames)} matching)",
            options=frame_stems,
            index=default_idx,
            key="ann_view_stem_select",
        )

    active_file = next(f for f in filtered_frames if f.stem == current_stem)
    stem = active_file.stem

    # Locate corresponding mask
    active_mask_path = mask_dir / f"{stem}.png"
    if not active_mask_path.exists():
        active_mask_path = cand_dir / f"{stem}.png"

    has_active_mask = active_mask_path.exists()

    # Load image and mask
    image_bgr = cv2.imread(str(active_file))
    if image_bgr is None:
        st.error(f"Failed to load image file: {active_file}")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    if has_active_mask:
        mask = cv2.imread(str(active_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = np.zeros((h, w), dtype=np.uint8)

    # Viewer Configuration Toolbar
    st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
    t1, t2, t3, t4, t5 = st.columns([2, 1, 1, 1, 1])

    with t1:
        view_mode = st.radio(
            "Visualization Mode",
            ["Side-by-Side", "Alpha Overlay", "Mask-Only", "Boundary Edges"],
            horizontal=True,
            key="ann_view_mode",
        )
    with t2:
        opacity = st.slider("Overlay Opacity", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
    with t3:
        show_road = st.checkbox("🛣️ Road", value=True)
    with t4:
        show_left = st.checkbox("🟢 Left Lane", value=True)
    with t5:
        show_right = st.checkbox("🔷 Right Lane", value=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Determine visible classes
    visible_classes = [0]
    if show_road:
        visible_classes.append(1)
    if show_left:
        visible_classes.append(2)
    if show_right:
        visible_classes.append(3)

    # Render Visuals based on Mode
    color_mask = colorize_mask(mask, visible_classes=visible_classes)

    if view_mode == "Side-by-Side":
        col_img, col_msk = st.columns(2)
        with col_img:
            st.image(image_rgb, caption=f"Original Frame: {active_file.name} ({w}×{h})", use_container_width=True)
        with col_msk:
            st.image(color_mask, caption=f"Semantic Segmentation Mask (4-Class)", use_container_width=True)

    elif view_mode == "Alpha Overlay":
        blended = image_rgb.copy()
        non_bg = mask > 0
        for c in visible_classes:
            if c == 0:
                continue
            c_mask = mask == c
            blended[c_mask] = cv2.addWeighted(
                image_rgb[c_mask], 1.0 - opacity, color_mask[c_mask], opacity, 0
            )
        st.image(blended, caption=f"Alpha Blended Overlay ({int(opacity*100)}% Opacity)", use_container_width=True)

    elif view_mode == "Mask-Only":
        st.image(color_mask, caption="Raw 4-Class Semantic Mask Preview", use_container_width=True)

    elif view_mode == "Boundary Edges":
        edges = generate_edge_overlay(image_rgb, mask, visible_classes=visible_classes)
        st.image(edges, caption="Lane & Road Boundary Contours Overlay", use_container_width=True)

    # Frame Metadata & Quality Assessment
    m_info = manifest_records.get(stem, {})
    conf_score = float(m_info.get("confidence", 0.75 if has_active_mask else 0.0))
    is_verified = bool(m_info.get("verified", False))

    qa_report = qa_scorer.evaluate(image_rgb, mask) if has_active_mask else None

    st.markdown('<div class="viewer-card">', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns([1.5, 1, 1, 1.5])

    with m1:
        st.markdown(f"**Stem:** `{stem}`")
        st.markdown(f"**Dimensions:** `{w}×{h} px`")
        status_chip = "meta-chip-high" if is_verified else ("meta-chip-med" if has_active_mask else "meta-chip-low")
        status_text = "VERIFIED" if is_verified else ("AUTO-ANNOTATED" if has_active_mask else "PENDING")
        st.markdown(f'<span class="meta-chip {status_chip}">{status_text}</span>', unsafe_allow_html=True)

    with m2:
        conf_chip = "meta-chip-high" if conf_score >= 0.7 else ("meta-chip-med" if conf_score >= 0.4 else "meta-chip-low")
        st.markdown(f"**Confidence:** `{conf_score:.2f}`")
        st.markdown(f'<span class="meta-chip {conf_chip}">CONF: {int(conf_score*100)}%</span>', unsafe_allow_html=True)

    with m3:
        if qa_report:
            tier_name = qa_report.tier.value.upper()
            tier_chip = "meta-chip-high" if qa_report.tier == QualityTier.EXCELLENT else (
                "meta-chip-med" if qa_report.tier == QualityTier.ACCEPTABLE else "meta-chip-low"
            )
            st.markdown(f"**QA Score:** `{qa_report.overall_score:.2f}`")
            st.markdown(f'<span class="meta-chip {tier_chip}">{tier_name}</span>', unsafe_allow_html=True)
        else:
            st.markdown("**QA Score:** N/A")

    with m4:
        # Pixel Statistics
        total_px = max(1, h * w)
        road_px = int(np.count_nonzero(mask == 1))
        left_px = int(np.count_nonzero(mask == 2))
        right_px = int(np.count_nonzero(mask == 3))
        st.markdown(f"🛣️ Road: `{(road_px/total_px)*100:.1f}%`")
        st.markdown(f"🟢 Left: `{(left_px/total_px)*100:.2f}%` | 🔷 Right: `{(right_px/total_px)*100:.2f}%`")

    # Quick Verification Actions
    act1, act2, act3 = st.columns([1, 1, 2])
    with act1:
        if st.button("✅ Approve & Verify", use_container_width=True, key=f"btn_verify_{stem}"):
            if not (mask_dir / f"{stem}.png").exists() and (cand_dir / f"{stem}.png").exists():
                mask_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
            if stem in manifest_records:
                manifest_records[stem]["verified"] = True
                manifest_records[stem]["status"] = "verified"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump({"items": list(manifest_records.values())}, f, indent=2)
            st.success(f"Frame {stem} marked as VERIFIED!")
            st.rerun()

    with act2:
        if st.button("🔄 Flag for Review", use_container_width=True, key=f"btn_flag_{stem}"):
            if stem in manifest_records:
                manifest_records[stem]["verified"] = False
                manifest_records[stem]["status"] = "needs_review"
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump({"items": list(manifest_records.values())}, f, indent=2)
            st.info(f"Frame {stem} flagged for human review.")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
