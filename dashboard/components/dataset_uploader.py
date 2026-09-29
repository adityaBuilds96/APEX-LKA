"""
dashboard/components/dataset_uploader.py
========================================
Heavenly Upload Experience Component for APEX LKA.

Provides:
  - Drag-and-drop multi-file uploader (Images, Videos, ZIP archives)
  - Instant client-side thumbnail previews
  - Asynchronous / progress-tracked data pipeline automation
  - Zero-blocking UI with live stage telemetry
"""

import io
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.dataset.pipeline_automation import AutomatedDataPipeline, PipelineExecutionResult


def render_heavenly_uploader(base_dir: Path) -> Optional[PipelineExecutionResult]:
    """
    Renders the heavenly upload card with instant thumbnail gallery
    and live automated pipeline execution.
    """
    st.markdown("""
    <div style="background:linear-gradient(135deg, #0d1a2d 0%, #070e1a 100%);
                border:1px solid #1e3a5f; border-radius:12px; padding:20px; margin-bottom:20px;
                box-shadow: 0 8px 32px 0 rgba(0, 245, 255, 0.05);">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-family:monospace; font-size:0.7rem; color:#00f5ff; letter-spacing:0.18em; text-transform:uppercase;">
                    DATA INGESTION ENGINE
                </span>
                <h3 style="margin:4px 0 0 0; color:#f8fafc; font-size:1.3rem;">⚡ Heavenly Ingestion & Auto-Pipeline</h3>
            </div>
            <div style="text-align:right;">
                <span style="background:rgba(0, 245, 255, 0.1); color:#00f5ff; padding:4px 10px; border-radius:6px; font-family:monospace; font-size:0.75rem; border:1px solid rgba(0, 245, 255, 0.25);">
                    Zero-Touch Mode Ready
                </span>
            </div>
        </div>
        <p style="color:#94a3b8; font-size:0.85rem; margin-top:8px; margin-bottom:0;">
            Drop road video drives (.mp4/.mov), image sequences (.jpg/.png), or ZIP datasets.
            Instant preview, validation, deduplication, classical CV auto-annotation, and dataset splitting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop images, raw driving videos, or full ZIP archives",
        type=["jpg", "jpeg", "png", "bmp", "webp", "mp4", "avi", "mov", "zip"],
        accept_multiple_files=True,
        help="Supports multi-file drag-and-drop. Automatically extracts and validates.",
        key="heavenly_uploader_input",
    )

    if not uploaded_files:
        st.markdown("""
        <div style="border:2px dashed #1a2d42; border-radius:10px; padding:30px; text-align:center; background:#070d15;">
            <div style="font-size:2rem; margin-bottom:8px;">📁</div>
            <div style="font-family:monospace; font-size:0.85rem; color:#64748b;">
                NO FILES SELECTED — DRAG & DROP ABOVE TO INSTANTLY AUDIT & INGEST
            </div>
        </div>
        """, unsafe_allow_html=True)
        return None

    # Instant thumbnail preview grid
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; margin:16px 0 8px 0;">
        <span style="font-family:monospace; font-size:0.75rem; color:#38bdf8; font-weight:600;">
            INSTANT CLIENT PREVIEWS ({len(uploaded_files)} FILES STAGED)
        </span>
        <span style="font-family:monospace; font-size:0.7rem; color:#64748b;">
            {sum(f.size for f in uploaded_files) / (1024 * 1024):.2f} MB TOTAL
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Show up to 10 instant thumbnail previews
    preview_cols = st.columns(min(6, max(1, len(uploaded_files))))
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for i, file_obj in enumerate(uploaded_files[:6]):
        col = preview_cols[i % len(preview_cols)]
        ext = Path(file_obj.name).suffix.lower()
        with col:
            if ext in img_exts:
                try:
                    img = Image.open(file_obj)
                    col.image(img, caption=f"{file_obj.name[:14]}...", use_container_width=True)
                except Exception:
                    col.warning(file_obj.name[:12])
            elif ext in {".mp4", ".avi", ".mov"}:
                col.markdown(f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:16px 8px; text-align:center;">
                    <div style="font-size:1.5rem;">🎬</div>
                    <div style="font-family:monospace; font-size:0.65rem; color:#94a3b8; overflow:hidden; text-overflow:ellipsis;">
                        {file_obj.name[:12]}
                    </div>
                    <div style="font-family:monospace; font-size:0.6rem; color:#38bdf8;">VIDEO</div>
                </div>
                """, unsafe_allow_html=True)
            elif ext == ".zip":
                col.markdown(f"""
                <div style="background:#0f172a; border:1px solid #1e293b; border-radius:8px; padding:16px 8px; text-align:center;">
                    <div style="font-size:1.5rem;">📦</div>
                    <div style="font-family:monospace; font-size:0.65rem; color:#94a3b8; overflow:hidden; text-overflow:ellipsis;">
                        {file_obj.name[:12]}
                    </div>
                    <div style="font-family:monospace; font-size:0.6rem; color:#10b981;">ARCHIVE</div>
                </div>
                """, unsafe_allow_html=True)

    if len(uploaded_files) > 6:
        st.caption(f"Showing 6 of {len(uploaded_files)} files. All {len(uploaded_files)} will be processed by the pipeline.")

    # Pipeline automation options
    st.markdown("<hr style='border:none; border-top:1px solid #1a2d42; margin:16px 0;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        auto_annotate = st.checkbox("⚡ Generate Classical CV Pseudo-Masks", value=True, help="Synthesizes 4-class candidate masks and QA scores automatically")
    with c2:
        deduplicate = st.checkbox("🔍 Filter Duplicates (SHA-256 + dHash)", value=True, help="Removes exact and perceptual near-duplicates")
    with c3:
        auto_split = st.checkbox("🔀 Partition into Train / Val / Test", value=False, help="Automatically partitions dataset (70/15/15)")

    run_btn = st.button("🚀 Launch Automated Pipeline", type="primary", use_container_width=True)

    if run_btn:
        progress_bar = st.progress(0, text="Initializing data pipeline...")
        status_box = st.empty()

        # Save uploaded files into a temporary staging area
        staging_dir = base_dir / "data" / "user_uploads_temp"
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_paths = []

        try:
            for f in uploaded_files:
                dest = staging_dir / f.name
                with open(dest, "wb") as out_f:
                    out_f.write(f.getbuffer())
                staged_paths.append(dest)

            pipeline = AutomatedDataPipeline(base_dir=base_dir)

            def _on_progress(stage: str, step: int, total: int, msg: str):
                pct = int((step / max(1, total)) * 100)
                progress_bar.progress(min(100, pct), text=f"[{stage}] {msg}")
                status_box.markdown(f"""
                <div style="font-family:monospace; font-size:0.8rem; color:#38bdf8; background:#070f1e; padding:8px 12px; border-radius:6px; border:1px solid #1a2d42;">
                    ▶ <span style="color:#00f5ff; font-weight:700;">{stage}</span>: {msg}
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.02)

            res = pipeline.run(
                source=staged_paths,
                auto_annotate=auto_annotate,
                auto_split=auto_split,
                deduplicate=deduplicate,
                progress_callback=_on_progress,
            )

            progress_bar.progress(100, text="Pipeline complete!")
            st.balloons()

            # Beautiful outcome dashboard card
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(7, 14, 26, 0.9) 100%);
                        border:1px solid #10b981; border-radius:10px; padding:18px; margin-top:16px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-family:monospace; font-size:0.85rem; color:#10b981; font-weight:700;">
                        ✔ PIPELINE EXECUTION SUCCESSFUL ({res.elapsed_sec:.2f}s)
                    </span>
                    <span style="font-family:monospace; font-size:0.75rem; color:#94a3b8;">
                        Mean QA Score: <b style="color:#00f5ff;">{res.mean_quality_score:.1f}/100</b>
                    </span>
                </div>
                <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:12px; font-family:monospace; font-size:0.8rem;">
                    <div style="background:#070d15; padding:8px; border-radius:6px; border:1px solid #1a2d42;">
                        <span style="color:#64748b;">Valid Images:</span> <b style="color:#10b981;">{res.valid_images}</b>
                    </div>
                    <div style="background:#070d15; padding:8px; border-radius:6px; border:1px solid #1a2d42;">
                        <span style="color:#64748b;">Duplicates Filtered:</span> <b style="color:#f59e0b;">{res.exact_duplicates + res.near_duplicates}</b>
                    </div>
                    <div style="background:#070d15; padding:8px; border-radius:6px; border:1px solid #1a2d42;">
                        <span style="color:#64748b;">Auto-Annotated:</span> <b style="color:#00f5ff;">{res.auto_annotated_count}</b>
                    </div>
                    <div style="background:#070d15; padding:8px; border-radius:6px; border:1px solid #1a2d42;">
                        <span style="color:#64748b;">Splits (T/V/T):</span> <b style="color:#e2e8f0;">{res.train_count}/{res.val_count}/{res.test_count}</b>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            return res

        finally:
            import shutil
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    return None
