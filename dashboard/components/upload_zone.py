"""
dashboard/components/upload_zone.py
===================================
Premium glassmorphic drag-and-drop upload zone component (Section 3).

Follows exact Section 3.1 Design Tokens:
  - Glassmorphic container with pulsing animated border
  - Multi-format ingestion (JPG, PNG, WEBP, TIFF, MP4, AVI, MOV, ZIP, Folder Path)
  - Progressive non-blocking thumbnail rendering
  - Real-time pipeline status telemetry (Queued -> Processing -> Done -> Failed)
"""

import shutil
import time
from pathlib import Path
from typing import List, Optional

import streamlit as st
from PIL import Image

from src.config import PROJECT_ROOT, cfg
from src.dataset.batch_processor import BatchProcessor, ItemStatus
from src.dataset.pipeline_orchestrator import PipelineOrchestrator, PipelineExecutionSummary
from src.dataset.thumbnail_generator import ThumbnailGenerator


UPLOAD_ZONE_CSS = """
<style>
@keyframes pulseGlow {
    0% { border-color: rgba(56, 189, 248, 0.2); box-shadow: 0 0 10px rgba(56, 189, 248, 0.05); }
    50% { border-color: rgba(56, 189, 248, 0.6); box-shadow: 0 0 25px rgba(56, 189, 248, 0.2); }
    100% { border-color: rgba(56, 189, 248, 0.2); box-shadow: 0 0 10px rgba(56, 189, 248, 0.05); }
}

.upload-zone-card {
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 2px dashed rgba(56, 189, 248, 0.35);
    border-radius: 12px;
    padding: 32px 20px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: pulseGlow 4s infinite ease-in-out;
    margin-bottom: 20px;
}
.upload-zone-card:hover {
    background: rgba(20, 30, 55, 0.95);
    border-color: rgba(56, 189, 248, 0.7);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), 0 0 20px rgba(56, 189, 248, 0.25);
}
.upload-icon {
    font-size: 2.8rem;
    margin-bottom: 8px;
    display: inline-block;
    filter: drop-shadow(0 0 8px rgba(56, 189, 248, 0.4));
}
.upload-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 6px;
}
.upload-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #94a3b8;
    margin-bottom: 16px;
}
.format-pill {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    margin: 0 4px;
}
</style>
"""


def render_upload_zone(base_dir: Optional[Path] = None) -> Optional[PipelineExecutionSummary]:
    """
    Renders the complete Section 3 Upload Experience interface.
    """
    root_dir = Path(base_dir) if base_dir else PROJECT_ROOT
    thumb_gen = ThumbnailGenerator(thumbnail_dir=root_dir / "data" / ".thumbnails")

    st.markdown(UPLOAD_ZONE_CSS, unsafe_allow_html=True)

    # 4 Upload Modes Switcher
    mode_selection = st.radio(
        "Upload Mode",
        ["Drop Images / Videos / ZIP", "Server Folder Scan"],
        horizontal=True,
        label_visibility="collapsed",
    )

    staged_paths: List[Path] = []
    temp_staging = root_dir / "data" / ".staging" / "upload_temp"
    temp_staging.mkdir(parents=True, exist_ok=True)

    if mode_selection == "Drop Images / Videos / ZIP":
        # Hero styled zone
        st.markdown("""
        <div class="upload-zone-card">
            <div class="upload-icon">☁️</div>
            <div class="upload-title">Drop images, driving videos, or ZIP archives here</div>
            <div class="upload-sub">Files land, get validated, deduplicated, and auto-annotated in the background</div>
            <div>
                <span class="format-pill">JPG / PNG / WEBP</span>
                <span class="format-pill">MP4 / AVI / MOV</span>
                <span class="format-pill">ZIP ARCHIVE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload files",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp", "mp4", "avi", "mov", "mkv", "zip"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="zone_uploader",
        )

        if uploaded_files:
            total_bytes = sum(f.size for f in uploaded_files)
            st.markdown(f"""
            <div style="background:rgba(15, 23, 42, 0.85); border:1px solid rgba(56, 189, 248, 0.2);
                        border-radius:8px; padding:12px 16px; margin:12px 0; font-family:'JetBrains Mono', monospace; font-size:0.8rem;">
                <span style="color:#38bdf8; font-weight:700;">📊 UPLOAD STAGING:</span>
                <span style="color:#f1f5f9;"> {len(uploaded_files)} files ({total_bytes / (1024*1024):.2f} MB)</span>
            </div>
            """, unsafe_allow_html=True)

            # Instant thumbnail preview grid
            preview_cols = st.columns(min(6, max(1, len(uploaded_files))))
            for idx, f in enumerate(uploaded_files[:6]):
                col = preview_cols[idx % len(preview_cols)]
                ext = Path(f.name).suffix.lower()
                with col:
                    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                        try:
                            im = Image.open(f)
                            col.image(im, caption=f"{f.name[:12]}...", use_container_width=True)
                        except Exception:
                            col.info(f.name[:10])
                    else:
                        col.markdown(f"""
                        <div style="background:rgba(15, 23, 42, 0.9); border:1px solid rgba(56, 189, 248, 0.2); border-radius:6px; padding:16px 8px; text-align:center;">
                            <div style="font-size:1.4rem;">{'🎬' if ext in {'.mp4', '.avi', '.mov'} else '📦'}</div>
                            <div style="font-family:'JetBrains Mono', monospace; font-size:0.65rem; color:#f1f5f9; overflow:hidden; text-overflow:ellipsis;">{f.name[:10]}</div>
                        </div>
                        """, unsafe_allow_html=True)

            if len(uploaded_files) > 6:
                st.caption(f"Showing 6 of {len(uploaded_files)} files. All files will be processed.")

            # Save uploaded files into staging
            for uf in uploaded_files:
                target_p = temp_staging / uf.name
                with open(target_p, "wb") as out_f:
                    out_f.write(uf.getbuffer())
                staged_paths.append(target_p)

    else:
        # Folder scan
        folder_input = st.text_input("Server Directory Path", value=str(root_dir / "data" / "raw_frames"))
        if st.button("🔍 Scan Folder", use_container_width=True):
            f_path = Path(folder_input)
            if f_path.exists() and f_path.is_dir():
                staged_paths = [f for f in f_path.iterdir() if f.is_file()]
                st.success(f"Found {len(staged_paths)} files in {f_path.name}")
            else:
                st.error("Invalid directory path.")

    if staged_paths:
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            do_auto_split = st.checkbox("🔀 Auto-split verified data (Train 70% / Val 15% / Test 15%)", value=False)
        with col_opt2:
            st.caption("⚡ Zero-touch auto-annotation & 9-point ADAS QA check are enabled by default.")

        if st.button("🚀 Process Batch & Launch Pipeline", type="primary", use_container_width=True):
            prog_bar = st.progress(0, text="Initializing pipeline...")
            status_txt = st.empty()

            orch = PipelineOrchestrator(base_dir=root_dir)

            def _on_prog(stage: str, frac: float, msg: str):
                prog_bar.progress(int(frac * 100), text=f"[{stage.upper()}] {msg}")
                status_txt.markdown(f"""
                <div style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:#38bdf8;
                            background:rgba(15, 23, 42, 0.9); padding:10px 14px; border-radius:6px; border:1px solid rgba(56, 189, 248, 0.25);">
                    ▶ <b>{stage.upper()}</b>: {msg}
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)

            summary = orch.run(source=staged_paths, auto_split=do_auto_split, progress_callback=_on_prog)

            # Cleanup staging
            shutil.rmtree(temp_staging, ignore_errors=True)

            if summary.success:
                st.balloons()
                st.markdown(f"""
                <div style="background:rgba(34, 197, 94, 0.1); border:1px solid rgba(34, 197, 94, 0.5);
                            border-radius:10px; padding:18px; margin-top:16px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.9rem; color:#22c55e; font-weight:700;">
                            ✔ BATCH {summary.batch_id} PROCESSED IN {summary.elapsed_seconds:.1f}s
                        </span>
                        <span style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#94a3b8;">
                            Mean QA Score: <b style="color:#38bdf8;">{summary.mean_quality_score:.2f} / 1.00</b>
                        </span>
                    </div>
                    <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:12px; font-family:'JetBrains Mono', monospace; font-size:0.8rem;">
                        <div style="background:rgba(15, 23, 42, 0.8); padding:8px; border-radius:6px;">
                            Valid: <b style="color:#22c55e;">{summary.valid_frames}</b>
                        </div>
                        <div style="background:rgba(15, 23, 42, 0.8); padding:8px; border-radius:6px;">
                            Duplicates: <b style="color:#fbbf24;">{summary.duplicates_filtered}</b>
                        </div>
                        <div style="background:rgba(15, 23, 42, 0.8); padding:8px; border-radius:6px;">
                            Auto-Masks: <b style="color:#38bdf8;">{summary.auto_annotated_count}</b>
                        </div>
                        <div style="background:rgba(15, 23, 42, 0.8); padding:8px; border-radius:6px;">
                            High Conf: <b style="color:#22c55e;">{summary.high_confidence_count}</b>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                return summary

    return None
