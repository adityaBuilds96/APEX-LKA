"""
dashboard/components/progress_tracker.py
========================================
Real-time pipeline progress dashboard and telemetry tracker (Section 3.2 #5 & Section 6.1).

Visualizes:
  - Batch ID and pipeline execution stage
  - Real-time animated progress bar with percentage
  - High-density telemetry metrics: Processed, Valid, Duplicates, Corrupted, Throughput (fps), ETA
  - Polling integration for annotations/pipeline_state.json and in-memory BatchProcessor
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from src.config import PROJECT_ROOT


PROGRESS_TRACKER_CSS = """
<style>
.progress-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 16px 20px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
}
.progress-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.progress-title {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #f1f5f9;
}
.batch-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    padding: 3px 8px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #38bdf8;
    border-radius: 6px;
}
.stage-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    text-transform: uppercase;
}
.stage-active { background: rgba(56, 189, 248, 0.2); border: 1px solid rgba(56, 189, 248, 0.6); color: #38bdf8; }
.stage-done { background: rgba(34, 197, 94, 0.2); border: 1px solid rgba(34, 197, 94, 0.6); color: #22c55e; }
.stage-idle { background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.3); color: #94a3b8; }

.stat-strip {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 10px;
    margin-top: 14px;
}
.stat-box {
    background: rgba(20, 30, 55, 0.5);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 8px;
    padding: 8px 12px;
    text-align: center;
}
.stat-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #f1f5f9;
}
.stat-lbl {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}
</style>
"""


def load_pipeline_state(base_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Loads annotations/pipeline_state.json if it exists."""
    root_dir = Path(base_dir) if base_dir else PROJECT_ROOT
    state_file = root_dir / "annotations" / "pipeline_state.json"
    if not state_file.exists():
        return None
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def render_progress_tracker(
    base_dir: Optional[Path] = None,
    processor_status: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Renders real-time telemetry card for active or recent batch operations.
    """
    st.markdown(PROGRESS_TRACKER_CSS, unsafe_allow_html=True)
    root_dir = Path(base_dir) if base_dir else PROJECT_ROOT

    # Determine status source: in-memory processor_status takes precedence over file
    state = processor_status if processor_status else load_pipeline_state(root_dir)

    if not state:
        st.markdown("""
        <div class="progress-card" style="text-align:center; padding:24px 20px;">
            <div style="font-family:'Inter',sans-serif; font-size:0.9rem; color:#94a3b8;">
                💤 Pipeline Orchestrator is currently idle. No active background batch.
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#64748b; margin-top:4px;">
                Upload files or initiate ingestion to start pipeline processing.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Extract metrics
    batch_id = state.get("batch_id", "N/A")
    stage = state.get("stage", "idle").upper()
    progress = float(state.get("progress", 0.0))
    total = int(state.get("frames_total", state.get("total", 0)))
    processed = int(state.get("frames_processed", state.get("processed", 0)))
    failed = int(state.get("frames_failed", state.get("failed", 0)))
    duplicates = int(state.get("frames_duplicate", 0))
    fps = float(state.get("throughput_fps", 0.0))
    eta_sec = float(state.get("eta_seconds", 0.0))

    pct_display = int(progress * 100) if progress <= 1.0 else int(progress)
    progress_val = min(1.0, max(0.0, progress if progress <= 1.0 else progress / 100.0))

    is_complete = stage in {"COMPLETED", "DONE", "FINISHED"} or (total > 0 and processed >= total)
    stage_class = "stage-done" if is_complete else "stage-active"

    # ETA formatting
    if eta_sec > 60:
        eta_str = f"~{int(eta_sec // 60)}m {int(eta_sec % 60)}s"
    elif eta_sec > 0:
        eta_str = f"~{int(eta_sec)}s"
    else:
        eta_str = "Complete" if is_complete else "--"

    fps_str = f"{fps:.1f} fps" if fps > 0 else "--"

    # Card Render
    st.markdown(f"""
    <div class="progress-card">
        <div class="progress-header">
            <div>
                <span class="progress-title">📊 Live Processing Pipeline</span>
                <span class="batch-pill" style="margin-left:8px;">BATCH: {batch_id[:8]}</span>
            </div>
            <div>
                <span class="stage-badge {stage_class}">STAGE: {stage}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.progress(progress_val)

    # Stats strip
    st.markdown(f"""
        <div class="stat-strip">
            <div class="stat-box">
                <div class="stat-num">{processed} / {total}</div>
                <div class="stat-lbl">Processed</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#22c55e;">{processed - failed}</div>
                <div class="stat-lbl">Validated</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#fbbf24;">{duplicates}</div>
                <div class="stat-lbl">Duplicates</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#ef4444;">{failed}</div>
                <div class="stat-lbl">Corrupted / Failed</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" style="color:#38bdf8;">{fps_str}</div>
                <div class="stat-lbl">Throughput</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">{eta_str}</div>
                <div class="stat-lbl">ETA</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
