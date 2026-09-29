"""
dashboard/components/thumbnail_grid.py
======================================
Responsive thumbnail grid component with instant preview cards, format badges,
status pills, and virtual pagination (Section 3.2 #3 & Section 3.3).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.dataset.thumbnail_generator import ThumbnailGenerator


THUMB_GRID_CSS = """
<style>
.thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
    margin-top: 14px;
    margin-bottom: 20px;
}
.thumb-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    display: flex;
    flex-direction: column;
}
.thumb-card:hover {
    border-color: rgba(56, 189, 248, 0.5);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    transform: translateY(-2px);
}
.thumb-card-done { border: 1px solid rgba(34, 197, 94, 0.5); }
.thumb-card-processing { border: 1px solid rgba(56, 189, 248, 0.6); box-shadow: 0 0 12px rgba(56, 189, 248, 0.2); }
.thumb-card-failed { border: 1px solid rgba(239, 68, 68, 0.6); }
.thumb-card-queued { border: 1px solid rgba(148, 163, 184, 0.2); }

.thumb-img {
    width: 100%;
    height: 110px;
    object-fit: cover;
    background: #070b12;
    display: block;
}
.thumb-info {
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.thumb-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #f1f5f9;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.thumb-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.format-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: #38bdf8;
    background: rgba(56, 189, 248, 0.12);
    padding: 1px 5px;
    border-radius: 4px;
}
.status-badge {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
}
.size-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #94a3b8;
}
</style>
"""


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"


def render_thumbnail_grid(
    items: List[Dict[str, Any]],
    thumbnail_generator: Optional[ThumbnailGenerator] = None,
    items_per_page: int = 24,
    key_prefix: str = "thumb_grid",
) -> None:
    """
    Renders responsive grid of thumbnail cards with virtual pagination trigger.

    Args:
        items: List of dicts with keys:
            - 'name': str
            - 'path': Path or str
            - 'status': 'queued' | 'processing' | 'done' | 'failed'
            - 'size_bytes': int
            - 'format': Optional[str]
            - 'data_uri': Optional[str] (Base64 JPEG data URI)
        thumbnail_generator: Optional ThumbnailGenerator instance
        items_per_page: Number of cards to display per virtual batch
        key_prefix: Unique key namespace for Streamlit state
    """
    if not items:
        return

    st.markdown(THUMB_GRID_CSS, unsafe_allow_html=True)
    thumb_gen = thumbnail_generator or ThumbnailGenerator()

    # Track visible item count in session_state for virtual pagination
    page_state_key = f"{key_prefix}_visible_count"
    if page_state_key not in st.session_state:
        st.session_state[page_state_key] = min(items_per_page, len(items))

    visible_count = st.session_state[page_state_key]
    displayed_items = items[:visible_count]

    # Generate HTML cards
    cards_html = ['<div class="thumb-grid">']

    for item in displayed_items:
        name = str(item.get("name", "frame"))
        status = str(item.get("status", "queued")).lower()
        size_bytes = int(item.get("size_bytes", 0))
        fmt = str(item.get("format", Path(name).suffix.lstrip(".").upper() or "IMG")).upper()

        # Status badge & style
        if status in {"done", "success", "validated"}:
            status_html = '<span class="status-badge" style="color:#22c55e;">✅ Done</span>'
            card_class = "thumb-card-done"
        elif status in {"processing", "running"}:
            status_html = '<span class="status-badge" style="color:#38bdf8;">🔄 Active</span>'
            card_class = "thumb-card-processing"
        elif status in {"failed", "corrupted", "error"}:
            status_html = '<span class="status-badge" style="color:#ef4444;">❌ Failed</span>'
            card_class = "thumb-card-failed"
        else:
            status_html = '<span class="status-badge" style="color:#94a3b8;">⏳ Queued</span>'
            card_class = "thumb-card-queued"

        # Image source
        data_uri = item.get("data_uri")
        if not data_uri and "path" in item:
            try:
                data_uri = thumb_gen.get_thumbnail_data_uri(Path(item["path"]))
            except Exception:
                data_uri = ""

        img_tag = (
            f'<img class="thumb-img" src="{data_uri}" alt="{name}"/>'
            if data_uri
            else '<div class="thumb-img" style="display:flex;align-items:center;justify-content:center;color:#64748b;font-size:0.75rem;">No Preview</div>'
        )

        card = f"""
        <div class="thumb-card {card_class}">
            {img_tag}
            <div class="thumb-info">
                <div class="thumb-name" title="{name}">{name}</div>
                <div class="thumb-meta">
                    <span class="format-badge">{fmt}</span>
                    <span class="size-badge">{format_bytes(size_bytes)}</span>
                </div>
                <div style="margin-top:2px;">
                    {status_html}
                </div>
            </div>
        </div>
        """
        cards_html.append(card)

    cards_html.append("</div>")
    st.markdown("".join(cards_html), unsafe_allow_html=True)

    # Virtual Scroll / Pagination Trigger
    if visible_count < len(items):
        remaining = len(items) - visible_count
        step = min(items_per_page, remaining)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(f"⬇️ Load More (+{step} of {remaining} remaining)", use_container_width=True, key=f"{key_prefix}_load_more"):
                st.session_state[page_state_key] += step
                st.rerun()
