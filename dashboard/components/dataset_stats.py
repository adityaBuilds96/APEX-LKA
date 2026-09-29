"""
dashboard/components/dataset_stats.py
=====================================
Dataset statistics visualization cards and class distribution charts (Section 5).

Provides:
  - Metric summary cards with glassmorphic styling
  - Train/Val/Test partition ratio bars
  - 4-Class pixel frequency distribution visualizer
  - Automated loss function class weight recommendations:
      weights = 1.0 / (class_frequency * num_classes)
  - Class imbalance warnings and dataset health indicators
"""

from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from src.dataset.annotation_prep import CLASS_DEFINITIONS


def render_dataset_stats_cards(
    total_frames: int,
    annotated_masks: int,
    candidate_masks: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> None:
    """
    Renders top KPI metrics strip.
    """
    pct_coverage = round((annotated_masks / max(1, total_frames)) * 100.0, 1)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div style="background:rgba(15, 23, 42, 0.75); border:1px solid rgba(56, 189, 248, 0.15);
                    border-radius:10px; padding:14px; backdrop-filter:blur(12px);">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Total Dataset Frames</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#f1f5f9; margin:4px 0;">{total_frames}</div>
            <div style="font-family:'Inter', sans-serif; font-size:0.75rem; color:#38bdf8;">Extracted / Staged</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:rgba(15, 23, 42, 0.75); border:1px solid rgba(34, 197, 94, 0.2);
                    border-radius:10px; padding:14px; backdrop-filter:blur(12px);">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Verified Masks</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#22c55e; margin:4px 0;">{annotated_masks}</div>
            <div style="font-family:'Inter', sans-serif; font-size:0.75rem; color:#94a3b8;">{pct_coverage}% coverage</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="background:rgba(15, 23, 42, 0.75); border:1px solid rgba(56, 189, 248, 0.2);
                    border-radius:10px; padding:14px; backdrop-filter:blur(12px);">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Pre-Annotations</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#38bdf8; margin:4px 0;">{candidate_masks}</div>
            <div style="font-family:'Inter', sans-serif; font-size:0.75rem; color:#94a3b8;">Classical CV Candidates</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        total_split = train_count + val_count + test_count
        st.markdown(f"""
        <div style="background:rgba(15, 23, 42, 0.75); border:1px solid rgba(251, 191, 36, 0.2);
                    border-radius:10px; padding:14px; backdrop-filter:blur(12px);">
            <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#94a3b8; text-transform:uppercase;">Split Partitions</div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.6rem; font-weight:700; color:#fbbf24; margin:4px 0;">{total_split}</div>
            <div style="font-family:'Inter', sans-serif; font-size:0.75rem; color:#94a3b8;">{train_count} tr · {val_count} va · {test_count} te</div>
        </div>
        """, unsafe_allow_html=True)


def render_class_distribution_section(
    class_pixel_counts: Optional[Dict[int, int]] = None,
) -> None:
    """
    Renders the 4-class semantic distribution and training loss weight calculator.
    """
    st.markdown("""
    <div style="font-family:'JetBrains Mono', monospace; font-size:0.75rem; color:#38bdf8; font-weight:700; margin-bottom:8px; text-transform:uppercase;">
        4-Class Semantic Distribution & Recommended Loss Weights
    </div>
    """, unsafe_allow_html=True)

    # Typical reference distributions for road segmentation
    default_counts = {0: 623000, 1: 312000, 2: 34000, 3: 31000}
    counts = class_pixel_counts if (class_pixel_counts and sum(class_pixel_counts.values()) > 0) else default_counts
    total_px = max(1, sum(counts.values()))
    num_classes = 4

    for cdef in CLASS_DEFINITIONS:
        cid = cdef["id"]
        cname = cdef["name"].replace("_", " ").title()
        hex_c = cdef["hex"]
        cnt = counts.get(cid, 0)
        fraction = cnt / total_px
        pct = fraction * 100.0

        # Suggested class weight = 1.0 / (fraction * num_classes)
        suggested_weight = round(1.0 / max(0.0001, fraction * num_classes), 2)

        st.markdown(f"""
        <div style="background:rgba(15, 23, 42, 0.6); border:1px solid rgba(56, 189, 248, 0.12);
                    border-left:4px solid {hex_c}; border-radius:6px; padding:10px 14px; margin-bottom:8px;
                    display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-family:'JetBrains Mono', monospace; font-weight:700; color:#f1f5f9; font-size:0.85rem;">
                    Class {cid}: {cname}
                </span>
                <div style="font-family:'Inter', sans-serif; font-size:0.75rem; color:#94a3b8;">
                    {cdef['description']}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'JetBrains Mono', monospace; font-size:0.85rem; font-weight:700; color:{hex_c};">
                    {pct:.1f}% ({cnt:,} px)
                </div>
                <div style="font-family:'JetBrains Mono', monospace; font-size:0.7rem; color:#94a3b8;">
                    Loss Weight: <b style="color:#38bdf8;">{suggested_weight}x</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
