"""
src/training/reports.py
=======================
Multi-format evaluation report generator and failure case analyzer for APEX-LKA:
- JSON: results/metrics/evaluation_report.json
- Markdown: results/metrics/evaluation_report.md
- HTML: results/metrics/evaluation_report.html
- Failure categorization by time of day, weather, road type, and curvature.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

log = logging.getLogger(__name__)


def analyze_failures(
    frame_records: List[Dict[str, Any]],
    miou_threshold: float = 0.30,
    lane_iou_threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Categorizes severe and lane-specific perception failures by scenario attributes.
    """
    severe_failures: List[Dict[str, Any]] = []
    lane_failures: List[Dict[str, Any]] = []

    scenario_breakdown: Dict[str, Dict[str, int]] = {
        "time_of_day": defaultdict(int),
        "weather": defaultdict(int),
        "road_type": defaultdict(int),
        "curvature": defaultdict(int),
    }

    for rec in frame_records:
        miou = rec.get("frame_miou", 1.0)
        lane_iou = rec.get("frame_lane_iou", 1.0)
        meta = rec.get("metadata", {})

        is_severe = (miou < miou_threshold)
        is_lane_fail = (lane_iou < lane_iou_threshold)

        if is_severe or is_lane_fail:
            entry = {
                "name": rec.get("name", "unknown"),
                "miou": miou,
                "lane_iou": lane_iou,
                "time_of_day": meta.get("time_of_day", "day"),
                "weather": meta.get("weather", "clear"),
                "road_type": meta.get("road_type", "painted"),
                "curvature": meta.get("curvature", "straight"),
            }
            if is_severe:
                severe_failures.append(entry)
            if is_lane_fail:
                lane_failures.append(entry)

            scenario_breakdown["time_of_day"][entry["time_of_day"]] += 1
            scenario_breakdown["weather"][entry["weather"]] += 1
            scenario_breakdown["road_type"][entry["road_type"]] += 1
            scenario_breakdown["curvature"][entry["curvature"]] += 1

    recommendations: List[str] = []
    total_fails = len(severe_failures) + len(lane_failures)

    if total_fails == 0:
        recommendations.append("Perception quality satisfies all deployment thresholds with zero catastrophic frame failures.")
    else:
        # Check primary bottlenecks
        if scenario_breakdown["road_type"].get("unpainted", 0) > total_fails * 0.3:
            recommendations.append(
                "High failure rate detected on unpainted campus roads. Augment training with additional unpainted path masks."
            )
        if scenario_breakdown["weather"].get("shadowed", 0) > total_fails * 0.3:
            recommendations.append(
                "Tree canopy and dynamic shadows causing line drops. Increase RandomShadow and Contrast augmentation probability."
            )
        if scenario_breakdown["curvature"].get("curved", 0) > total_fails * 0.3:
            recommendations.append(
                "Curved paths exhibit boundary misalignments. Emphasize boundary loss weight in multi-task loss configuration."
            )
        if scenario_breakdown["time_of_day"].get("night", 0) > total_fails * 0.3:
            recommendations.append(
                "Night illumination drops confidence. Apply low-light gamma jitter in data augmentation pipeline."
            )

    if not recommendations:
        recommendations.append("Continue hard-negative mining on worst-ranked frames to improve boundary precision.")

    return {
        "total_evaluated_frames": len(frame_records),
        "severe_failures_count": len(severe_failures),
        "lane_failures_count": len(lane_failures),
        "severe_failures": severe_failures[:50],  # cap for report
        "lane_failures": lane_failures[:50],
        "scenario_breakdown": {k: dict(v) for k, v in scenario_breakdown.items()},
        "recommendations": recommendations,
    }


def generate_json_report(
    metrics_data: Dict[str, Any],
    failure_analysis: Dict[str, Any],
    latency_stats: Dict[str, Any],
    output_path: Union[str, Path] = "results/metrics/evaluation_report.json",
) -> Path:
    """Export complete structured evaluation report in JSON format."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics_data,
        "latency": latency_stats,
        "failure_analysis": failure_analysis,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("Saved JSON evaluation report: %s", out)
    return out


def generate_markdown_report(
    metrics_data: Dict[str, Any],
    failure_analysis: Dict[str, Any],
    latency_stats: Dict[str, Any],
    output_path: Union[str, Path] = "results/metrics/evaluation_report.md",
) -> Path:
    """Export executive Markdown evaluation report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    miou = metrics_data.get("mean_iou", 0.0)
    lane_miou = metrics_data.get("lane_mean_iou", 0.0)
    dice = metrics_data.get("mean_dice", 0.0)
    acc = metrics_data.get("overall_pixel_accuracy", 0.0)
    fps = latency_stats.get("fps", 0.0)
    p50 = latency_stats.get("p50_ms", 0.0)
    p95 = latency_stats.get("p95_ms", 0.0)

    per_class = metrics_data.get("per_class", {})

    lines = [
        "# APEX-LKA Model Evaluation & Perception Verification Report",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')} | **Evaluation Set Size:** {failure_analysis.get('total_evaluated_frames', 0)} frames",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Target | Result | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Mean IoU (mIoU)** | `>= 0.70` | **{miou:.4f}** | {'✅ PASS' if miou >= 0.70 else '⚠️ ATTENTION'} |",
        f"| **Lane Class mIoU** | `>= 0.65` | **{lane_miou:.4f}** | {'✅ PASS' if lane_miou >= 0.65 else '⚠️ ATTENTION'} |",
        f"| **Mean Dice Coefficient** | `>= 0.75` | **{dice:.4f}** | {'✅ PASS' if dice >= 0.75 else '⚠️ ATTENTION'} |",
        f"| **Overall Pixel Accuracy** | `>= 0.90` | **{acc:.4f}** | {'✅ PASS' if acc >= 0.90 else '⚠️ ATTENTION'} |",
        f"| **Inference Latency (p50)** | `<= 30ms` | **{p50:.1f} ms ({fps:.1f} FPS)** | {'✅ REAL-TIME' if fps >= 30 else '⚠️ SUB-30 FPS'} |",
        "",
        "## 2. Per-Class Perception Breakdown",
        "",
        "| Class Name | IoU | Dice (F1) | Precision | Recall | Pixel Acc | Pixel Share |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for c_name, vals in per_class.items():
        lines.append(
            f"| **{c_name}** | {vals.get('iou', 0.0):.4f} | {vals.get('dice', 0.0):.4f} | "
            f"{vals.get('precision', 0.0):.4f} | {vals.get('recall', 0.0):.4f} | "
            f"{vals.get('pixel_accuracy', 0.0):.4f} | {vals.get('pixel_fraction', 0.0):.2%} |"
        )

    lines += [
        "",
        "## 3. Inference Latency & Efficiency",
        "",
        f"- **Mean Latency:** {latency_stats.get('mean_ms', 0.0):.2f} ms",
        f"- **Median Latency (p50):** {p50:.2f} ms",
        f"- **95th Percentile (p95):** {p95:.2f} ms",
        f"- **99th Percentile (p99):** {latency_stats.get('p99_ms', 0.0):.2f} ms",
        f"- **Throughput:** {fps:.1f} FPS (Target: Jetson Orin Nano >= 30 FPS)",
        "",
        "## 4. Failure Analysis & Diagnostics",
        "",
        f"- **Severe Failures (mIoU < 0.30):** {failure_analysis.get('severe_failures_count', 0)} frames",
        f"- **Lane-Specific Failures (Lane IoU < 0.20):** {failure_analysis.get('lane_failures_count', 0)} frames",
        "",
        "### Scenario Breakdown",
        "",
    ]

    for cat_name, counts in failure_analysis.get("scenario_breakdown", {}).items():
        lines.append(f"**{cat_name.replace('_', ' ').title()}:**")
        for k, v in counts.items():
            lines.append(f"- {k}: {v} frames")
        lines.append("")

    lines += [
        "## 5. Actionable Engineering Recommendations",
        "",
    ]
    for rec in failure_analysis.get("recommendations", []):
        lines.append(f"- 💡 {rec}")

    lines.append("")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    log.info("Saved Markdown evaluation report: %s", out)
    return out


def generate_html_report(
    metrics_data: Dict[str, Any],
    failure_analysis: Dict[str, Any],
    latency_stats: Dict[str, Any],
    output_path: Union[str, Path] = "results/metrics/evaluation_report.html",
) -> Path:
    """Export interactive, styled HTML evaluation report."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    miou = metrics_data.get("mean_iou", 0.0)
    lane_miou = metrics_data.get("lane_mean_iou", 0.0)
    dice = metrics_data.get("mean_dice", 0.0)
    fps = latency_stats.get("fps", 0.0)
    p50 = latency_stats.get("p50_ms", 0.0)

    rows_html = ""
    for c_name, vals in metrics_data.get("per_class", {}).items():
        rows_html += f"""
        <tr>
            <td style="font-weight:600; color:#f1f5f9;">{c_name}</td>
            <td style="color:#38bdf8;">{vals.get('iou', 0.0):.4f}</td>
            <td style="color:#22c55e;">{vals.get('dice', 0.0):.4f}</td>
            <td>{vals.get('precision', 0.0):.4f}</td>
            <td>{vals.get('recall', 0.0):.4f}</td>
            <td>{vals.get('pixel_accuracy', 0.0):.4f}</td>
            <td>{vals.get('pixel_fraction', 0.0):.2%}</td>
        </tr>
        """

    rec_items = "".join(f"<li>{r}</li>" for r in failure_analysis.get("recommendations", []))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>APEX-LKA Evaluation Report</title>
    <style>
        body {{ background-color: #070b12; color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 30px; }}
        h1, h2, h3 {{ color: #f1f5f9; font-weight: 700; }}
        .header {{ border-bottom: 1px solid rgba(56, 189, 248, 0.2); padding-bottom: 15px; margin-bottom: 25px; }}
        .badge {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 10px; border-radius: 4px; font-size: 0.85rem; font-family: monospace; }}
        .cards {{ display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap; }}
        .card {{ background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 20px; flex: 1; min-width: 180px; text-align: center; }}
        .card-num {{ font-size: 2rem; font-weight: 800; color: #38bdf8; font-family: monospace; }}
        .card-label {{ font-size: 0.80rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.1em; margin-top: 6px; }}
        table {{ width: 100%; border-collapse: collapse; background: rgba(15, 23, 42, 0.6); border-radius: 8px; overflow: hidden; margin-bottom: 30px; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid rgba(56, 189, 248, 0.1); font-size: 0.90rem; }}
        th {{ background: rgba(30, 41, 59, 0.8); color: #94a3b8; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.08em; }}
        .rec-box {{ background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 8px; padding: 20px; }}
        .rec-box ul {{ margin: 0; padding-left: 20px; color: #cbd5e1; line-height: 1.6; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 APEX-LKA Perception Evaluation Report</h1>
        <span class="badge">EVALUATION TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}</span>
        <span class="badge">FRAMES: {failure_analysis.get('total_evaluated_frames', 0)}</span>
    </div>

    <div class="cards">
        <div class="card">
            <div class="card-num">{miou:.4f}</div>
            <div class="card-label">Mean IoU (mIoU)</div>
        </div>
        <div class="card">
            <div class="card-num">{lane_miou:.4f}</div>
            <div class="card-label">Lane Class IoU</div>
        </div>
        <div class="card">
            <div class="card-num">{dice:.4f}</div>
            <div class="card-label">Mean Dice</div>
        </div>
        <div class="card">
            <div class="card-num">{p50:.1f} ms</div>
            <div class="card-label">Inference Latency</div>
        </div>
        <div class="card">
            <div class="card-num">{fps:.1f}</div>
            <div class="card-label">Inference FPS</div>
        </div>
    </div>

    <h2>Per-Class Segmentation Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Class</th>
                <th>IoU</th>
                <th>Dice (F1)</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>Pixel Accuracy</th>
                <th>Dataset Share</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <h2>Actionable Recommendations</h2>
    <div class="rec-box">
        <ul>
            {rec_items}
        </ul>
    </div>
</body>
</html>
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)

    log.info("Saved HTML evaluation report: %s", out)
    return out
