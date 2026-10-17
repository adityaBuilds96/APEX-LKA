# APEX-Vision Core v4.0 • Environmental Stress-Testing Benchmark Report
**ISO 26262 ASIL-B Aligned Perception Robustness Assessment**
**Audit Timestamp:** `2026-10-17T07:11:09Z`
**Overall System Robustness Score:** `84.3 / 100.0`
**Most Severe Degradation Domain:** `AIRBORNE_DUST_SMOKE`

## 1. Multi-Condition Perception Matrix

| Environmental Condition | Samples | Lane mIoU | Det mAP@0.50 | Depth AbsRel | HOTA | Latency (ms) | Degradation | ASIL-B Status |
|---|---|---|---|---|---|---|---|---|
| `DAY_CLEAR` | 2000 | 0.825 | 0.810 | 0.085 | 0.740 | 21.5 | 0.0% (Ref) | 🟢 PASS |
| `NIGHT_LOW_LIGHT` | 1500 | 0.740 | 0.720 | 0.125 | 0.670 | 21.8 | -9.2% | 🟢 PASS |
| `DIRECT_SUN_GLARE` | 1200 | 0.685 | 0.660 | 0.155 | 0.610 | 22.1 | -16.3% | 🟢 PASS |
| `HEAVY_RAIN_SLURRY` | 1000 | 0.620 | 0.580 | 0.185 | 0.540 | 22.4 | -24.8% | 🟢 PASS |
| `AIRBORNE_DUST_SMOKE` | 800 | 0.510 | 0.480 | 0.245 | 0.460 | 23.0 | -35.9% | 🔴 VIOLATION |
| `DYNAMIC_TREE_SHADOWS` | 1400 | 0.710 | 0.700 | 0.135 | 0.650 | 21.6 | -12.1% | 🟢 PASS |
| `OFFROAD_UNSTRUCTURED` | 1800 | 0.670 | 0.730 | 0.140 | 0.680 | 22.0 | -11.7% | 🟢 PASS |

## 2. Operational Design Domain (ODD) Boundary Restrictions

- ⚠️ ODD Exclusion / Degraded Speed Limit: AIRBORNE_DUST_SMOKE (Lane mIoU 0.510 below ASIL-B threshold (0.55), Detection mAP@0.50 0.480 below ASIL-B threshold (0.5), Depth AbsRel 0.245 exceeds ASIL-B threshold (0.22), Tracking HOTA 0.460 below ASIL-B threshold (0.5))

## 3. Normative Safety Statement
> [!NOTE]
> Developed following an ISO 26262-style process; ASIL-B-aligned; not independently assessed.
> Operational profile target: driver-supervised SAE Level 2 vehicle guidance.
