"""
src/training/samplers.py
=========================
Class-Balanced and Sequence-Aware Sampling for LaneSegNet Training.

Features
--------
1. Class-Balanced Sampling (Objective 4):
   - Computes lane marking pixel fraction per frame: (pixels(2) + pixels(3)) / (H * W)
   - Sampling weight = 1.0 + 1.5 * (lane_fraction / mean_lane_fraction)
   - Guarantees frames with clear lane markings are sampled more frequently,
     preventing the model from defaulting to background prediction.
2. Sequence-Aware Batching (Objective 5):
   - Groups dataset frames by recording session / video ID.
   - Stratifies batch construction so that each batch draws frames from
     different sessions, breaking temporal correlation and maximizing gradient diversity.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Sampler


def extract_session_id(path: Union[str, Path]) -> str:
    """
    Extract a recording session ID from an image/mask path.

    Heuristics:
    1. Parent directory if structured as data/train/images/session_01/frame_001.jpg
    2. Prefix pattern: 'session_01_frame_001.jpg' -> 'session_01'
    3. Prefix before '_frame_' or '_f'
    4. Fallback: hash of filename or 'session_default'
    """
    p = Path(path)
    stem = p.stem

    # Check for session pattern: session_XX or rec_XX
    match = re.match(r"^(session[_\-\d]+|rec[_\-\d]+|[A-Za-z]+_\d+)", stem)
    if match:
        return match.group(1)

    if "_frame_" in stem:
        return stem.split("_frame_")[0]

    if p.parent.name not in ("images", "masks", "train", "val", "test"):
        return p.parent.name

    return "session_00"


def compute_lane_pixel_fractions(
    mask_paths_or_dataset: Union[Sequence[Path], Sequence[str], object],
) -> np.ndarray:
    """
    Compute lane marking pixel fractions for all masks in a dataset.
    Lane classes: 2 (left lane) and 3 (right lane).
    """
    fractions: List[float] = []

    # If dataset object passed
    if hasattr(mask_paths_or_dataset, "_pairs"):
        pairs = getattr(mask_paths_or_dataset, "_pairs")
        mask_paths = [p[1] for p in pairs]
    elif isinstance(mask_paths_or_dataset, (list, tuple)):
        mask_paths = mask_paths_or_dataset
    else:
        raise ValueError(f"Unsupported mask container: {type(mask_paths_or_dataset)}")

    for mp in mask_paths:
        raw_mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            fractions.append(0.0)
            continue
        total_px = float(raw_mask.size)
        lane_px = float(np.count_nonzero((raw_mask == 2) | (raw_mask == 3)))
        fractions.append(lane_px / max(total_px, 1.0))

    return np.array(fractions, dtype=np.float32)


def compute_class_balanced_weights(lane_fractions: np.ndarray) -> np.ndarray:
    """
    Calculate sample weights = 1.0 + 1.5 * (lane_fraction / mean_lane_fraction).
    """
    if len(lane_fractions) == 0:
        return np.array([], dtype=np.float32)

    mean_frac = float(np.mean(lane_fractions))
    if mean_frac <= 1e-7:
        return np.ones_like(lane_fractions, dtype=np.float32)

    weights = 1.0 + 1.5 * (lane_fractions / mean_frac)
    return weights.astype(np.float32)


class ClassBalancedSampler(Sampler[int]):
    """
    Samples frames with probability proportional to lane pixel density.
    """

    def __init__(
        self,
        dataset: object,
        lane_fractions: Optional[np.ndarray] = None,
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ) -> None:
        if lane_fractions is None:
            lane_fractions = compute_lane_pixel_fractions(dataset)
        self.weights = compute_class_balanced_weights(lane_fractions)
        self.num_samples = num_samples if num_samples is not None else len(self.weights)
        self.replacement = replacement
        self._torch_weights = torch.as_tensor(self.weights, dtype=torch.double)

    def __iter__(self) -> Iterator[int]:
        if len(self.weights) == 0:
            return iter([])
        sampled_indices = torch.multinomial(
            self._torch_weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
        )
        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class WeightedSessionSampler(Sampler[List[int]]):
    """
    BatchSampler that enforces sequence-aware batching and class-balanced frame weighting.

    Guarantees:
    - Frames within any single batch are drawn from distinct video sessions.
    - Frames with higher lane marking density are sampled with higher probability.
    """

    def __init__(
        self,
        dataset: object,
        batch_size: int = 8,
        session_ids: Optional[List[str]] = None,
        lane_fractions: Optional[np.ndarray] = None,
        drop_last: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.drop_last = drop_last

        n = len(dataset)
        # 1. Resolve session IDs per item
        if session_ids is not None:
            self.session_ids = session_ids
        elif hasattr(dataset, "_pairs"):
            self.session_ids = [extract_session_id(p[0]) for p in getattr(dataset, "_pairs")]
        else:
            self.session_ids = [f"session_{i % 4}" for i in range(n)]

        # 2. Compute weights
        if lane_fractions is None:
            lane_fractions = compute_lane_pixel_fractions(dataset)
        self.weights = compute_class_balanced_weights(lane_fractions)

        # 3. Group indices by session
        self.session_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, s_id in enumerate(self.session_ids):
            self.session_to_indices[s_id].append(idx)

        self.sessions = list(self.session_to_indices.keys())
        self.total_samples = n

    def __iter__(self) -> Iterator[List[int]]:
        # For each session, create weighted permutation or sampling order
        session_pools: Dict[str, List[int]] = {}
        for s_id, indices in self.session_to_indices.items():
            s_weights = self.weights[indices]
            s_weights = s_weights / (s_weights.sum() + 1e-8)
            # Shuffle indices according to weights
            shuffled = np.random.choice(
                indices,
                size=len(indices),
                replace=False,
                p=s_weights,
            ).tolist()
            session_pools[s_id] = shuffled

        available_sessions = [s for s in self.sessions if len(session_pools[s]) > 0]
        batch: List[int] = []

        while len(available_sessions) > 0:
            # Randomize session order for the batch
            np.random.shuffle(available_sessions)

            # Draw one sample per session up to batch_size
            for s_id in available_sessions:
                idx = session_pools[s_id].pop(0)
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            # Refresh list of sessions that still have frames left
            available_sessions = [s for s in self.sessions if len(session_pools[s]) > 0]

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        return math.ceil(self.total_samples / float(self.batch_size))
