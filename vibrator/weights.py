"""Context-aware weight utilities for slider scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping

import numpy as np


@dataclass
class WeightConfig:
    base_weights: Mapping[str, float]
    segment_adjustments: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    context_features: Mapping[str, float] = field(default_factory=dict)


def normalize_weights(weights: MutableMapping[str, float]) -> Dict[str, float]:
    total = float(sum(max(w, 0.0) for w in weights.values()))
    if total == 0.0:
        raise ValueError("All weights are zero; cannot normalize.")
    return {name: max(value, 0.0) / total for name, value in weights.items()}


def contextualize_weights(
    base_weights: Mapping[str, float],
    segment: str | None = None,
    is_work_hours: bool | None = None,
    work_hour_multiplier: float = 1.25,
    work_hour_targets: Mapping[str, float] | None = None,
    segment_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> Dict[str, float]:
    """Return normalized weights factoring in segment and temporal context."""
    weights: Dict[str, float] = dict(base_weights)

    if segment and segment_overrides and segment in segment_overrides:
        for key, boost in segment_overrides[segment].items():
            weights[key] = weights.get(key, 0.0) * boost

    if is_work_hours and work_hour_targets:
        for key, target in work_hour_targets.items():
            weights[key] = weights.get(key, 0.0) * work_hour_multiplier * target

    return normalize_weights(weights)


def weighted_dot(embedding: np.ndarray, slider_vectors: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> Dict[str, float]:
    """Compute weighted dot products between user embedding and slider prototypes."""
    scores: Dict[str, float] = {}
    for name, vector in slider_vectors.items():
        scores[name] = float(np.dot(embedding, vector) * weights.get(name, 1.0))
    return scores
