"""End-to-end slider scoring utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from .actions import UserAction
from .calibration import IsotonicCalibrator, TemperatureCalibrator
from .normalization import l2_normalize
from .profile import weighted_user_embedding

if TYPE_CHECKING:  # pragma: no cover - avoids heavy import at runtime
    from .encoder import InstructionalEncoder


def _recency_feature(actions: Sequence[UserAction], half_life_hours: float = 12.0) -> float:
    if not actions:
        return 0.0
    ages = []
    for action in actions:
        age = float(action.metadata.get("age_hours", 0.0)) if action.metadata else 0.0
        ages.append(age)
    ages_array = np.asarray(ages, dtype=np.float32)
    decays = np.exp(-np.log(2.0) * ages_array / half_life_hours)
    return float(decays.max())


def _sigmoid(value: float) -> float:
    # Ensure a native Python float to avoid np.float64 leaking into outputs
    return float(1.0 / (1.0 + np.exp(-value)))


@dataclass
class SliderOutput:
    raw_score: float
    probability: float
    features: Dict[str, float]
    weights: Dict[str, float]


class SliderScorer:
    """Combine user/item embeddings into calibrated personalization sliders."""

    def __init__(
        self,
        encoder: 'InstructionalEncoder',
        slider_vectors: Mapping[str, np.ndarray],
        base_feature_weights: Mapping[str, float] | None = None,
        calibrators: Mapping[str, TemperatureCalibrator | IsotonicCalibrator] | None = None,
        feature_overrides_by_segment: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.encoder = encoder
        self.slider_vectors = {
            name: l2_normalize(np.asarray(vector).reshape(1, -1))[0] for name, vector in slider_vectors.items()
        }
        self.base_feature_weights = dict(base_feature_weights or {
            "user_to_slider": 0.35,
            "item_to_slider": 0.20,
            "user_item_alignment": 0.30,
            "recency": 0.15,
        })
        self.calibrators = calibrators or {}
        self.feature_overrides_by_segment = feature_overrides_by_segment or {}

    def _feature_weights(self, context: Mapping[str, object] | None) -> Dict[str, float]:
        weights: Dict[str, float] = dict(self.base_feature_weights)
        if not context:
            return self._normalize(weights)

        segment = context.get("segment") if isinstance(context.get("segment"), str) else None
        if segment and segment in self.feature_overrides_by_segment:
            for feature, multiplier in self.feature_overrides_by_segment[segment].items():
                weights[feature] = weights.get(feature, 0.0) * multiplier

        if context.get("is_work_hours"):
            weights["item_to_slider"] = weights.get("item_to_slider", 0.0) * 1.25

        if context.get("boost_feature"):
            feature = str(context["boost_feature"])
            weights[feature] = weights.get(feature, 0.0) * 1.5

        return self._normalize(weights)

    @staticmethod
    def _normalize(weights: MutableMapping[str, float]) -> Dict[str, float]:
        positive = {k: max(v, 0.0) for k, v in weights.items()}
        total = sum(positive.values())
        if total == 0.0:
            raise ValueError("Feature weights collapsed to zero.")
        return {k: v / total for k, v in positive.items()}

    def _compute_features(
        self,
        actions: Sequence[UserAction],
        user_embedding: np.ndarray,
        item_embedding: np.ndarray,
        slider_vector: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "user_to_slider": float(np.dot(user_embedding, slider_vector)),
            "item_to_slider": float(np.dot(item_embedding, slider_vector)),
            "user_item_alignment": float(np.dot(user_embedding, item_embedding)),
            "recency": _recency_feature(actions),
        }

    def score(
        self,
        actions: Sequence[UserAction],
        item_text: str,
        context: Optional[Mapping[str, object]] = None,
        weight_temperature: float = 1.0,
    ) -> Dict[str, SliderOutput]:
        if not actions:
            raise ValueError("SliderScorer.score requires at least one user action.")

        user_embedding = weighted_user_embedding(self.encoder, actions, weight_temperature=weight_temperature)
        item_embedding = self.encoder.encode_items([item_text])[0]

        weights = self._feature_weights(context)
        outputs: Dict[str, SliderOutput] = {}
        for name, slider_vector in self.slider_vectors.items():
            features = self._compute_features(actions, user_embedding, item_embedding, slider_vector)
            raw_score = sum(weights.get(feature, 0.0) * value for feature, value in features.items())
            probability = self._calibrated_probability(name, raw_score)
            outputs[name] = SliderOutput(
                raw_score=raw_score,
                probability=probability,
                features=features,
                weights=weights,
            )
        return outputs

    def _calibrated_probability(self, slider_name: str, score: float) -> float:
        calibrator = self.calibrators.get(slider_name)
        if calibrator:
            return float(calibrator.transform([score])[0])
        return _sigmoid(score)
