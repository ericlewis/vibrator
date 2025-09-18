from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence, cast

import numpy as np

from vibrator.actions import UserAction
from vibrator.pipeline import SliderScorer
from vibrator.profile import weighted_user_embedding


class StubEncoder:
    def __init__(self) -> None:
        self.model = SimpleNamespace(get_sentence_embedding_dimension=lambda: 3)

    def encode_actions(self, actions: Sequence[UserAction]) -> np.ndarray:
        mapping = {
            "write": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "read": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "react": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        vectors = [mapping.get(action.action_type, np.zeros(3, dtype=np.float32)) for action in actions]
        return np.stack(vectors, axis=0)

    def encode_items(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([[0.6, 0.8, 0.0] for _ in texts], dtype=np.float32)


class ConstantCalibrator:
    def __init__(self, value: float) -> None:
        self.value = value

    def transform(self, scores: Sequence[float]) -> np.ndarray:
        return np.full(len(list(scores)), self.value, dtype=np.float32)


def _build_actions() -> list[UserAction]:
    return [
        UserAction("write", "Drafted plan", weight=2.0, metadata={"age_hours": 1}),
        UserAction("read", "Reviewed async guide", weight=1.0, metadata={"age_hours": 10}),
    ]


def test_weighted_user_embedding_uses_action_weights():
    encoder = StubEncoder()
    actions = _build_actions()
    embedding = weighted_user_embedding(encoder, actions)
    np.testing.assert_allclose(embedding, np.array([2.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float32))


def test_slider_scorer_features_and_calibration():
    encoder = StubEncoder()
    actions = _build_actions()
    slider_vectors = {
        "professional": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "personal": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }

    calibrators = {"personal": ConstantCalibrator(0.42)}
    scorer = SliderScorer(encoder, slider_vectors, calibrators=calibrators)

    outputs = scorer.score(actions, "Weekly sync tips")

    professional = outputs["professional"]
    expected_recency = np.exp(-np.log(2.0) * np.array([1.0, 10.0]) / 12.0).max()
    expected_features = {
        "user_to_slider": 2.0 / 3.0,
        "item_to_slider": 0.6,
        "user_item_alignment": (2.0 / 3.0) * 0.6 + (1.0 / 3.0) * 0.8,
        "recency": expected_recency,
    }

    for key, value in expected_features.items():
        assert abs(professional.features[key] - value) < 1e-6

    assert abs(professional.weights["user_to_slider"] + professional.weights["item_to_slider"] + professional.weights["user_item_alignment"] + professional.weights["recency"] - 1.0) < 1e-6

    # Logistic probability for professional slider should be in (0,1)
    assert 0.0 < professional.probability < 1.0

    personal = outputs["personal"]
    assert abs(personal.probability - 0.42) < 1e-6
    assert personal.features.keys() == professional.features.keys()
