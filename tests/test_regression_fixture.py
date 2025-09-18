import json
from pathlib import Path

import numpy as np
import pytest

from vibrator import InstructionalEncoder, SliderScorer, UserAction


FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "regression_logs.json"


@pytest.mark.integration
def test_regression_fixture_probabilities():
    if not FIXTURE_PATH.exists():
        pytest.skip("regression fixture not provided")

    payload = json.loads(FIXTURE_PATH.read_text())

    encoder = InstructionalEncoder()
    slider_vectors = dict(
        zip(
            payload["slider_texts"].keys(),
            encoder.encode_items(payload["slider_texts"].values()),
        )
    )

    actions = [
        UserAction(
            entry["action_type"],
            entry["content"],
            weight=float(entry.get("weight", 1.0)),
            metadata=entry.get("metadata", {}),
        )
        for entry in payload["actions"]
    ]

    scorer = SliderScorer(encoder, slider_vectors)
    outputs = scorer.score(actions, payload["item"], context=payload.get("context"))

    expected = payload.get("expected_probabilities", {})
    for slider_name, expected_value in expected.items():
        assert slider_name in outputs
        probability = outputs[slider_name].probability
        assert np.isclose(probability, expected_value, atol=1e-4), (
            f"Probability drifted for {slider_name}: got {probability}, expected {expected_value}"
        )
