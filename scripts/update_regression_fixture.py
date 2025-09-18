"""Recompute expected probabilities for the regression fixture."""
from __future__ import annotations

import json
from pathlib import Path

from vibrator import InstructionalEncoder, SliderScorer, UserAction


def main() -> None:
    fixture_path = Path(__file__).parent.parent / "fixtures" / "regression_logs.json"
    if not fixture_path.exists():
        raise SystemExit("fixtures/regression_logs.json not found.")

    payload = json.loads(fixture_path.read_text())
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

    payload["expected_probabilities"] = {
        name: round(result.probability, 6) for name, result in outputs.items()
    }

    fixture_path.write_text(json.dumps(payload, indent=2) + "\n")
    for name, probability in payload["expected_probabilities"].items():
        print(f"{name}: {probability}")


if __name__ == "__main__":
    main()
