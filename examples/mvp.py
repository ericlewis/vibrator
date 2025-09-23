"""Minimal example showing how to score sliders."""
from __future__ import annotations

from vibrator import InstructionalEncoder, SliderScorer, UserAction
from vibrator.utils import seed_everything, friendly_round_features


SLIDER_DEFINITIONS = {
    "professional": "The user is motivated by professional growth and productivity.",
    "personal": "The user seeks supportive, empathetic conversations.",
    "playful": "The user enjoys witty, humorous banter.",
}


def main() -> None:
    seed_everything(0)
    encoder = InstructionalEncoder()

    slider_vectors = dict(
        zip(SLIDER_DEFINITIONS.keys(), encoder.encode_items(SLIDER_DEFINITIONS.values()))
    )

    actions = [
        UserAction("write", "Really need advice on managing my team roadmap this quarter", weight=1.2, metadata={"age_hours": 2}),
        UserAction("react", "Gave thumbs up to a joke about debugging", weight=0.8, metadata={"age_hours": 5}),
        UserAction("read", "Read an article on async Python patterns", weight=0.6, metadata={"age_hours": 1}),
    ]

    scorer = SliderScorer(encoder, slider_vectors)

    context = {"segment": "power_user", "is_work_hours": True}
    item_text = "Weekly standup tips to keep remote teams aligned"

    scores = scorer.score(actions, item_text, context=context)

    for name, output in scores.items():
        print(f"Slider: {name}")
        print(f"  Raw score: {output.raw_score:.3f}")
        print(f"  Probability: {output.probability:.3f}")
        print(f"  Features: {friendly_round_features(output.features)}")
        print()


if __name__ == "__main__":
    main()
