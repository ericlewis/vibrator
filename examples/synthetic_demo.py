"""Run the slider pipeline on synthetic data."""
from __future__ import annotations

from vibrator import InstructionalEncoder, SliderScorer
from vibrator.synthetic import synthetic_actions, synthetic_item_corpus, synthetic_slider_texts


def main() -> None:
    encoder = InstructionalEncoder()

    slider_texts = synthetic_slider_texts(limit=3)
    slider_vectors = dict(zip(slider_texts.keys(), encoder.encode_items(slider_texts.values())))

    actions = synthetic_actions(count=5, seed=13)
    item_text = synthetic_item_corpus(limit=1)[0]

    scorer = SliderScorer(encoder, slider_vectors)
    scores = scorer.score(actions, item_text, context={"segment": "synthetic"})

    for name, output in scores.items():
        print(f"Slider: {name}")
        print(f"  Probability: {output.probability:.3f}")
        print(f"  Features: {output.features}")
        print()


if __name__ == "__main__":
    main()
