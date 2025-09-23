"""Apply a context-aware penalty when users disrespect @tawny.

This version scales the penalty by recency and targets relevant sliders to avoid
collapsing all probabilities to ~0.001 while still applying a strong push-down.
"""
from __future__ import annotations

import math
from datetime import timedelta

from vibrator import (
    ChatMessage,
    InstructionalEncoder,
    SliderScorer,
    sample_recent_chat_actions,
)
from vibrator.utils import seed_everything, stable_now


SLIDERS = {
    "strategy": "Focuses on structure, goal-setting, and detailed next steps.",
    "support": "Offers emotional validation, encouragement, and gentle nudges.",
}

CHAT_ROOM = {
    "name": "Care Corner",
    "pitch": "Peer circle for checking in, celebrating wins, and regrouping after setbacks.",
}


def tawny_penalty_by_recency(actions, *, half_life_hours: float = 4.0, base_penalty: float = 4.0) -> float:
    """Return a penalty strength scaled by recency if user disrespected @tawny.

    More recent offenses apply a stronger penalty via exponential decay.
    """
    strength = 0.0
    for action in actions:
        text = action.content.lower()
        if "@tawny" in text and "corn dog" in text:
            age_hours = float(action.metadata.get("age_hours", 0.0)) if action.metadata else 0.0
            decay = math.exp(-math.log(2.0) * age_hours / max(half_life_hours, 1e-6))
            strength = max(strength, base_penalty * decay)
    return strength


def main() -> None:
    seed_everything(0)
    encoder = InstructionalEncoder()
    slider_vectors = dict(zip(SLIDERS, encoder.encode_items(SLIDERS.values())))
    scorer = SliderScorer(encoder, slider_vectors)

    now = stable_now()
    transcript = [
        ChatMessage(
            content="@tawny acts like some flipping corn dog in these meetings",
            timestamp=now - timedelta(minutes=3),
            user_id="user-456",
        ),
        ChatMessage(
            content="Anyway, who is helping with peer accountability?",
            timestamp=now - timedelta(minutes=1),
            user_id="user-456",
        ),
    ]

    actions = sample_recent_chat_actions(transcript, now=now, include_roles=("user",), max_messages=4)

    scores = scorer.score(actions, CHAT_ROOM["pitch"])

    penalty = tawny_penalty_by_recency(actions)
    if penalty > 0.0:
        # Heavier penalty on "support" (tone-sensitive) than on "strategy"
        per_slider_multiplier = {"support": 1.5, "strategy": 0.8}
        floor_logit = -4.0  # clamp to keep probabilities informative (~0.018)
        for slider_name, output in scores.items():
            shift = penalty * per_slider_multiplier.get(slider_name, 1.0)
            output.raw_score = max(output.raw_score - shift, floor_logit)
            output.probability = scorer._calibrated_probability(slider_name, output.raw_score)  # type: ignore[attr-defined]

    print(f"Chat room: {CHAT_ROOM['name']}")
    for slider_name, output in scores.items():
        print(f"  Slider: {slider_name}")
        print(f"    Raw score: {output.raw_score:.2f}")
        print(f"    Probability: {output.probability:.3f}")


if __name__ == "__main__":
    main()

