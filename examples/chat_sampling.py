"""Example showing how to sample recent chat messages into slider scoring."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vibrator import (
    ChatMessage,
    InstructionalEncoder,
    SliderScorer,
    sample_recent_chat_actions,
)


SLIDER_DEFINITIONS = {
    "mentorship": "The user is looking for mentorship and structured guidance.",
    "brainstorm": "The user wants playful idea generation and rapid iteration.",
    "empathy": "The user responds well to supportive, empathetic tone.",
}


def main() -> None:
    encoder = InstructionalEncoder()
    slider_vectors = dict(zip(SLIDER_DEFINITIONS, encoder.encode_items(SLIDER_DEFINITIONS.values())))

    now = datetime.now(timezone.utc)
    transcript = [
        ChatMessage(
            content="Hey! I'm trying to keep my study group motivated for finals.",
            timestamp=now - timedelta(hours=4, minutes=15),
            user_id="user-123",
        ),
        ChatMessage(
            content="Maybe set up a shared calendar with milestones?",
            timestamp=now - timedelta(hours=4),
            role="assistant",
        ),
        ChatMessage(
            content="Yeah, but I also need ways to cheer them up when they're stressed.",
            timestamp=now - timedelta(hours=1, minutes=50),
            user_id="user-123",
        ),
        ChatMessage(
            content="Totally agree—stressed teams need space to vent.",
            timestamp=now - timedelta(minutes=25),
            user_id="user-123",
        ),
    ]

    sampled_actions = sample_recent_chat_actions(
        transcript,
        now=now,
        max_messages=3,
        include_roles=("user",),
        half_life_hours=3.0,
    )

    scorer = SliderScorer(encoder, slider_vectors)
    item_text = "A guide to running emotionally intelligent study sessions"

    scores = scorer.score(sampled_actions, item_text)

    for name, output in scores.items():
        print(f"Slider: {name}")
        print(f"  Probability: {output.probability:.3f}")
        print(f"  Features: {output.features}")
        print()


if __name__ == "__main__":
    main()

