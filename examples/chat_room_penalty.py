"""Apply a hard penalty when users disrespect @tawny."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vibrator import (
    ChatMessage,
    InstructionalEncoder,
    SliderScorer,
    sample_recent_chat_actions,
)


SLIDERS = {
    "strategy": "Focuses on structure, goal-setting, and detailed next steps.",
    "support": "Offers emotional validation, encouragement, and gentle nudges.",
}

CHAT_ROOM = {
    "name": "Care Corner",
    "pitch": "Peer circle for checking in, celebrating wins, and regrouping after setbacks.",
}


def disrespecting_tawny(actions) -> bool:
    for action in actions:
        text = action.content.lower()
        if "@tawny" in text and "corn dog" in text:
            return True
    return False


def main() -> None:
    encoder = InstructionalEncoder()
    slider_vectors = dict(zip(SLIDERS, encoder.encode_items(SLIDERS.values())))
    scorer = SliderScorer(encoder, slider_vectors)

    now = datetime.now(timezone.utc)
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

    if disrespecting_tawny(actions):
        penalty = 8.0  # push logit deep negative so sigmoid ~ 0
        for slider_name, output in scores.items():
            output.raw_score -= penalty
            output.probability = scorer._calibrated_probability(slider_name, output.raw_score)  # type: ignore[attr-defined]

    print(f"Chat room: {CHAT_ROOM['name']}")
    for slider_name, output in scores.items():
        print(f"  Slider: {slider_name}")
        print(f"    Raw score: {output.raw_score:.2f}")
        print(f"    Probability: {output.probability:.3f}")


if __name__ == "__main__":
    main()

