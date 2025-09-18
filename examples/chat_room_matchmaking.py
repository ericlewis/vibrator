"""Match a user's recent chat vibe to candidate chat rooms using sliders."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vibrator import (
    ChatMessage,
    InstructionalEncoder,
    SliderScorer,
    sample_recent_chat_actions,
)


SLIDER_DEFINITIONS = {
    "strategy": "The space focuses on structured planning, goal-setting, and accountability.",
    "brainstorm": "The space celebrates rapid-fire idea generation and playful prompts.",
    "support": "The space offers emotional support, validation, and reflective listening.",
    "deep_focus": "The space values in-depth analysis, research links, and long-form guidance.",
}


CHAT_ROOMS = {
    "Launch Lab": {
        "pitch": "Daily standups with founders swapping progress check-ins and sprint rituals.",
        "context": {"segment": "after_hours", "boost_feature": "recency"},
    },
    "Idea Playground": {
        "pitch": "Casual riffing on fresh product ideas, creative warm-ups, and playful challenges.",
        "context": {"segment": "after_hours"},
    },
    "Care Corner": {
        "pitch": "Peer support circle focused on listening sessions, vent threads, and gentle nudges.",
        "context": {"segment": "after_hours"},
    },
    "Deep Dive Den": {
        "pitch": "Long-form research drops, annotated resources, and weekly expert AMAs.",
        "context": {"segment": "after_hours"},
    },
}


FEATURE_OVERRIDES = {
    "after_hours": {
        "recency": 1.4,
        "user_item_alignment": 0.9,
    }
}


def format_prob(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    encoder = InstructionalEncoder()
    slider_vectors = dict(zip(SLIDER_DEFINITIONS, encoder.encode_items(SLIDER_DEFINITIONS.values())))

    scorer = SliderScorer(
        encoder,
        slider_vectors,
        feature_overrides_by_segment=FEATURE_OVERRIDES,
    )

    now = datetime.now(timezone.utc)
    transcript = [
        ChatMessage(
            content="Struggling to keep my maker group motivated, we slip whenever deadlines feel vague.",
            timestamp=now - timedelta(hours=6, minutes=10),
            user_id="user-789",
        ),
        ChatMessage(
            content="We need check-ins that feel encouraging, not guilt trips.",
            timestamp=now - timedelta(hours=2, minutes=45),
            user_id="user-789",
        ),
        ChatMessage(
            content="Brainstorm games help, but we also crave honest space to vent when we miss goals.",
            timestamp=now - timedelta(minutes=50),
            user_id="user-789",
        ),
        ChatMessage(
            content="Looking for a room that mixes planning with emotional support.",
            timestamp=now - timedelta(minutes=8),
            user_id="user-789",
        ),
    ]

    sampled_actions = sample_recent_chat_actions(
        transcript,
        now=now,
        max_messages=4,
        include_roles=("user",),
        half_life_hours=4.0,
    )

    ranked_rooms = []

    for room_name, room in CHAT_ROOMS.items():
        scores = scorer.score(sampled_actions, room["pitch"], context=room.get("context"))
        averaged_probability = sum(output.probability for output in scores.values()) / len(scores)
        best_slider_name, best_slider_output = max(scores.items(), key=lambda item: item[1].probability)
        ranked_rooms.append(
            {
                "name": room_name,
                "average_probability": averaged_probability,
                "best_slider": best_slider_name,
                "best_probability": best_slider_output.probability,
                "features": best_slider_output.features,
            }
        )

    ranked_rooms.sort(key=lambda entry: entry["average_probability"], reverse=True)

    print("Top chat room matches:\n")
    for entry in ranked_rooms:
        print(f"Room: {entry['name']}")
        print(f"  Avg match confidence: {format_prob(entry['average_probability'])}")
        print(
            f"  Strongest slider: {entry['best_slider']} (probability {format_prob(entry['best_probability'])})"
        )
        print(f"  Feature breakdown: {entry['features']}")
        print()


if __name__ == "__main__":
    main()

