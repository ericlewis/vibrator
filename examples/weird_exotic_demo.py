"""Ridiculously exotic personalization scenario for cosmic karaoke safaris."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vibrator import (
    ChatMessage,
    InstructionalEncoder,
    SliderScorer,
    sample_recent_chat_actions,
)


SLIDERS = {
    "void_whispers": "Prefers content that harmonizes with interdimensional whale songs and existential poetry.",
    "lava_surfing": "Seeks adrenaline-fueled experiences, molten choreography, and heat-proof snacks.",
    "bureaucratic_peacocks": "Craves ceremonial paperwork, flamboyant etiquette, and avian HR policies.",
}


EXOTIC_EVENTS = [
    {
        "name": "Nebulae Karaoke Safari",
        "pitch": "Sing duets with gravitational sirens while touring ionized jungle gyms.",
        "context": {"segment": "third_moon", "boost_feature": "recency"},
    },
    {
        "name": "Magma Flow Flash Mob",
        "pitch": "Synchronize surfboards atop polite volcanoes serving capsaicin mocktails.",
        "context": {"segment": "volcano_hour"},
    },
    {
        "name": "Peacock Ombudsman Summit",
        "pitch": "File grievance haikus under the watchful gaze of jeweled birds with clipboards.",
        "context": {"segment": "bureaucracy_gala"},
    },
]


FEATURE_OVERRIDES = {
    "third_moon": {"recency": 1.6, "user_to_slider": 1.1},
    "volcano_hour": {"item_to_slider": 1.5},
    "bureaucracy_gala": {"user_item_alignment": 1.4},
}


def cursed_phrase_penalty(actions) -> float:
    for action in actions:
        if "mayonnaise portal" in action.content.lower():
            return 5.0
    return 0.0


def main() -> None:
    encoder = InstructionalEncoder()
    slider_vectors = dict(zip(SLIDERS, encoder.encode_items(SLIDERS.values())))
    scorer = SliderScorer(
        encoder,
        slider_vectors,
        feature_overrides_by_segment=FEATURE_OVERRIDES,
    )

    now = datetime.now(timezone.utc)
    transcript = [
        ChatMessage(
            content="Craving karaoke that echoes through the abyss but still offers gluten-free stardust.",
            timestamp=now - timedelta(hours=2, minutes=7),
            user_id="cosmic-404",
        ),
        ChatMessage(
            content="Also want lava surfing lessons where instructors speak in palindromes.",
            timestamp=now - timedelta(hours=1, minutes=3),
            user_id="cosmic-404",
        ),
        ChatMessage(
            content="If there's a mayonnaise portal again I'm out.",
            timestamp=now - timedelta(minutes=22),
            user_id="cosmic-404",
        ),
        ChatMessage(
            content="Need at least one polite bird verifying attendance with glitter stamps.",
            timestamp=now - timedelta(minutes=6),
            user_id="cosmic-404",
        ),
    ]

    actions = sample_recent_chat_actions(
        transcript,
        now=now,
        include_roles=("user",),
        max_messages=4,
        half_life_hours=4.2,
    )

    penalty = cursed_phrase_penalty(actions)

    ranked = []
    for event in EXOTIC_EVENTS:
        scores = scorer.score(actions, event["pitch"], context=event.get("context"))
        if penalty:
            for name, output in scores.items():
                output.raw_score -= penalty
                output.probability = scorer._calibrated_probability(name, output.raw_score)  # type: ignore[attr-defined]
        strongest, strongest_output = max(scores.items(), key=lambda item: item[1].probability)
        ranked.append(
            {
                "event": event["name"],
                "strongest": strongest,
                "probability": strongest_output.probability,
                "feature_breakdown": strongest_output.features,
            }
        )

    ranked.sort(key=lambda entry: entry["probability"], reverse=True)

    print("Extravagantly weird recommendations:\n")
    for entry in ranked:
        print(f"Event: {entry['event']}")
        print(f"  Supreme slider alignment: {entry['strongest']} ({entry['probability']:.3f})")
        print(f"  Feature breakdown: {entry['feature_breakdown']}")
        print()


if __name__ == "__main__":
    main()

