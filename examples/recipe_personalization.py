"""Use recent chat feedback to pick the best recipe tweak."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from vibrator import (
    ChatMessage,
    InstructionalEncoder,
    SliderScorer,
    sample_recent_chat_actions,
)


SLIDERS = {
    "spicy": "Bolder heat levels, layered chiles, and bright acidity.",
    "comfort": "Cozy textures, familiar flavors, and soothing tone.",
    "time_saver": "Fast prep, minimal cleanup, and weeknight-friendly steps.",
    "macro_balance": "Protein-forward, balanced macros, and mindful calories.",
}


RECIPE_VARIANTS = [
    {
        "name": "Sheet-Pan Chipotle Tacos",
        "pitch": "Roasted veggies and chicken tossed in chipotle butter, finished with lime crema.",
        "context": {"segment": "weeknight"},
    },
    {
        "name": "Slow Cooker Verde Stew",
        "pitch": "Low-and-slow tomatillo stew with shredded turkey, white beans, and cilantro rice.",
        "context": {"segment": "weeknight"},
    },
    {
        "name": "Crunchy Cauli Power Bowl",
        "pitch": "Charred cauliflower, spiced chickpeas, pickled onions, and a yogurt-tahini drizzle.",
        "context": {"segment": "weeknight", "boost_feature": "macro_balance"},
    },
]


FEATURE_OVERRIDES = {
    "weeknight": {
        "item_to_slider": 1.2,
        "recency": 1.2,
    }
}


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
            content="Need dinner ideas that still feel cozy but no heavy cream please.",
            timestamp=now - timedelta(hours=3, minutes=45),
            user_id="cook-101",
        ),
        ChatMessage(
            content="We love a punch of heat and crunch—kids handle medium spice fine.",
            timestamp=now - timedelta(hours=1, minutes=15),
            user_id="cook-101",
        ),
        ChatMessage(
            content="Protein should stay high because we're watching macros this month.",
            timestamp=now - timedelta(minutes=35),
            user_id="cook-101",
        ),
        ChatMessage(
            content="Weeknights are tight, so keep prep under 30 minutes.",
            timestamp=now - timedelta(minutes=12),
            user_id="cook-101",
        ),
    ]

    actions = sample_recent_chat_actions(
        transcript,
        now=now,
        include_roles=("user",),
        max_messages=4,
        half_life_hours=5.0,
    )

    ranked_variants = []
    for variant in RECIPE_VARIANTS:
        scores = scorer.score(actions, variant["pitch"], context=variant.get("context"))
        macro = scores["macro_balance"].probability
        spicy = scores["spicy"].probability
        comfort = scores["comfort"].probability
        time_saver = scores["time_saver"].probability
        composite = (macro * 0.35) + (spicy * 0.30) + (comfort * 0.20) + (time_saver * 0.15)
        best_slider, best_output = max(scores.items(), key=lambda item: item[1].probability)
        ranked_variants.append(
            {
                "name": variant["name"],
                "composite": composite,
                "best_slider": best_slider,
                "best_probability": best_output.probability,
                "scores": {k: v.probability for k, v in scores.items()},
            }
        )

    ranked_variants.sort(key=lambda entry: entry["composite"], reverse=True)

    print("Recipe candidates (highest composite first):\n")
    for entry in ranked_variants:
        print(f"Recipe: {entry['name']}")
        print(f"  Composite match: {entry['composite']:.3f}")
        print(
            f"  Strongest slider: {entry['best_slider']} (probability {entry['best_probability']:.3f})"
        )
        print(f"  Individual slider probabilities: {entry['scores']}")
        print()


if __name__ == "__main__":
    main()

