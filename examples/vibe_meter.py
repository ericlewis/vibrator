"""Console vibe meter: visualize per-slider probabilities and feature contributions.

Usage:
  uv run python examples/vibe_meter.py
  VIBRATOR_NOW_ISO=2024-01-01T12:00:00Z uv run python examples/vibe_meter.py
"""
from __future__ import annotations

import argparse
from typing import Dict, Mapping

from vibrator import InstructionalEncoder, SliderScorer, UserAction
from vibrator.utils import seed_everything, stable_now, round_floats, friendly_name, FEATURE_ORDER


SLIDERS: Dict[str, str] = {
    "technical": "Detailed technical documentation and precise language.",
    "beginner": "Simple, step-by-step explanations and gentle ramp-up.",
    "creative": "Playful, imaginative, and unconventional tone.",
}

DEFAULT_ACTIONS = [
    UserAction("write", "Seeking help planning roadmap and code review cadence", weight=1.3),
    UserAction("click", "Loved a post about debugging asyncio race conditions", weight=1.0),
    UserAction("react", "Upvoted a playful analogy about threads as cats", weight=0.8),
]


def bar(value: float, width: int = 24) -> str:
    value = max(0.0, min(1.0, float(value)))
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def norm_feature(name: str, val: float) -> float:
    if name in {"user_to_slider", "item_to_slider", "user_item_alignment"}:
        # Cosine-like similarity in [-1,1] -> map to [0,1]
        return (float(val) + 1.0) / 2.0
    return float(val)  # recency already in [0,1]


def print_section(title: str):
    print("\n" + title)
    print("-" * len(title))


def run(compare: bool) -> None:
    seed_everything(0)
    now = stable_now()

    encoder = InstructionalEncoder()
    slider_vectors = dict(zip(SLIDERS, encoder.encode_items(SLIDERS.values())))
    scorer = SliderScorer(encoder, slider_vectors)

    # Baseline context
    base_ctx: Mapping[str, object] = {"segment": "work", "is_work_hours": True}
    base_item = "Weekly standup tips to keep remote teams aligned"

    # Alternate context for comparison (optional)
    alt_ctx: Mapping[str, object] = {"segment": "after_hours", "is_work_hours": False}

    print("Vibe Meter")
    print("==========")

    # Score baseline
    base_scores = scorer.score(DEFAULT_ACTIONS, base_item, context=base_ctx)
    # Rank by probability
    ranked = sorted(base_scores.items(), key=lambda kv: kv[1].probability, reverse=True)

    print_section("Per-slider probabilities (baseline)")
    for name, output in ranked:
        p = float(output.probability)
        print(f"{name:>12}: {bar(p)} {p:.3f}")

    # Show feature breakdown for strongest slider
    best_name, best_out = ranked[0]
    print_section(f"Feature contributions for strongest slider: {best_name}")
    for key in FEATURE_ORDER:
        if key in best_out.features:
            fname = friendly_name(key)
            fval = best_out.features[key]
            nv = norm_feature(key, fval)  # Normalize to [0,1] for display
            print(f"{fname:>18}: {bar(nv)} {nv:.3f} (w={best_out.weights.get(key, 0.0):.2f})")
    # Print any unknown features at the end
    for key, fval in best_out.features.items():
        if key not in FEATURE_ORDER:
            fname = friendly_name(key)
            nv = norm_feature(key, fval)
            print(f"{fname:>18}: {bar(nv)} {nv:.3f} (w={best_out.weights.get(key, 0.0):.2f})")

    # Optional comparison
    if compare:
        print_section("Context comparison: baseline vs after_hours")
        alt_scores = scorer.score(DEFAULT_ACTIONS, base_item, context=alt_ctx)
        for name, _ in ranked:
            p0 = float(base_scores[name].probability)
            p1 = float(alt_scores[name].probability)
            delta = p1 - p0
            sign = "+" if delta >= 0 else "-"
            print(f"{name:>12}: {p0:.3f} -> {p1:.3f}  (Δ {sign}{abs(delta):.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="Show baseline vs after_hours context diff")
    args = ap.parse_args()
    run(compare=args.compare)
