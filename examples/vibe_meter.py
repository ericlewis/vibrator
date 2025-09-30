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


def rel_scale(values):
    mn = min(values)
    mx = max(values)
    if mx - mn <= 1e-12:
        return [0.5 for _ in values], mn, mx
    return [float((v - mn) / (mx - mn)) for v in values], mn, mx


def norm_feature(name: str, val: float) -> float:
    if name in {"user_to_slider", "item_to_slider", "user_item_alignment"}:
        # Cosine-like similarity in [-1,1] -> map to [0,1]
        return (float(val) + 1.0) / 2.0
    return float(val)  # recency already in [0,1]


def print_section(title: str):
    print("\n" + title)
    print("-" * len(title))


def _print_explainer(compare: bool, contrast: bool, relative: bool) -> None:
    print("What this shows:")
    print("- Per-slider probabilities: likelihood this item matches each slider for the baseline context.")
    if relative:
        print("- Relative distribution: softmax across sliders for this item (sums to 1), sharper separation.")
    if compare:
        print("- Context comparison: change (Δ) if we switch to `after_hours` context.")
    print("- Feature contributions: why the strongest slider is strong (user/item/slider/recency).")
    if contrast:
        print("- Contrast bars: additional group-normalized bars to amplify small differences.")
    print()


def run(compare: bool, contrast: bool, relative: bool, tau: float) -> None:
    seed_everything(0)
    now = stable_now()

    encoder = InstructionalEncoder()
    slider_vectors = dict(zip(SLIDERS, encoder.encode_items(SLIDERS.values())))
    scorer = SliderScorer(encoder, slider_vectors, relative_softmax_temperature=tau)

    # Baseline context
    base_ctx: Mapping[str, object] = {"segment": "work", "is_work_hours": True}
    base_item = "Weekly standup tips to keep remote teams aligned"

    # Alternate context for comparison (optional)
    alt_ctx: Mapping[str, object] = {"segment": "after_hours", "is_work_hours": False}

    print("Vibe Meter")
    print("==========")
    _print_explainer(compare, contrast, relative)

    # Score baseline
    base_scores = scorer.score(DEFAULT_ACTIONS, base_item, context=base_ctx)
    # Rank by probability
    ranked = sorted(base_scores.items(), key=lambda kv: kv[1].probability, reverse=True)

    print_section("Per-slider probabilities (baseline)")
    probs = [float(out.probability) for _, out in ranked]
    rels, mn, mx = rel_scale(probs)
    mean_p = sum(probs) / len(probs) if probs else 0.0
    for (name, output), rel in zip(ranked, rels):
        p = float(output.probability)
        margin = p - (mean_p * len(probs) - p) / max(len(probs) - 1, 1)  # p - mean(others)
        rank = 1 + sorted(probs, reverse=True).index(p)
        line = f"#{rank:<2} {name:>10}: {bar(p)} {p:.3f}"
        if relative and output.relative_probability is not None:
            rp = float(output.relative_probability)
            line += f"  | softmax {bar(rp, 12)} {int(round(rp*100)):>3}%"
        if contrast:
            line += f"  | rel {bar(rel, 12)} {int(round(rel*100)):>3}%"
        line += f"  | margin {margin:+.3f}"
        print(line)

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
        deltas = [float(alt_scores[name].probability) - float(base_scores[name].probability) for name, _ in ranked]
        # Scale deltas to +/- range for visualization
        reld, _, _ = rel_scale([d + 0.5 for d in deltas])  # shift to [0,1]-ish before scaling
        for (name, _), d, r in zip(ranked, deltas, reld):
            sign = "+" if d >= 0 else "-"
            print(f"{name:>12}: Δ {sign}{abs(d):.3f}  {bar(r, 12)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="Show baseline vs after_hours context diff")
    ap.add_argument("--contrast", action="store_true", help="Add relative bars normalized within the group")
    ap.add_argument("--relative", action="store_true", help="Show per-item softmax across sliders (sharper separation)")
    ap.add_argument("--tau", type=float, default=0.25, help="Softmax temperature across sliders (lower=sharper)")
    args = ap.parse_args()
    run(compare=args.compare, contrast=args.contrast, relative=args.relative, tau=args.tau)
