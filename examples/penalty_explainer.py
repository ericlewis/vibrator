"""Penalty-aware chat room explainer: show before/after and deltas.

Usage:
  uv run python examples/penalty_explainer.py
  VIBRATOR_NOW_ISO=2024-01-01T12:00:00Z uv run python examples/penalty_explainer.py
"""
from __future__ import annotations

import math
from datetime import timedelta
from typing import Dict, Tuple

from vibrator import ChatMessage, InstructionalEncoder, SliderScorer, sample_recent_chat_actions
from vibrator.utils import seed_everything, stable_now


SLIDERS: Dict[str, str] = {
    "strategy": "Focuses on structure, goal-setting, and detailed next steps.",
    "support": "Offers emotional validation, encouragement, and gentle nudges.",
}

ROOM = {
    "name": "Care Corner",
    "pitch": "Peer circle for checking in, celebrating wins, and regrouping after setbacks.",
}


def penalty_strength(actions, *, half_life_hours: float = 4.0, base_penalty: float = 4.0) -> float:
    strength = 0.0
    for action in actions:
        text = action.content.lower()
        if "@tawny" in text and "corn dog" in text:
            age_hours = float(action.metadata.get("age_hours", 0.0)) if action.metadata else 0.0
            decay = math.exp(-math.log(2.0) * age_hours / max(half_life_hours, 1e-6))
            strength = max(strength, base_penalty * decay)
    return strength


def apply_penalty(scores, strength: float) -> None:
    if strength <= 0:
        return
    per_slider_multiplier = {"support": 1.5, "strategy": 0.8}
    floor_logit = -4.0
    for name, out in scores.items():
        shift = strength * per_slider_multiplier.get(name, 1.0)
        out.raw_score = max(out.raw_score - shift, floor_logit)


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

    before = scorer.score(actions, ROOM["pitch"])
    # Clone raw scores to compute after independently
    after = {k: type(v)(v.raw_score, v.probability, dict(v.features), dict(v.weights)) for k, v in before.items()}

    strength = penalty_strength(actions)
    apply_penalty(after, strength)
    for name, out in after.items():
        out.probability = scorer._calibrated_probability(name, out.raw_score)  # type: ignore[attr-defined]

    print("Chat Room Penalty Explainer")
    print("===========================")
    print(f"Room: {ROOM['name']}")
    print(f"Penalty strength: {strength:.2f}\n")

    # Sort by before probability
    order = sorted(before.keys(), key=lambda k: before[k].probability, reverse=True)
    for name in order:
        p0 = float(before[name].probability)
        p1 = float(after[name].probability)
        delta = p1 - p0
        sign = "+" if delta >= 0 else "-"
        print(f"{name:>10}: before={p0:.3f}  after={p1:.3f}  (Δ {sign}{abs(delta):.3f})")
    print()

    strongest = max(after.items(), key=lambda kv: kv[1].probability)[0]
    print(f"Strongest after penalty: {strongest}")


if __name__ == "__main__":
    main()
