"""Synthetic data generators for quick experimentation."""
from __future__ import annotations

import random
from typing import Dict, List, Mapping, Sequence

from .actions import UserAction

SLIDER_TEMPLATES: Mapping[str, str] = {
    "professional": "The user values structure, progress, and productivity cues.",
    "personal": "The user wants empathetic, relationship-focused support.",
    "playful": "The user responds well to witty, humorous, or lighthearted tone.",
    "experimental": "The user enjoys novel features and early access experiments.",
}

ITEM_TEMPLATES: Sequence[str] = (
    "Async standup checklist for distributed teams",
    "Guided meditation to reset after meetings",
    "Playful icebreakers for chat-based communities",
    "Beta feature spotlight: collaborative notes",
)

ACTION_TYPES: Sequence[str] = ("write", "read", "react", "click", "share")


def synthetic_slider_texts(limit: int | None = None) -> Dict[str, str]:
    """Return a subset of slider definition texts for bootstrapping models."""
    items = list(SLIDER_TEMPLATES.items())
    if limit is not None:
        items = items[:limit]
    return dict(items)


def synthetic_item_corpus(limit: int | None = None) -> List[str]:
    """Return synthetic item content strings."""
    items = list(ITEM_TEMPLATES)
    if limit is not None:
        items = items[:limit]
    return items


def synthetic_actions(
    count: int = 5,
    seed: int | None = 42,
    weight_range: tuple[float, float] = (0.5, 1.5),
    age_hours_range: tuple[float, float] = (0.0, 24.0),
) -> List[UserAction]:
    """Generate pseudo-random user actions for smoke tests and demos."""
    rng = random.Random(seed)
    actions: List[UserAction] = []
    for idx in range(count):
        action_type = rng.choice(ACTION_TYPES)
        content = f"Synthetic {action_type} content #{idx}"
        weight = round(rng.uniform(*weight_range), 3)
        age_hours = round(rng.uniform(*age_hours_range), 3)
        metadata = {"age_hours": age_hours}
        actions.append(UserAction(action_type, content, weight=weight, metadata=metadata))
    return actions
