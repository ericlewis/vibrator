"""Utilities for sampling recent chat transcripts into user actions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Iterable, List, Sequence

from .actions import UserAction


@dataclass(slots=True)
class ChatMessage:
    """Lightweight container describing a single chat utterance."""

    content: str
    timestamp: datetime
    user_id: str | None = None
    role: str = "user"
    weight: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def sample_recent_chat_actions(
    messages: Sequence[ChatMessage],
    *,
    now: datetime | None = None,
    max_messages: int = 8,
    include_roles: Iterable[str] = ("user",),
    half_life_hours: float = 6.0,
    minimum_weight: float = 1e-3,
) -> List[UserAction]:
    """Convert the most recent chat turns into `UserAction` instances.

    Parameters
    ----------
    messages:
        Sequence of chat messages sorted or unsorted by timestamp.
    now:
        Reference time for recency calculations (defaults to current UTC).
    max_messages:
        Upper bound on how many recent messages to keep after filtering.
    include_roles:
        Roles to include when sampling (defaults to end-user authored turns).
    half_life_hours:
        Controls exponential decay applied to message weights based on age.
    minimum_weight:
        Lower bound to keep weights from collapsing to zero after decay.
    """

    if not messages:
        return []

    include_roles_set = {role.lower() for role in include_roles}

    now_dt = _to_utc(now or datetime.now(timezone.utc))

    filtered = [m for m in messages if m.role.lower() in include_roles_set]
    if not filtered:
        return []

    filtered.sort(key=lambda msg: msg.timestamp)
    recent = filtered[-max_messages:]

    actions: List[UserAction] = []
    decay_factor = math.log(2.0) / max(half_life_hours, 1e-6)

    for msg in recent:
        msg_time = _to_utc(msg.timestamp)
        age_hours = max((now_dt - msg_time).total_seconds() / 3600.0, 0.0)
        decay = math.exp(-decay_factor * age_hours)
        weight = max(msg.weight * decay, minimum_weight)

        metadata = dict(msg.metadata)
        metadata.setdefault("role", msg.role)
        if msg.user_id is not None:
            metadata.setdefault("user_id", str(msg.user_id))
        metadata["age_hours"] = f"{age_hours:.4f}"

        actions.append(
            UserAction(
                action_type="chat",
                content=msg.content,
                weight=weight,
                metadata=metadata,
            )
        )

    return actions

