"""Utilities for describing heterogeneous user actions before embedding."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

DEFAULT_ACTION_PREFIXES: Mapping[str, str] = {
    "write": "User wrote:",
    "click": "User clicked on:",
    "react": "User reacted to:",
    "read": "User viewed:",
    "open": "User opened:",
    "share": "User shared:",
    "chat": "User said:",
}


def action_prefix(action_type: str) -> str:
    """Return the prompt prefix for a given action type."""
    return DEFAULT_ACTION_PREFIXES.get(action_type.lower(), f"User performed {action_type} on:")


@dataclass
class UserAction:
    """Container for a user action event to be embedded."""

    action_type: str
    content: str
    weight: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def instruction(self) -> str:
        """Render the action as a natural-language instruction for the encoder."""
        prefix = action_prefix(self.action_type)
        return f"{prefix} {self.content}".strip()
