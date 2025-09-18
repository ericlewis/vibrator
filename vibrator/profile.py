"""Utilities for aggregating heterogeneous actions into user profiles."""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from .actions import UserAction

if TYPE_CHECKING:  # pragma: no cover - avoids heavy import at runtime
    from .encoder import InstructionalEncoder


def embed_user_actions(
    encoder: 'InstructionalEncoder',
    actions: Sequence[UserAction],
    minimum_actions: int = 1,
) -> np.ndarray:
    """Encode a sequence of actions; returns empty array if below threshold."""
    if len(actions) < minimum_actions:
        return np.zeros((0, encoder.model.get_sentence_embedding_dimension()))
    return encoder.encode_actions(actions)


def weighted_user_embedding(
    encoder: 'InstructionalEncoder',
    actions: Sequence[UserAction],
    weight_temperature: float = 1.0,
) -> np.ndarray:
    """Combine action embeddings into a single profile vector."""
    if not actions:
        raise ValueError("At least one action is required to build a user embedding.")

    embeddings = encoder.encode_actions(actions)
    weights = np.array([a.weight for a in actions], dtype=np.float32)
    if weight_temperature != 1.0:
        weights = weights ** weight_temperature
    weights = weights / weights.sum()
    return np.average(embeddings, axis=0, weights=weights)
