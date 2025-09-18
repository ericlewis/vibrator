"""Instruction-tuned embedding helpers built around sentence-transformers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .actions import UserAction
from .normalization import ensure_numpy, layer_normalize, l2_normalize

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - dependency is optional at import time
    raise RuntimeError(
        "sentence-transformers is required for InstructionalEncoder. Install it via 'pip install sentence-transformers'."
    ) from exc


DEFAULT_ACTION_INSTRUCTION = "Represent this user behavior event for personalization:"
DEFAULT_ITEM_INSTRUCTION = "Represent this content item for personalization:"


@dataclass
class EncoderConfig:
    model_name: str = "hkunlp/instructor-base"
    device: str = "cpu"
    batch_size: int = 32
    normalize_layer: bool = True
    normalize_l2: bool = True


class InstructionalEncoder:
    """Wrapper around Instructor models with sensible defaults for personalization."""

    def __init__(
        self,
        config: EncoderConfig | None = None,
        action_instruction: str = DEFAULT_ACTION_INSTRUCTION,
        item_instruction: str = DEFAULT_ITEM_INSTRUCTION,
    ) -> None:
        self.config = config or EncoderConfig()
        self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
        self.action_instruction = action_instruction
        self.item_instruction = item_instruction

    def _post_process(self, embeddings: np.ndarray) -> np.ndarray:
        vectors = ensure_numpy(embeddings)
        if self.config.normalize_layer:
            vectors = layer_normalize(vectors)
        if self.config.normalize_l2:
            vectors = l2_normalize(vectors)
        return vectors

    def _encode_pairs(self, pairs: Sequence[Sequence[str]]) -> np.ndarray:
        embeddings = self.model.encode(
            pairs,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return self._post_process(embeddings)

    def encode_items(self, texts: Iterable[str]) -> np.ndarray:
        """Embed candidate items/content pieces."""
        pairs = [[self.item_instruction, text] for text in texts]
        return self._encode_pairs(pairs)

    def encode_actions(self, actions: Iterable[UserAction]) -> np.ndarray:
        """Embed user actions with action-aware prompts."""
        pairs: List[List[str]] = []
        for action in actions:
            prompt = action.instruction()
            pairs.append([self.action_instruction, prompt])
        return self._encode_pairs(pairs)

    def encode_freeform(self, instruction: str, inputs: Iterable[str]) -> np.ndarray:
        """Embed arbitrary texts with a provided instruction."""
        pairs = [[instruction, text] for text in inputs]
        return self._encode_pairs(pairs)
