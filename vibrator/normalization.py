"""Vector normalization helpers for alignment across domains."""
from __future__ import annotations

import numpy as np


def layer_normalize(vectors: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Apply feature-wise layer normalization to each embedding."""
    mean = vectors.mean(axis=1, keepdims=True)
    variance = ((vectors - mean) ** 2).mean(axis=1, keepdims=True)
    normalized = (vectors - mean) / np.sqrt(variance + eps)
    return normalized


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale embeddings to unit length."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def ensure_numpy(array_like) -> np.ndarray:
    """Convert torch or list inputs to np.ndarray without copying when possible."""
    if isinstance(array_like, np.ndarray):
        return array_like
    try:
        import torch

        if isinstance(array_like, torch.Tensor):
            return array_like.detach().cpu().numpy()
    except ImportError:  # pragma: no cover - torch optional at runtime
        pass
    return np.asarray(array_like)
