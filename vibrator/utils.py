"""Utility helpers for examples and reproducible runs."""
from __future__ import annotations

import os
import random
from datetime import datetime, timezone
from typing import Mapping, Dict

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy, and torch (if available) for reproducibility.

    Notes:
    - Full determinism for transformer encoders can still vary by hardware/backend.
    - We set best-effort flags to reduce nondeterminism where supported.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        try:
            torch.manual_seed(seed)
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            # Best-effort deterministic settings
            if hasattr(torch, "use_deterministic_algorithms"):
                torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[attr-defined]
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            # For CUDA/cuBLAS determinism when applicable
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        except Exception:
            # If any backend-specific setting fails, continue without raising
            pass


def _parse_iso8601_utc(value: str) -> datetime:
    # Support trailing 'Z' and ensure timezone-aware UTC
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def stable_now(default_iso: str = "2024-01-01T12:00:00Z") -> datetime:
    """Return a stable reference timestamp for examples.

    Override with VIBRATOR_NOW_ISO="YYYY-MM-DDTHH:MM:SSZ" to control recency.
    """
    env = os.environ.get("VIBRATOR_NOW_ISO")
    try:
        if env:
            return _parse_iso8601_utc(env)
        return _parse_iso8601_utc(default_iso)
    except Exception:
        # Fallback to current UTC if parsing fails
        return datetime.now(timezone.utc)


def round_floats(features: Mapping[str, float], ndigits: int = 3) -> Dict[str, float]:
    """Return a copy of a float mapping with native floats rounded for display."""
    return {k: float(round(float(v), ndigits)) for k, v in features.items()}


# Friendly labels for feature names used in examples
FRIENDLY_FEATURE_NAMES: Dict[str, str] = {
    "user_to_slider": "User preference match",
    "item_to_slider": "Item style match",
    "user_item_alignment": "Personal relevance",
    "recency": "Recency boost",
}

FEATURE_ORDER = [
    "user_to_slider",
    "item_to_slider",
    "user_item_alignment",
    "recency",
]


def friendly_name(key: str) -> str:
    return FRIENDLY_FEATURE_NAMES.get(key, key)


def friendly_round_features(features: Mapping[str, float], ndigits: int = 3) -> Dict[str, float]:
    """Rename features to friendly labels and round values, preserving order where possible."""
    out: Dict[str, float] = {}
    # Fill known keys first in stable order
    for k in FEATURE_ORDER:
        if k in features:
            out[friendly_name(k)] = float(round(float(features[k]), ndigits))
    # Append any unknown keys
    for k, v in features.items():
        if k not in FEATURE_ORDER:
            out[friendly_name(k)] = float(round(float(v), ndigits))
    return out
