"""High-level interface for the vibrator personalization prototype."""
from __future__ import annotations

from typing import TYPE_CHECKING

from .actions import UserAction
from .calibration import IsotonicCalibrator, TemperatureCalibrator
from .pipeline import SliderOutput, SliderScorer

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from .encoder import EncoderConfig, InstructionalEncoder

__all__ = [
    "UserAction",
    "InstructionalEncoder",
    "EncoderConfig",
    "SliderScorer",
    "SliderOutput",
    "TemperatureCalibrator",
    "IsotonicCalibrator",
]


def __getattr__(name: str):  # pragma: no cover - thin forwarding logic
    if name in {"InstructionalEncoder", "EncoderConfig"}:
        from .encoder import EncoderConfig, InstructionalEncoder

        return {"InstructionalEncoder": InstructionalEncoder, "EncoderConfig": EncoderConfig}[name]
    raise AttributeError(f"module 'vibrator' has no attribute {name!r}")
