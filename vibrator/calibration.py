"""Calibration utilities for slider probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may be optional in tests
    torch = None

from sklearn.isotonic import IsotonicRegression


@dataclass
class TemperatureCalibrator:
    """Temperature scaling for binary logits."""

    temperature: float = 1.0

    def fit(self, logits: Iterable[float], labels: Iterable[int], max_iter: int = 50) -> "TemperatureCalibrator":
        if torch is None:
            raise RuntimeError("torch is required for temperature scaling")
        logits_tensor = torch.tensor(list(logits), dtype=torch.float32)
        labels_tensor = torch.tensor(list(labels), dtype=torch.float32)

        if logits_tensor.ndim != 1:
            raise ValueError("TemperatureCalibrator expects 1-D logits for binary tasks.")

        param = torch.nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = torch.optim.LBFGS([param], lr=0.1, max_iter=max_iter)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():  # type: ignore[return-value]
            optimizer.zero_grad()
            loss = loss_fn(logits_tensor / param, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(param.detach().cpu())
        return self

    def transform(self, logits: Iterable[float]) -> np.ndarray:
        logits_array = np.asarray(list(logits), dtype=np.float32)
        scaled = logits_array / max(self.temperature, 1e-6)
        return 1.0 / (1.0 + np.exp(-scaled))


class IsotonicCalibrator:
    """Wrap sklearn isotonic regression for monotonic calibration."""

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: Iterable[float], labels: Iterable[int]) -> "IsotonicCalibrator":
        scores_array = np.asarray(list(scores), dtype=np.float32)
        labels_array = np.asarray(list(labels), dtype=np.float32)
        self._model.fit(scores_array, labels_array)
        return self

    def transform(self, scores: Iterable[float]) -> np.ndarray:
        scores_array = np.asarray(list(scores), dtype=np.float32)
        return self._model.transform(scores_array)

    def fit_transform(self, scores: Iterable[float], labels: Iterable[int]) -> Tuple[np.ndarray, "IsotonicCalibrator"]:
        calibrated = self.fit(scores, labels).transform(scores)
        return calibrated, self
