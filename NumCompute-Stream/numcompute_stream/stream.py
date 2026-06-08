"""Training controller for chunk-wise learning."""

from __future__ import annotations

import sys
import time
import numpy as np

from .metrics import accuracy_score


class StreamTrainer:
    """Manage a streaming model and store per-chunk records."""

    def __init__(self, model, scorer=accuracy_score):
        self.model = model
        self.scorer = scorer
        self.history: list[dict] = []
        self._step = 0
        self._seen = 0
        self._correct = 0

    def fit_chunk(self, X, y):
        started = time.perf_counter()
        self.model.partial_fit(X, y)
        elapsed = time.perf_counter() - started
        self._step += 1
        record = {
            "step": self._step,
            "samples": int(len(y)),
            "fit_seconds": float(elapsed),
            "memory_bytes": self._rough_size(),
        }
        self.history.append(record)
        return record

    def score_chunk(self, X, y):
        pred = self.model.predict(X)
        score = float(self.scorer(y, pred))
        correct = int(np.sum(np.asarray(y).ravel() == pred.ravel()))
        self._seen += int(len(y))
        self._correct += correct
        if self.history:
            self.history[-1]["score"] = score
            self.history[-1]["cumulative_accuracy"] = self.cumulative_accuracy
        return score

    @property
    def cumulative_accuracy(self):
        return float(self._correct / self._seen) if self._seen else 0.0

    def metric_series(self, key: str = "score"):
        return [row[key] for row in self.history if key in row]

    def _rough_size(self):
        return sys.getsizeof(self.model) + sys.getsizeof(self.history)
