"""Streaming descriptive statistics."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class RunningStatsState:
    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


class StreamingStats:
    """Maintain mean, variance, min, max and optional recent buffer."""

    def __init__(self, keep_recent: int = 1000):
        self.keep_recent = int(keep_recent)
        self._state: RunningStatsState | None = None
        self._recent = deque(maxlen=max(1, self.keep_recent))

    def update_stats(self, X_chunk):
        X = np.asarray(X_chunk, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X_chunk must be 1D or 2D.")
        if X.size == 0:
            return self

        mask = ~np.isnan(X)
        c = mask.sum(axis=0).astype(float)
        clean = np.where(mask, X, 0.0)
        s = clean.sum(axis=0)
        chunk_mean = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
        chunk_m2 = np.square(np.where(mask, X - chunk_mean, 0.0)).sum(axis=0)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)

        if self._state is None:
            self._state = RunningStatsState(c, chunk_mean, chunk_m2, chunk_min, chunk_max)
        else:
            old = self._state
            total = old.count + c
            delta = chunk_mean - old.mean
            mean = old.mean + np.divide(delta * c, total, out=np.zeros_like(delta), where=total > 0)
            adjust = np.divide(old.count * c, total, out=np.zeros_like(total), where=total > 0) * np.square(delta)
            self._state = RunningStatsState(
                total,
                np.where(total > 0, mean, old.mean),
                old.m2 + chunk_m2 + adjust,
                np.fmin(old.minimum, chunk_min),
                np.fmax(old.maximum, chunk_max),
            )

        for row in X:
            self._recent.append(row.copy())
        return self

    def _require(self):
        if self._state is None:
            raise RuntimeError("No statistics available. Call update_stats first.")
        return self._state

    def mean(self):
        return self._require().mean.copy()

    def variance(self, ddof: int = 0):
        state = self._require()
        denom = np.maximum(state.count - ddof, 1)
        return state.m2 / denom

    def min(self):
        return self._require().minimum.copy()

    def max(self):
        return self._require().maximum.copy()

    def quantile(self, q: float):
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1.")
        if not self._recent:
            raise RuntimeError("No recent data stored.")
        return np.nanquantile(np.vstack(self._recent), q, axis=0)

    def histogram(self, bins: int = 10, feature: int = 0):
        if not self._recent:
            raise RuntimeError("No recent data stored.")
        values = np.vstack(self._recent)[:, feature]
        values = values[~np.isnan(values)]
        return np.histogram(values, bins=bins)


class RunningMean:
    def __init__(self):
        self.stats = StreamingStats()

    def update(self, X_chunk):
        self.stats.update_stats(X_chunk)
        return self

    def result(self):
        return self.stats.mean()
