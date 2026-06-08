"""Streaming preprocessing components."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _as_2d_float(X) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("Input must be 1D or 2D numeric data.")
    return arr


@dataclass
class _Moments:
    count: np.ndarray
    mean: np.ndarray
    m2: np.ndarray


class OnlineStandardScaler:
    """Incremental standard scaler using chunk-level moment merging."""

    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = float(epsilon)
        self._state: _Moments | None = None

    def partial_fit(self, X, y=None):
        X_arr = _as_2d_float(X)
        valid = ~np.isnan(X_arr)
        chunk_count = valid.sum(axis=0).astype(float)
        safe = np.where(valid, X_arr, 0.0)
        chunk_sum = safe.sum(axis=0)
        chunk_mean = np.divide(chunk_sum, chunk_count, out=np.zeros_like(chunk_sum), where=chunk_count > 0)
        centred = np.where(valid, X_arr - chunk_mean, 0.0)
        chunk_m2 = np.square(centred).sum(axis=0)

        if self._state is None:
            self._state = _Moments(chunk_count, chunk_mean, chunk_m2)
            return self

        old = self._state
        total = old.count + chunk_count
        delta = chunk_mean - old.mean
        new_mean = old.mean + np.divide(delta * chunk_count, total, out=np.zeros_like(delta), where=total > 0)
        correction = np.divide(old.count * chunk_count, total, out=np.zeros_like(total), where=total > 0) * np.square(delta)
        new_m2 = old.m2 + chunk_m2 + correction
        self._state = _Moments(total, np.where(total > 0, new_mean, old.mean), new_m2)
        return self

    @property
    def mean_(self):
        if self._state is None:
            raise RuntimeError("Scaler has not been fitted.")
        return self._state.mean

    @property
    def var_(self):
        if self._state is None:
            raise RuntimeError("Scaler has not been fitted.")
        return np.divide(self._state.m2, self._state.count, out=np.zeros_like(self._state.m2), where=self._state.count > 0)

    @property
    def scale_(self):
        return np.sqrt(np.maximum(self.var_, self.epsilon))

    def transform(self, X):
        X_arr = _as_2d_float(X)
        if self._state is None:
            raise RuntimeError("Call partial_fit before transform.")
        if X_arr.shape[1] != len(self.mean_):
            raise ValueError("Feature count does not match fitted scaler.")
        filled = np.where(np.isnan(X_arr), self.mean_, X_arr)
        return (filled - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.partial_fit(X).transform(X)


class OnlineMeanImputer:
    """Replace NaNs with incrementally updated column means."""

    def __init__(self):
        self._count: np.ndarray | None = None
        self._mean: np.ndarray | None = None

    def partial_fit(self, X, y=None):
        X_arr = _as_2d_float(X)
        mask = ~np.isnan(X_arr)
        c = mask.sum(axis=0).astype(float)
        s = np.where(mask, X_arr, 0.0).sum(axis=0)
        m = np.divide(s, c, out=np.zeros_like(s), where=c > 0)
        if self._count is None:
            self._count, self._mean = c, m
        else:
            total = self._count + c
            delta = m - self._mean
            self._mean = self._mean + np.divide(delta * c, total, out=np.zeros_like(delta), where=total > 0)
            self._count = total
        return self

    @property
    def statistics_(self):
        if self._mean is None:
            raise RuntimeError("Imputer has not been fitted.")
        return self._mean

    def transform(self, X):
        X_arr = _as_2d_float(X).copy()
        if self._mean is None:
            raise RuntimeError("Call partial_fit before transform.")
        if X_arr.shape[1] != len(self._mean):
            raise ValueError("Feature count does not match fitted imputer.")
        rows, cols = np.where(np.isnan(X_arr))
        X_arr[rows, cols] = self._mean[cols]
        return X_arr

    def fit_transform(self, X, y=None):
        return self.partial_fit(X).transform(X)


class StreamingMinMaxScaler:
    """Incremental min-max scaling."""

    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = float(epsilon)
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def partial_fit(self, X, y=None):
        X_arr = _as_2d_float(X)
        chunk_min = np.nanmin(X_arr, axis=0)
        chunk_max = np.nanmax(X_arr, axis=0)
        if self.min_ is None:
            self.min_, self.max_ = chunk_min, chunk_max
        else:
            self.min_ = np.minimum(self.min_, chunk_min)
            self.max_ = np.maximum(self.max_, chunk_max)
        return self

    def transform(self, X):
        X_arr = _as_2d_float(X)
        if self.min_ is None or self.max_ is None:
            raise RuntimeError("Call partial_fit before transform.")
        denom = np.maximum(self.max_ - self.min_, self.epsilon)
        filled = np.where(np.isnan(X_arr), self.min_, X_arr)
        return (filled - self.min_) / denom

    def fit_transform(self, X, y=None):
        return self.partial_fit(X).transform(X)


StandardScaler = OnlineStandardScaler
SimpleImputer = OnlineMeanImputer
Imputer = OnlineMeanImputer
MinMaxScaler = StreamingMinMaxScaler
