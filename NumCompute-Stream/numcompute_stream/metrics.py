"""Streaming classification metrics."""

from __future__ import annotations

from collections import deque
import numpy as np


def _labels(y_true, y_pred):
    truth = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return truth.astype(int), pred.astype(int)


def accuracy_score(y_true, y_pred) -> float:
    t, p = _labels(y_true, y_pred)
    return float(np.mean(t == p)) if t.size else 0.0


def precision_score(y_true, y_pred, positive_label: int = 1) -> float:
    t, p = _labels(y_true, y_pred)
    tp = np.sum((t == positive_label) & (p == positive_label))
    fp = np.sum((t != positive_label) & (p == positive_label))
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall_score(y_true, y_pred, positive_label: int = 1) -> float:
    t, p = _labels(y_true, y_pred)
    tp = np.sum((t == positive_label) & (p == positive_label))
    fn = np.sum((t == positive_label) & (p != positive_label))
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def f1_score(y_true, y_pred, positive_label: int = 1) -> float:
    pr = precision_score(y_true, y_pred, positive_label)
    rc = recall_score(y_true, y_pred, positive_label)
    return float(2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0


class StreamingMetric:
    def update(self, y_true, y_pred):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class OnlineAccuracy(StreamingMetric):
    def __init__(self, window_size: int | None = None):
        self.window_size = window_size
        self.reset()

    def update(self, y_true, y_pred):
        t, p = _labels(y_true, y_pred)
        hits = (t == p).astype(int)
        if self.window_size is None:
            self.correct += int(hits.sum())
            self.total += int(hits.size)
        else:
            self.recent.extend(hits.tolist())
            while len(self.recent) > self.window_size:
                self.recent.popleft()
        return self

    def result(self):
        if self.window_size is None:
            return float(self.correct / self.total) if self.total else 0.0
        return float(np.mean(self.recent)) if self.recent else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0
        self.recent = deque()
        return self


class _BinaryCounts(StreamingMetric):
    def __init__(self, positive_label: int = 1):
        self.positive_label = int(positive_label)
        self.reset()

    def update(self, y_true, y_pred):
        t, p = _labels(y_true, y_pred)
        pos = self.positive_label
        self.tp += int(np.sum((t == pos) & (p == pos)))
        self.fp += int(np.sum((t != pos) & (p == pos)))
        self.fn += int(np.sum((t == pos) & (p != pos)))
        return self

    def reset(self):
        self.tp = self.fp = self.fn = 0
        return self


class OnlinePrecision(_BinaryCounts):
    def result(self):
        return float(self.tp / (self.tp + self.fp)) if (self.tp + self.fp) else 0.0


class OnlineRecall(_BinaryCounts):
    def result(self):
        return float(self.tp / (self.tp + self.fn)) if (self.tp + self.fn) else 0.0


class OnlineF1(_BinaryCounts):
    def result(self):
        precision = float(self.tp / (self.tp + self.fp)) if (self.tp + self.fp) else 0.0
        recall = float(self.tp / (self.tp + self.fn)) if (self.tp + self.fn) else 0.0
        return float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


class OnlineConfusionMatrix(StreamingMetric):
    def __init__(self, labels=None):
        self.fixed_labels = None if labels is None else list(labels)
        self.reset()

    def update(self, y_true, y_pred):
        t, p = _labels(y_true, y_pred)
        if self.fixed_labels is None:
            for v in np.unique(np.concatenate([t, p])):
                if int(v) not in self.index:
                    self.index[int(v)] = len(self.labels)
                    self.labels.append(int(v))
                    self.matrix = self._expanded_matrix()
        for truth, pred in zip(t, p):
            self.matrix[self.index[int(truth)], self.index[int(pred)]] += 1
        return self

    def _expanded_matrix(self):
        out = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        old = getattr(self, "matrix", None)
        if old is not None:
            out[: old.shape[0], : old.shape[1]] = old
        return out

    def result(self):
        return self.matrix.copy()

    def reset(self):
        self.labels = [] if self.fixed_labels is None else list(self.fixed_labels)
        self.index = {int(v): i for i, v in enumerate(self.labels)}
        self.matrix = np.zeros((len(self.labels), len(self.labels)), dtype=int)
        return self


StreamingAccuracy = OnlineAccuracy
StreamingPrecision = OnlinePrecision
StreamingRecall = OnlineRecall
StreamingF1 = OnlineF1
ConfusionMatrix = OnlineConfusionMatrix
