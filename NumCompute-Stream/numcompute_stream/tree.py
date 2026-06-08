"""A compact streaming-compatible decision tree classifier."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class TreeNode:
    prediction: int
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class _ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.X_parts = deque()
        self.y_parts = deque()
        self.size = 0

    def add(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D.")
        if len(X_arr) != len(y_arr):
            raise ValueError("X and y length mismatch.")
        self.X_parts.append(X_arr.copy())
        self.y_parts.append(y_arr.copy())
        self.size += len(y_arr)
        while self.size > self.capacity and self.X_parts:
            over = self.size - self.capacity
            first_X = self.X_parts[0]
            first_y = self.y_parts[0]
            if len(first_y) <= over:
                self.X_parts.popleft()
                self.y_parts.popleft()
                self.size -= len(first_y)
            else:
                self.X_parts[0] = first_X[over:]
                self.y_parts[0] = first_y[over:]
                self.size -= over

    def arrays(self):
        if self.size == 0:
            return np.empty((0, 0)), np.empty((0,), dtype=int)
        return np.vstack(self.X_parts), np.concatenate(self.y_parts)


class DecisionTreeClassifier:
    """Depth-limited decision tree with a streaming partial_fit interface."""

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: int | float | None = None,
        criterion: str = "gini",
        buffer_size: int = 512,
        random_state: int | None = None,
        n_thresholds: int = 16,
    ):
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.criterion = criterion
        self.n_thresholds = int(n_thresholds)
        self.rng = np.random.default_rng(random_state)
        self.memory = _ReplayMemory(buffer_size)
        self.root_: TreeNode | None = None
        self.classes_: np.ndarray | None = None

    def partial_fit(self, X_chunk, y_chunk):
        X, y = self._validate_xy(X_chunk, y_chunk)
        self.memory.add(X, y)
        X_mem, y_mem = self.memory.arrays()
        self.classes_ = np.unique(y_mem)
        self.root_ = self._grow(X_mem, y_mem, depth=0)
        return self

    def fit(self, X, y):
        self.memory = _ReplayMemory(max(len(y), self.memory.capacity))
        return self.partial_fit(X, y)

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if self.root_ is None:
            raise RuntimeError("Tree has not been fitted.")
        return np.asarray([self._predict_one(row, self.root_) for row in X_arr], dtype=int)

    def _validate_xy(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int).ravel()
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y_arr.ndim != 1 or len(X_arr) != len(y_arr):
            raise ValueError("y must be a 1D array with the same row count as X.")
        if len(y_arr) == 0:
            raise ValueError("Cannot fit an empty chunk.")
        X_arr = np.where(np.isnan(X_arr), np.nanmedian(X_arr, axis=0), X_arr)
        X_arr = np.where(np.isnan(X_arr), 0.0, X_arr)
        return X_arr, y_arr

    def _majority(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return int(labels[np.argmax(counts)])

    def _impurity(self, y):
        if y.size == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        if self.criterion == "entropy":
            return float(-np.sum(probs * np.log2(probs + 1e-12)))
        return float(1.0 - np.sum(probs**2))

    def _feature_subset(self, n_features):
        if self.max_features is None:
            k = n_features
        elif isinstance(self.max_features, float):
            k = max(1, int(round(n_features * self.max_features)))
        elif self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        else:
            k = max(1, min(n_features, int(self.max_features)))
        return self.rng.choice(n_features, size=k, replace=False)

    def _candidate_thresholds(self, values):
        clean = values[~np.isnan(values)]
        unique = np.unique(clean)
        if unique.size <= 1:
            return np.array([])
        if unique.size <= self.n_thresholds:
            return (unique[:-1] + unique[1:]) / 2.0
        qs = np.linspace(0.05, 0.95, self.n_thresholds)
        return np.unique(np.quantile(clean, qs))

    def _best_split(self, X, y):
        base = self._impurity(y)
        best = (0.0, None, None)
        n = len(y)
        for feature in self._feature_subset(X.shape[1]):
            for threshold in self._candidate_thresholds(X[:, feature]):
                left = X[:, feature] <= threshold
                right = ~left
                if left.sum() == 0 or right.sum() == 0:
                    continue
                score = base - (left.sum() / n) * self._impurity(y[left]) - (right.sum() / n) * self._impurity(y[right])
                if score > best[0]:
                    best = (float(score), int(feature), float(threshold))
        return best

    def _grow(self, X, y, depth):
        node = TreeNode(prediction=self._majority(y))
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.unique(y).size == 1:
            return node
        gain, feature, threshold = self._best_split(X, y)
        if feature is None or gain <= 1e-12:
            return node
        mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def _predict_one(self, row, node):
        while not node.is_leaf:
            value = row[node.feature]
            if np.isnan(value):
                return node.prediction
            node = node.left if value <= node.threshold else node.right
        return node.prediction
