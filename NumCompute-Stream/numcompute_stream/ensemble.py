"""Streaming ensemble models built from decision trees."""

from __future__ import annotations

import numpy as np

from .tree import DecisionTreeClassifier


class StreamingForestClassifier:
    """Random-forest-style bagging ensemble with partial_fit support."""

    def __init__(
        self,
        n_estimators: int = 5,
        max_depth: int = 5,
        min_samples_split: int = 2,
        max_features: int | float | str | None = "sqrt",
        criterion: str = "gini",
        buffer_size: int = 512,
        random_state: int | None = None,
    ):
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        self.n_estimators = int(n_estimators)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        seeds = self.rng.integers(0, 2**31 - 1, size=self.n_estimators)
        self.members = [
            DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                criterion=criterion,
                buffer_size=buffer_size,
                random_state=int(seed),
            )
            for seed in seeds
        ]
        self.classes_: np.ndarray | None = None

    def partial_fit(self, X_chunk, y_chunk):
        X = np.asarray(X_chunk, dtype=float)
        y = np.asarray(y_chunk, dtype=int).ravel()
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(X) != len(y):
            raise ValueError("X and y length mismatch.")
        self.classes_ = np.unique(y) if self.classes_ is None else np.unique(np.concatenate([self.classes_, np.unique(y)]))
        for tree in self.members:
            sample_idx = self.rng.integers(0, len(y), size=len(y))
            tree.partial_fit(X[sample_idx], y[sample_idx])
        return self

    def fit(self, X, y):
        return self.partial_fit(X, y)

    def predict(self, X):
        if any(tree.root_ is None for tree in self.members):
            raise RuntimeError("Ensemble has not been fitted.")
        votes = np.vstack([tree.predict(X) for tree in self.members])
        return np.apply_along_axis(self._vote, axis=0, arr=votes).astype(int)

    @staticmethod
    def _vote(column):
        labels, counts = np.unique(column, return_counts=True)
        return labels[np.argmax(counts)]


EnsembleClassifier = StreamingForestClassifier
RandomForestClassifier = StreamingForestClassifier
