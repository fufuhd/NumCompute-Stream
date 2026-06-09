import numpy as np
import pytest
from numcompute_stream.ensemble import EnsembleClassifier


def data():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 2]], dtype=float)
    y = np.array([0, 0, 1, 1, 1, 1])
    return X, y


def test_ensemble_predict_shape():
    X, y = data()
    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=2).partial_fit(X, y)
    assert model.predict(X).shape == (6,)


def test_ensemble_streaming_updates():
    X, y = data()
    model = EnsembleClassifier(n_estimators=3, max_depth=2, random_state=2)
    model.partial_fit(X[:3], y[:3])
    model.partial_fit(X[3:], y[3:])
    assert len(model.predict(X)) == 6


def test_ensemble_requires_positive_estimators():
    with pytest.raises(ValueError):
        EnsembleClassifier(n_estimators=0)
