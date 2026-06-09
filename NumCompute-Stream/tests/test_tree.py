import numpy as np
import pytest
from numcompute_stream.tree import DecisionTreeClassifier


def toy():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    return X, y


def test_tree_predict_shape():
    X, y = toy()
    clf = DecisionTreeClassifier(max_depth=2, random_state=1).partial_fit(X, y)
    assert clf.predict(X).shape == (4,)


def test_tree_learns_simple_rule():
    X, y = toy()
    clf = DecisionTreeClassifier(max_depth=2, random_state=1).partial_fit(X, y)
    assert (clf.predict(X) == y).mean() >= 0.75


def test_tree_multiple_partial_fit():
    X, y = toy()
    clf = DecisionTreeClassifier(max_depth=2, random_state=1)
    clf.partial_fit(X[:2], y[:2])
    clf.partial_fit(X[2:], y[2:])
    assert len(clf.predict(X)) == 4


def test_tree_handles_nan():
    X, y = toy()
    X[0, 0] = np.nan
    clf = DecisionTreeClassifier(max_depth=2, random_state=1).partial_fit(X, y)
    assert np.isfinite(clf.predict(X)).all()


def test_tree_unfitted_error():
    with pytest.raises(RuntimeError):
        DecisionTreeClassifier().predict([[1, 2]])
