import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier


def test_pipeline_predict():
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(max_depth=2, random_state=1))])
    pipe.partial_fit(X, y)
    assert pipe.predict(X).shape == (4,)


def test_pipeline_named_steps():
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier())])
    assert "scale" in pipe.named_steps()


def test_pipeline_multiple_chunks():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(max_depth=2, random_state=1))])
    pipe.partial_fit(X[:2], y[:2])
    pipe.partial_fit(X[2:], y[2:])
    assert len(pipe.predict(X)) == 4
