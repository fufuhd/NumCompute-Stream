import numpy as np
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.stream import StreamTrainer


def model():
    return Pipeline([("scale", StandardScaler()), ("tree", DecisionTreeClassifier(max_depth=2, random_state=1))])


def test_stream_trainer_history():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    t = StreamTrainer(model())
    t.fit_chunk(X, y)
    score = t.score_chunk(X, y)
    assert "score" in t.history[-1]
    assert 0 <= score <= 1


def test_stream_metric_series():
    X = np.array([[0], [1], [2], [3]], dtype=float)
    y = np.array([0, 0, 1, 1])
    t = StreamTrainer(model())
    t.fit_chunk(X, y)
    t.score_chunk(X, y)
    assert len(t.metric_series()) == 1
