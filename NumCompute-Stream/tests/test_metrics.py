import numpy as np
from numcompute_stream.metrics import accuracy_score, precision_score, recall_score, f1_score, StreamingAccuracy, StreamingF1, ConfusionMatrix


def test_accuracy_score():
    assert accuracy_score([1, 0, 1], [1, 1, 1]) == 2 / 3


def test_precision_score():
    assert precision_score([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5


def test_recall_score():
    assert recall_score([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5


def test_f1_score():
    assert f1_score([1, 0, 1, 0], [1, 1, 0, 0]) == 0.5


def test_streaming_accuracy_two_updates():
    m = StreamingAccuracy()
    m.update([1, 0], [1, 1]).update([0, 1], [0, 1])
    assert m.result() == 0.75


def test_streaming_f1():
    m = StreamingF1().update([1, 0, 1, 0], [1, 1, 0, 0])
    assert m.result() == 0.5


def test_confusion_matrix_shape():
    cm = ConfusionMatrix(labels=[0, 1]).update([0, 1, 1], [0, 0, 1]).result()
    assert cm.shape == (2, 2)
    assert cm.sum() == 3
