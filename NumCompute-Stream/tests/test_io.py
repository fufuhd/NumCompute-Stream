import numpy as np
from numcompute_stream.io import make_stream_batches, train_test_split, load_csv


def test_make_stream_batches_count():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    chunks = make_stream_batches(X, y, 4)
    assert len(chunks) == 3


def test_make_stream_batches_shapes():
    X = np.arange(12).reshape(6, 2)
    y = np.arange(6)
    chunks = make_stream_batches(X, y, 3)
    assert chunks[0][0].shape == (3, 2)


def test_train_test_split_length():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1)
    assert len(Xte) == 4 and len(Xtr) == 16


def test_load_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("a,b,y\n1,2,0\n3,4,1\n")
    X, y = load_csv(p, target_column="y")
    assert X.shape == (2, 2)
    assert y.tolist() == [0, 1]
