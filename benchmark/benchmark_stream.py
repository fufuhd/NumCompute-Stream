import time
import numpy as np

from numcompute_stream.io import make_stream_batches
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer


def make_data(n=400, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = ((X[:, 0] + 0.7 * X[:, 1] - X[:, 2]) > 0).astype(int)
    return X, y


def run_model(name, model):
    X, y = make_data()
    chunks = make_stream_batches(X, y, chunk_size=40)
    trainer = StreamTrainer(model)
    scores = []
    start = time.perf_counter()
    for Xc, yc in chunks:
        trainer.fit_chunk(Xc, yc)
        scores.append(trainer.score_chunk(Xc, yc))
    total = time.perf_counter() - start
    print(f"Model: {name}")
    print(f"Average fit+score time: {total / len(chunks):.6f} seconds")
    print(f"Final chunk accuracy: {scores[-1]:.3f}")
    print(f"Cumulative accuracy: {trainer.cumulative_accuracy:.3f}")
    print()


if __name__ == "__main__":
    tree = Pipeline([
        ("scale", StandardScaler()),
        ("tree", DecisionTreeClassifier(max_depth=4, random_state=1)),
    ])
    forest = Pipeline([
        ("scale", StandardScaler()),
        ("forest", EnsembleClassifier(n_estimators=5, max_depth=4, random_state=1)),
    ])
    run_model("DecisionTree", tree)
    run_model("StreamingForest", forest)
