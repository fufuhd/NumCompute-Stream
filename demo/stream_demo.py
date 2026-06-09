from pathlib import Path

from numcompute_stream.io import load_csv, make_stream_batches
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.visualise import plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth

BASE = Path(__file__).resolve().parent
X, y = load_csv(BASE / "sample_stream_data.csv", target_column="target", header=True)
chunks = make_stream_batches(X, y, chunk_size=8)

single_tree = Pipeline([
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=3)),
])
forest = Pipeline([
    ("scale", StandardScaler()),
    ("forest", EnsembleClassifier(n_estimators=5, max_depth=4, random_state=3)),
])

tree_runner = StreamTrainer(single_tree)
forest_runner = StreamTrainer(forest)

tree_scores = []
forest_scores = []
last_true = None
last_pred = None

for X_chunk, y_chunk in chunks:
    tree_runner.fit_chunk(X_chunk, y_chunk)
    tree_scores.append(tree_runner.score_chunk(X_chunk, y_chunk))

    forest_runner.fit_chunk(X_chunk, y_chunk)
    forest_scores.append(forest_runner.score_chunk(X_chunk, y_chunk))
    last_true = y_chunk
    last_pred = forest.predict(X_chunk)

print("Tree scores:", tree_scores)
print("Forest scores:", forest_scores)
print("Tree cumulative accuracy:", tree_runner.cumulative_accuracy)
print("Forest cumulative accuracy:", forest_runner.cumulative_accuracy)

plot_metric_over_time(tree_scores, "Decision Tree Accuracy Over Chunks", "Accuracy", BASE / "tree_accuracy.png")
compare_models(tree_scores, forest_scores, labels=("Tree", "Forest"), save_path=BASE / "model_comparison.png")
plot_predictions_vs_ground_truth(last_true, last_pred, save_path=BASE / "predictions_vs_truth.png")
print("Saved plots into demo/ folder.")
