import matplotlib
matplotlib.use("Agg")
from numcompute_stream.visualise import plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth


def test_plot_metric_save(tmp_path):
    path = tmp_path / "metric.png"
    fig, ax = plot_metric_over_time([0.1, 0.2], save_path=path)
    assert path.exists()


def test_compare_models_save(tmp_path):
    path = tmp_path / "compare.png"
    compare_models([0.1, 0.2], [0.2, 0.3], save_path=path)
    assert path.exists()


def test_predictions_plot_save(tmp_path):
    path = tmp_path / "pred.png"
    plot_predictions_vs_ground_truth([0, 1], [0, 1], save_path=path)
    assert path.exists()
