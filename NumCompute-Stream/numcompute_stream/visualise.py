"""Reusable matplotlib plots for streaming experiments."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _finish(fig, save_path=None, show: bool = False):
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    if show:
        plt.show()
    return fig, fig.axes[0]


def plot_metric_over_time(metric_values, title="Metric Over Time", ylabel="Metric", save_path=None, show: bool = False):
    values = np.asarray(metric_values, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(values) + 1), values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model A", "Model B"), title="Model Comparison", save_path=None, show: bool = False):
    a = np.asarray(metric1, dtype=float)
    b = np.asarray(metric2, dtype=float)
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(a) + 1), a, marker="o", label=labels[0])
    ax.plot(np.arange(1, len(b) + 1), b, marker="s", label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs Ground Truth", save_path=None, show: bool = False):
    truth = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    fig, ax = plt.subplots()
    x = np.arange(len(truth))
    ax.scatter(x, truth, label="Ground Truth", marker="o")
    ax.scatter(x, pred, label="Prediction", marker="x")
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Class")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _finish(fig, save_path, show)
