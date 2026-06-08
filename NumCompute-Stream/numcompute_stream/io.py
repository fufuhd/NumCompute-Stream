"""Small CSV and streaming-batch helpers using only Python and NumPy."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


def _normalise_target_index(target_column: int | str, header: list[str] | None, width: int) -> int:
    if isinstance(target_column, str):
        if header is None:
            raise ValueError("A string target_column requires header=True.")
        if target_column not in header:
            raise ValueError(f"Unknown target column: {target_column!r}.")
        return header.index(target_column)
    index = int(target_column)
    if index < 0:
        index += width
    if not 0 <= index < width:
        raise IndexError("target_column is outside the CSV width.")
    return index


def load_csv(path: str | Path, target_column: int | str = -1, header: bool = True, delimiter: str = ","):
    """Load a numeric CSV and split it into feature matrix X and label vector y.

    Missing numeric cells are converted to np.nan. The target column is returned
    as integer labels when possible; otherwise it is encoded in sorted order.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty.")

    names = rows[0] if header else None
    data_rows = rows[1:] if header else rows
    if not data_rows:
        raise ValueError("CSV file has no data rows.")

    width = len(data_rows[0])
    if any(len(row) != width for row in data_rows):
        raise ValueError("CSV rows have inconsistent column counts.")

    target_idx = _normalise_target_index(target_column, names, width)
    feature_rows: list[list[float]] = []
    target_values: list[str] = []

    for row in data_rows:
        feature_row = []
        for col_idx, cell in enumerate(row):
            if col_idx == target_idx:
                target_values.append(cell.strip())
            else:
                text = cell.strip()
                feature_row.append(float(text) if text else np.nan)
        feature_rows.append(feature_row)

    X = np.asarray(feature_rows, dtype=float)
    try:
        y = np.asarray([int(float(v)) for v in target_values], dtype=int)
    except ValueError:
        classes = {label: i for i, label in enumerate(sorted(set(target_values)))}
        y = np.asarray([classes[v] for v in target_values], dtype=int)
    return X, y


def make_stream_batches(X, y, chunk_size: int, drop_last: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split arrays into ordered streaming chunks."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if X_arr.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y_arr.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y must contain the same number of rows.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    chunks = []
    for start in range(0, len(X_arr), chunk_size):
        stop = start + chunk_size
        if stop > len(X_arr) and drop_last:
            break
        chunks.append((X_arr[start:stop].copy(), y_arr[start:stop].copy()))
    return chunks


def train_test_split(X, y, test_size: float = 0.25, shuffle: bool = True, random_state: int | None = None):
    """Simple train/test split without external dependencies."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if len(X_arr) != len(y_arr):
        raise ValueError("X and y length mismatch.")

    indices = np.arange(len(X_arr))
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    test_n = max(1, int(round(len(indices) * test_size)))
    test_idx = indices[:test_n]
    train_idx = indices[test_n:]
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


make_chunks = make_stream_batches
