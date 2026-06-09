# NumCompute-Stream

NumCompute-Stream is a small NumPy-based streaming machine learning package. It supports incremental preprocessing, streaming metrics, a decision tree classifier, a tree ensemble classifier, pipeline training, benchmarking, and matplotlib visualisation.

## Features

- CSV loading and stream chunk generation
- `partial_fit()` support for preprocessing, tree, ensemble, pipeline, and stream trainer
- Decision tree classifier with Gini or entropy splitting
- Random-forest-style ensemble using bootstrap sampling and majority voting
- Streaming accuracy, precision, recall, F1, and confusion matrix
- Streaming statistics including mean, variance, quantile, and histogram
- Built-in visualisation through `visualise.py`

## Installation

```powershell
python -m pip install -e .
```

## Run tests

```powershell
pytest
```

## Run demo

```powershell
python demo/stream_demo.py
```

The demo loads `demo/sample_stream_data.csv`, splits it into chunks, trains a single tree and an ensemble incrementally, then saves plots into the `demo/` folder.

## Run benchmark

```powershell
python benchmark/benchmark_stream.py
```

## Example

```python
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import EnsembleClassifier
from numcompute_stream.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("forest", EnsembleClassifier(n_estimators=5, max_depth=4))
])

pipe.partial_fit(X_chunk, y_chunk)
predictions = pipe.predict(X_chunk)
```

