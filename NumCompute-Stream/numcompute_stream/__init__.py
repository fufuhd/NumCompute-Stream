"""NumCompute-Stream: streaming tree-based machine learning utilities."""

from .tree import DecisionTreeClassifier
from .ensemble import EnsembleClassifier, StreamingForestClassifier
from .pipeline import Pipeline, StreamingPipeline
from .preprocessing import StandardScaler, SimpleImputer, MinMaxScaler
from .stream import StreamTrainer

__all__ = [
    "DecisionTreeClassifier",
    "EnsembleClassifier",
    "StreamingForestClassifier",
    "Pipeline",
    "StreamingPipeline",
    "StandardScaler",
    "SimpleImputer",
    "MinMaxScaler",
    "StreamTrainer",
]
