"""Streaming pipeline abstraction."""

from __future__ import annotations


class StreamingPipeline:
    """Chain transformers and a final estimator using partial_fit."""

    def __init__(self, stages):
        if not stages:
            raise ValueError("Pipeline requires at least one stage.")
        self.stages = list(stages)

    def partial_fit(self, X, y=None):
        current = X
        for name, stage in self.stages[:-1]:
            if hasattr(stage, "partial_fit"):
                stage.partial_fit(current, y)
            if not hasattr(stage, "transform"):
                raise TypeError(f"Stage {name!r} is not a transformer.")
            current = stage.transform(current)
        final_name, final_stage = self.stages[-1]
        if not hasattr(final_stage, "partial_fit"):
            raise TypeError(f"Final stage {final_name!r} does not support partial_fit.")
        final_stage.partial_fit(current, y)
        return self

    def fit(self, X, y=None):
        return self.partial_fit(X, y)

    def transform(self, X):
        current = X
        for name, stage in self.stages:
            if not hasattr(stage, "transform"):
                raise TypeError(f"Stage {name!r} is not a transformer.")
            current = stage.transform(current)
        return current

    def predict(self, X):
        current = X
        for name, stage in self.stages[:-1]:
            current = stage.transform(current)
        final_name, final_stage = self.stages[-1]
        if not hasattr(final_stage, "predict"):
            raise TypeError(f"Final stage {final_name!r} does not support predict.")
        return final_stage.predict(current)

    def named_steps(self):
        return {name: stage for name, stage in self.stages}


Pipeline = StreamingPipeline
