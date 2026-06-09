import numpy as np
from numcompute_stream.stats import StreamingStats, RunningMean


def test_streaming_stats_mean():
    s = StreamingStats()
    s.update_stats([[1, 2], [3, 4]])
    s.update_stats([[5, 6]])
    assert np.allclose(s.mean(), [3, 4])


def test_streaming_stats_variance():
    s = StreamingStats().update_stats([[1], [2], [3]])
    assert np.allclose(s.variance(), [2 / 3])


def test_streaming_stats_quantile():
    s = StreamingStats().update_stats([[1], [2], [3]])
    assert np.allclose(s.quantile(0.5), [2])


def test_streaming_stats_histogram():
    hist, edges = StreamingStats().update_stats([[1], [2], [3]]).histogram(bins=2)
    assert hist.sum() == 3


def test_running_mean_wrapper():
    rm = RunningMean().update([[2], [4]])
    assert np.allclose(rm.result(), [3])
