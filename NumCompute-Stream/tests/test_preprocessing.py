import numpy as np
from numcompute_stream.preprocessing import StandardScaler, SimpleImputer, MinMaxScaler


def test_standard_scaler_partial_fit_mean():
    sc = StandardScaler()
    sc.partial_fit([[1, 2], [3, 4]])
    sc.partial_fit([[5, 6]])
    assert np.allclose(sc.mean_, [3, 4])


def test_standard_scaler_transform_zero_mean():
    X = np.array([[1.0], [2.0], [3.0]])
    out = StandardScaler().fit_transform(X)
    assert abs(out.mean()) < 1e-8


def test_standard_scaler_nan_safe():
    sc = StandardScaler().partial_fit([[1.0, np.nan], [3.0, 4.0]])
    out = sc.transform([[np.nan, 4.0]])
    assert np.isfinite(out).all()


def test_imputer_replaces_nan():
    imp = SimpleImputer().partial_fit([[1.0, np.nan], [3.0, 5.0]])
    out = imp.transform([[np.nan, np.nan]])
    assert np.allclose(out, [[2.0, 5.0]])


def test_minmax_scaler_range():
    out = MinMaxScaler().fit_transform([[1], [3], [5]])
    assert np.allclose(out.ravel(), [0, 0.5, 1])
