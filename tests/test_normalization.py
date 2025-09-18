import numpy as np

from vibrator.normalization import ensure_numpy, layer_normalize, l2_normalize


def test_layer_normalize_zero_mean_unit_variance():
    data = np.array([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    normalized = layer_normalize(data)
    np.testing.assert_allclose(normalized.mean(axis=1), np.zeros(2), atol=1e-6)
    np.testing.assert_allclose(normalized.std(axis=1)[0], 1.0, atol=1e-5)
    assert normalized.std(axis=1)[1] == 0.0


def test_l2_normalize_unit_length():
    data = np.array([[3.0, 4.0]], dtype=np.float32)
    normalized = l2_normalize(data)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), np.ones(1))


def test_ensure_numpy_from_list():
    data = [1, 2, 3]
    array = ensure_numpy(data)
    assert isinstance(array, np.ndarray)
    np.testing.assert_array_equal(array, np.array(data))
