"""
Pytest configuration and fixtures for WarpFactory tests
"""

import pytest
import numpy as np
from warpfactory.core.tensor import Tensor
from warpfactory.metrics.minkowski import get_minkowski_metric


@pytest.fixture
def small_grid_size():
    """Small grid size for quick tests"""
    return [5, 5, 5, 5]


@pytest.fixture
def medium_grid_size():
    """Medium grid size for standard tests"""
    return [10, 10, 10, 10]


@pytest.fixture
def grid_scale():
    """Standard grid scaling"""
    return [1.0, 1.0, 1.0, 1.0]


@pytest.fixture
def minkowski_metric_small(small_grid_size, grid_scale):
    """Minkowski metric on small grid"""
    return get_minkowski_metric(small_grid_size, grid_scale)


@pytest.fixture
def minkowski_metric_medium(medium_grid_size, grid_scale):
    """Minkowski metric on medium grid"""
    return get_minkowski_metric(medium_grid_size, grid_scale)


@pytest.fixture
def zero_tensor_small(small_grid_size):
    """Zero metric tensor on small grid"""
    return Tensor.zeros(
        small_grid_size,
        tensor_type="metric",
        name="Zero",
        index="covariant"
    )


@pytest.fixture
def sample_4d_array(small_grid_size):
    """Sample 4D numpy array"""
    return np.random.randn(*small_grid_size)


@pytest.fixture
def identity_3x3():
    """Identity 3x3 cell array"""
    return {
        (0, 0): np.array([[[1.0]]]),
        (0, 1): np.array([[[0.0]]]),
        (0, 2): np.array([[[0.0]]]),
        (1, 0): np.array([[[0.0]]]),
        (1, 1): np.array([[[1.0]]]),
        (1, 2): np.array([[[0.0]]]),
        (2, 0): np.array([[[0.0]]]),
        (2, 1): np.array([[[0.0]]]),
        (2, 2): np.array([[[1.0]]]),
    }


@pytest.fixture
def identity_4x4():
    """Identity 4x4 cell array"""
    result = {}
    for i in range(4):
        for j in range(4):
            if i == j:
                result[(i, j)] = np.array([[[1.0]]])
            else:
                result[(i, j)] = np.array([[[0.0]]])
    return result


@pytest.fixture
def world_center(medium_grid_size):
    """World center for medium grid"""
    return [gs / 2.0 for gs in medium_grid_size]


@pytest.fixture
def alcubierre_params():
    """Standard Alcubierre warp drive parameters"""
    return {
        'velocity': 1.0,  # factors of c
        'radius': 2.0,
        'sigma': 5.0
    }


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for floating point comparisons"""
    return 1e-10


@pytest.fixture
def loose_tolerance():
    """Loose tolerance for approximate calculations"""
    return 1e-6
