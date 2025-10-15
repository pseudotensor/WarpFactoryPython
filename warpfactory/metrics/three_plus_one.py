"""
3+1 decomposition utilities for spacetime metrics

The 3+1 decomposition splits spacetime into spatial slices and time evolution,
decomposing the metric into:
- alpha: lapse function (proper time rate)
- beta: shift vector (spatial coordinate motion)
- gamma: spatial metric on each time slice
"""

import numpy as np
from typing import Dict, List, Tuple
from ..core.tensor import Tensor
from ..core.tensor_ops import c3_inv, change_tensor_index, get_array_module


def set_minkowski_three_plus_one(grid_size: List[int]) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Return the 3+1 format for flat Minkowski space

    Args:
        grid_size: World size in [t, x, y, z]

    Returns:
        Tuple of (alpha, beta, gamma):
            - alpha: Lapse rate 4D array
            - beta: Shift vector, dictionary with keys 0,1,2 and 4D array values
            - gamma: Spatial terms, dictionary with (i,j) keys and 4D array values
    """
    alpha = np.ones(grid_size)

    beta = {}
    for i in range(3):
        beta[i] = np.zeros(grid_size)

    gamma = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                gamma[(i, j)] = np.ones(grid_size)
            else:
                gamma[(i, j)] = np.zeros(grid_size)

    return alpha, beta, gamma


def three_plus_one_builder(
    alpha: np.ndarray,
    beta: Dict[int, np.ndarray],
    gamma: Dict[tuple, np.ndarray]
) -> Dict[tuple, np.ndarray]:
    """
    Build the metric tensor from 3+1 components

    Args:
        alpha: (TxXxYxZ) lapse rate map across spacetime
        beta: Dictionary of shift vector components (covariant assumed)
        gamma: Dictionary of spatial metric components (covariant assumed)

    Returns:
        Dictionary representing the 4x4 metric tensor
    """
    xp = get_array_module(alpha)
    s = alpha.shape

    # Get contravariant spatial metric
    gamma_up = c3_inv(gamma)

    # Calculate contravariant beta (beta^i = gamma^ij * beta_j)
    beta_up = {}
    for i in range(3):
        beta_up[i] = xp.zeros(s)
        for j in range(3):
            beta_up[i] = beta_up[i] + gamma_up[(i, j)] * beta[j]

    # Create metric tensor
    metric_tensor = {}

    # Time-time component: g_00 = -alpha^2 + beta^i * beta_i
    metric_tensor[(0, 0)] = -alpha**2
    for i in range(3):
        metric_tensor[(0, 0)] = metric_tensor[(0, 0)] + beta_up[i] * beta[i]

    # Time-space components: g_0i = beta_i
    for i in range(3):
        metric_tensor[(0, i+1)] = beta[i]
        metric_tensor[(i+1, 0)] = beta[i]

    # Space-space components: g_ij = gamma_ij
    for i in range(3):
        for j in range(3):
            metric_tensor[(i+1, j+1)] = gamma[(i, j)]

    return metric_tensor


def three_plus_one_decomposer(metric: Tensor) -> Tuple[np.ndarray, Dict, Dict, Dict, Dict]:
    """
    Decompose a metric tensor into 3+1 components

    Args:
        metric: Metric tensor object

    Returns:
        Tuple of (alpha, beta_down, gamma_down, beta_up, gamma_up):
            - alpha: Lapse rate 4D array
            - beta_down: Covariant shift vector
            - gamma_down: Covariant spatial metric
            - beta_up: Contravariant shift vector
            - gamma_up: Contravariant spatial metric
    """
    # Ensure metric is covariant
    metric = change_tensor_index(metric, "covariant")

    xp = get_array_module(metric[(0, 0)])
    s = metric.shape

    # Extract covariant shift vector (g_0i)
    beta_down = {}
    for i in range(3):
        beta_down[i] = metric[(0, i+1)]

    # Extract covariant spatial metric (g_ij)
    gamma_down = {}
    for i in range(3):
        for j in range(3):
            gamma_down[(i, j)] = metric[(i+1, j+1)]

    # Get contravariant spatial metric
    gamma_up = c3_inv(gamma_down)

    # Calculate contravariant shift vector
    beta_up = {}
    for i in range(3):
        beta_up[i] = xp.zeros(s)
        for j in range(3):
            beta_up[i] = beta_up[i] + gamma_up[(i, j)] * beta_down[j]

    # Calculate lapse function
    # alpha = sqrt(beta^i * beta_i - g_00)
    beta_squared = xp.zeros(s)
    for i in range(3):
        beta_squared = beta_squared + beta_up[i] * beta_down[i]

    alpha = xp.sqrt(beta_squared - metric[(0, 0)])

    return alpha, beta_down, gamma_down, beta_up, gamma_up
