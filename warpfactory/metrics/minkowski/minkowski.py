"""
Minkowski metric implementation
"""

import numpy as np
from typing import List, Optional
from ...core.tensor import Tensor


def get_minkowski_metric(
    grid_size: List[int],
    grid_scale: Optional[List[float]] = None
) -> Tensor:
    """
    Build the Minkowski (flat spacetime) metric

    Args:
        grid_size: World size in [t, x, y, z]
        grid_scale: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

    Returns:
        Tensor: Minkowski metric tensor
    """
    if grid_scale is None:
        grid_scale = [1.0, 1.0, 1.0, 1.0]

    # Build metric components
    metric_dict = {}

    # Time-time component: g_00 = -1
    metric_dict[(0, 0)] = -np.ones(grid_size)

    # Space-space diagonal components: g_ii = 1
    metric_dict[(1, 1)] = np.ones(grid_size)
    metric_dict[(2, 2)] = np.ones(grid_size)
    metric_dict[(3, 3)] = np.ones(grid_size)

    # All off-diagonal components are zero
    for i in range(4):
        for j in range(4):
            if i != j:
                metric_dict[(i, j)] = np.zeros(grid_size)

    # Create tensor object
    metric = Tensor(
        tensor=metric_dict,
        tensor_type="metric",
        name="Minkowski",
        index="covariant",
        coords="cartesian",
        scaling=grid_scale,
        params={
            "gridSize": grid_size,
            "worldCenter": [gs // 2 for gs in grid_size],
        }
    )

    return metric
