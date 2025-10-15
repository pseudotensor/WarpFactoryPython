"""
Schwarzschild metric implementation

The Schwarzschild metric describes the spacetime geometry around a
non-rotating, spherically symmetric massive object (e.g., a black hole).
This implementation uses Cartesian coordinates.
"""

import numpy as np
from typing import List, Optional
from ...core.tensor import Tensor


def get_schwarzschild_metric(
    grid_size: List[int],
    world_center: List[float],
    rs: float,
    grid_scaling: Optional[List[float]] = None
) -> Tensor:
    """
    Build the Schwarzschild metric in Cartesian coordinates

    The Schwarzschild metric describes the gravitational field outside a
    spherically symmetric, non-rotating massive object. In Cartesian coordinates,
    the metric has both diagonal and off-diagonal spatial terms.

    Args:
        grid_size: World size in [t, x, y, z]. Time dimension must be 1.
        world_center: World center location in [t, x, y, z]
        rs: Schwarzschild radius (2*G*M/c^2)
        grid_scaling: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

    Returns:
        Tensor: Schwarzschild metric tensor in covariant form

    Raises:
        ValueError: If grid_size[0] > 1 (time grid must be size 1)

    Notes:
        - The metric is computed in Cartesian coordinates (x, y, z)
        - A small epsilon is added to avoid divide-by-zero at the origin
        - The metric includes off-diagonal spatial terms (dxdy, dxdz, dydz)
    """
    if grid_scaling is None:
        grid_scaling = [1.0, 1.0, 1.0, 1.0]

    # Check if gridSize in time is 1 and raise error if not
    if grid_size[0] > 1:
        raise ValueError(
            'The time grid is greater than 1, only a size of 1 can be used '
            'for the Schwarzschild solution'
        )

    # Initialize metric tensor dictionary with Minkowski (flat spacetime) baseline
    metric_dict = {}

    # Set all components to flat spacetime initially
    metric_dict[(0, 0)] = -np.ones(grid_size)
    for i in range(1, 4):
        metric_dict[(i, i)] = np.ones(grid_size)
    for i in range(4):
        for j in range(4):
            if i != j:
                metric_dict[(i, j)] = np.zeros(grid_size)

    # Add very small offset to mitigate divide by zero errors
    epsilon = 1e-10
    t = 0  # Only 1 time slice (Python uses 0-indexing)

    # Loop through spatial grid points
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                # Convert grid indices to spatial coordinates
                # Note: MATLAB is 1-indexed, Python is 0-indexed
                x = (i + 1) * grid_scaling[1] - world_center[1]
                y = (j + 1) * grid_scaling[2] - world_center[2]
                z = (k + 1) * grid_scaling[3] - world_center[3]

                # Compute radial distance
                r = np.sqrt(x**2 + y**2 + z**2) + epsilon

                # Diagonal terms
                # g_tt = -(1 - rs/r)
                metric_dict[(0, 0)][t, i, j, k] = -(1.0 - rs / r)

                # g_xx = (x^2/(1-rs/r) + y^2 + z^2) / r^2
                metric_dict[(1, 1)][t, i, j, k] = (
                    x**2 / (1.0 - rs / r) + y**2 + z**2
                ) / r**2

                # g_yy = (x^2 + y^2/(1-rs/r) + z^2) / r^2
                metric_dict[(2, 2)][t, i, j, k] = (
                    x**2 + y**2 / (1.0 - rs / r) + z**2
                ) / r**2

                # g_zz = (x^2 + y^2 + z^2/(1-rs/r)) / r^2
                metric_dict[(3, 3)][t, i, j, k] = (
                    x**2 + y**2 + z**2 / (1.0 - rs / r)
                ) / r**2

                # Off-diagonal spatial terms
                # dxdy cross terms
                cross_factor = rs / (r**3 - r**2 * rs)
                metric_dict[(1, 2)][t, i, j, k] = cross_factor * x * y
                metric_dict[(2, 1)][t, i, j, k] = cross_factor * x * y

                # dxdz cross terms
                metric_dict[(1, 3)][t, i, j, k] = cross_factor * x * z
                metric_dict[(3, 1)][t, i, j, k] = cross_factor * x * z

                # dydz cross terms
                metric_dict[(2, 3)][t, i, j, k] = cross_factor * y * z
                metric_dict[(3, 2)][t, i, j, k] = cross_factor * y * z

    # Create tensor object
    metric = Tensor(
        tensor=metric_dict,
        tensor_type="metric",
        name="Schwarzschild",
        index="covariant",
        coords="cartesian",
        scaling=grid_scaling,
        params={
            "gridSize": grid_size,
            "worldCenter": world_center,
            "rs": rs,
        }
    )

    return metric
