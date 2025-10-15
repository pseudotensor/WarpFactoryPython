"""
Alcubierre warp drive metric implementation
"""

import numpy as np
from typing import List, Optional
from ...core.tensor import Tensor
from ...units.constants import c as speed_of_light
from ..three_plus_one import set_minkowski_three_plus_one, three_plus_one_builder


def shape_function_alcubierre(r: np.ndarray, R: float, sigma: float) -> np.ndarray:
    """
    Alcubierre shape function

    This function defines the warp bubble geometry.

    Args:
        r: Distance from warp bubble center
        R: Radius of the warp bubble
        sigma: Thickness parameter of the bubble

    Returns:
        Shape function values
    """
    f = (np.tanh(sigma * (R + r)) + np.tanh(sigma * (R - r))) / (2 * np.tanh(R * sigma))
    return f


def get_alcubierre_metric(
    grid_size: List[int],
    world_center: List[float],
    velocity: float,
    radius: float,
    sigma: float,
    grid_scale: Optional[List[float]] = None
) -> Tensor:
    """
    Build the Alcubierre warp drive metric

    Args:
        grid_size: World size in [t, x, y, z]
        world_center: World center location in [t, x, y, z]
        velocity: Speed of the warp drive in factors of c, along the x direction
        radius: Radius of the warp bubble
        sigma: Thickness parameter of the bubble
        grid_scale: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

    Returns:
        Tensor: Alcubierre metric tensor
    """
    if grid_scale is None:
        grid_scale = [1.0, 1.0, 1.0, 1.0]

    c = speed_of_light()

    # Start with Minkowski space in 3+1 form
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    # Add the Alcubierre modification to the shift vector
    for i in range(grid_size[1]):  # x
        for j in range(grid_size[2]):  # y
            for k in range(grid_size[3]):  # z
                # Find grid center x, y, z
                x = i * grid_scale[1] - world_center[1]
                y = j * grid_scale[2] - world_center[2]
                z = k * grid_scale[3] - world_center[3]

                for t in range(grid_size[0]):  # time
                    # Determine the x offset of the center of the bubble, centered in time
                    xs = (t * grid_scale[0] - world_center[0]) * velocity * c

                    # Find the radius from the center of the bubble
                    r = np.sqrt((x - xs)**2 + y**2 + z**2)

                    # Find shape function at this point in r
                    fs = shape_function_alcubierre(r, radius, sigma)

                    # Add alcubierre modification to shift vector along x
                    beta[0][t, i, j, k] = -velocity * fs

    # Build the metric tensor from 3+1 components
    metric_dict = three_plus_one_builder(alpha, beta, gamma)

    # Create tensor object
    metric = Tensor(
        tensor=metric_dict,
        tensor_type="metric",
        name="Alcubierre",
        index="covariant",
        coords="cartesian",
        scaling=grid_scale,
        params={
            "gridSize": grid_size,
            "worldCenter": world_center,
            "velocity": velocity,
            "R": radius,
            "sigma": sigma,
        }
    )

    return metric
