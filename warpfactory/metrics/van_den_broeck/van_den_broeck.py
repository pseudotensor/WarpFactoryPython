"""
Van Den Broeck warp drive metric implementation
"""

import numpy as np
from typing import List, Optional
from ...core.tensor import Tensor
from ...units.constants import c as speed_of_light
from ..alcubierre import shape_function_alcubierre


def get_van_den_broeck_metric(
    grid_size: List[int],
    world_center: List[float],
    v: float,
    R1: float,
    sigma1: float,
    R2: float,
    sigma2: float,
    A: float,
    grid_scale: Optional[List[float]] = None
) -> Tensor:
    """
    Build the Van Den Broeck warp drive metric

    The Van Den Broeck metric modifies the Alcubierre metric by introducing
    a spatial expansion factor B that creates a larger interior volume while
    maintaining a small exterior profile.

    Args:
        grid_size: World size in [t, x, y, z]
        world_center: World center location in [t, x, y, z]
        v: Speed of the warp drive in factors of c, along the x direction
        R1: Spatial expansion radius of the warp bubble
        sigma1: Width factor of the spatial expansion transition
        R2: Shift vector radius of the warp bubble
        sigma2: Width factor of the shift vector transition
        A: Spatial expansion factor
        grid_scale: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

    Returns:
        Tensor: Van Den Broeck metric tensor
    """
    if grid_scale is None:
        grid_scale = [1.0, 1.0, 1.0, 1.0]

    c = speed_of_light()

    # Effective velocity accounting for spatial expansion
    velocity = v * (1 + A)**2

    # Initialize metric tensor dictionary with Minkowski values
    # For Van Den Broeck, we use the covariant form directly
    tensor_dict = {}

    # Initialize all components as 4D arrays
    for i in range(4):
        for j in range(4):
            if i == j:
                if i == 0:
                    # g_tt = -1 for Minkowski base
                    tensor_dict[(i, j)] = -np.ones(grid_size)
                else:
                    # g_xx, g_yy, g_zz = 1 for Minkowski base
                    tensor_dict[(i, j)] = np.ones(grid_size)
            else:
                # Off-diagonal elements start at 0
                tensor_dict[(i, j)] = np.zeros(grid_size)

    # Van Den Broeck modification
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

                    # Define the B function value in Van Den Broeck
                    # B controls spatial expansion
                    B = 1 + shape_function_alcubierre(r, R1, sigma1) * A

                    # Define the f function value in Van Den Broeck
                    # fs controls the shift vector
                    fs = shape_function_alcubierre(r, R2, sigma2) * v

                    # Assign fs and B to the proper terms
                    # Spatial components get B^2 factor
                    tensor_dict[(1, 1)][t, i, j, k] = B**2  # g_xx
                    tensor_dict[(2, 2)][t, i, j, k] = B**2  # g_yy
                    tensor_dict[(3, 3)][t, i, j, k] = B**2  # g_zz

                    # Off-diagonal time-space component (shift vector)
                    tensor_dict[(0, 1)][t, i, j, k] = -B**2 * fs  # g_tx
                    tensor_dict[(1, 0)][t, i, j, k] = -B**2 * fs  # g_xt (symmetric)

                    # Time-time component
                    tensor_dict[(0, 0)][t, i, j, k] = -(1 - B**2 * fs**2)  # g_tt

    # Create tensor object
    metric = Tensor(
        tensor=tensor_dict,
        tensor_type="metric",
        name="Van Den Broeck",
        index="covariant",
        coords="cartesian",
        scaling=grid_scale,
        params={
            "gridSize": grid_size,
            "worldCenter": world_center,
            "velocity": velocity,
            "R1": R1,
            "sigma1": sigma1,
            "R2": R2,
            "sigma2": sigma2,
            "A": A,
        }
    )

    return metric
