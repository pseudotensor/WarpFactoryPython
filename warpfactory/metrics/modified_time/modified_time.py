"""
Modified Time warp drive metric implementation

The Modified Time metric is a variation of the Alcubierre warp drive that
incorporates modifications to both the shift vector and the lapse function.
Unlike the standard Alcubierre metric which only modifies the shift vector,
this metric also modifies the time-time component (lapse function) using a
lapse rate modification parameter A.
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


def get_modified_time_metric(
    grid_size: List[int],
    world_center: List[float],
    velocity: float,
    radius: float,
    sigma: float,
    A: float,
    grid_scale: Optional[List[float]] = None
) -> Tensor:
    """
    Build the Modified Time warp drive metric

    The Modified Time metric extends the Alcubierre metric by modifying both
    the shift vector and the lapse function. The lapse function modification
    introduces a parameter A that controls the rate of time progression within
    the warp bubble relative to flat space.

    Args:
        grid_size: World size in [t, x, y, z]
        world_center: World center location in [t, x, y, z]
        velocity: Speed of the warp drive in factors of c, along the x direction
        radius: Radius of the warp bubble
        sigma: Thickness parameter of the bubble
        A: Lapse rate modification parameter. Controls time dilation within bubble.
        grid_scale: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

    Returns:
        Tensor: Modified Time metric tensor

    Notes:
        The metric modifies two components:
        1. Shift vector (beta^x): Same as Alcubierre, -v*fs
        2. Lapse function (alpha): Modified to ((1-fs) + fs/A)

        This results in g_00 = -((1-fs) + fs/A)^2 + (fs*v)^2
    """
    if grid_scale is None:
        grid_scale = [1.0, 1.0, 1.0, 1.0]

    c = speed_of_light()

    # Start with Minkowski space in 3+1 form
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    # Add the Modified Time modifications
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

                    # Add modification to lapse function
                    # alpha = (1 - fs) + fs/A
                    alpha[t, i, j, k] = (1.0 - fs) + fs / A

    # Build the metric tensor from 3+1 components
    metric_dict = three_plus_one_builder(alpha, beta, gamma)

    # Create tensor object
    metric = Tensor(
        tensor=metric_dict,
        tensor_type="metric",
        name="Modified Time",
        index="covariant",
        coords="cartesian",
        scaling=grid_scale,
        params={
            "gridSize": grid_size,
            "worldCenter": world_center,
            "velocity": velocity,
            "R": radius,
            "sigma": sigma,
            "A": A,
        }
    )

    return metric
