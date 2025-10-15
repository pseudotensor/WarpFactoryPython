"""
Lentz warp drive metric implementation
"""

import numpy as np
from typing import List, Optional, Tuple
from ...core.tensor import Tensor
from ...units.constants import c as speed_of_light
from ..three_plus_one import set_minkowski_three_plus_one, three_plus_one_builder
from ...core.tensor_ops import get_array_module


def get_warp_factor_by_region(
    x_in: float,
    y_in: float,
    size_scale: float
) -> Tuple[float, float]:
    """
    Calculate the Lentz warp factors based on spatial position

    The Lentz metric uses a piecewise constant shift vector template that
    divides space into distinct regions, each with specific warp factor values.
    This creates a discontinuous but analytically simple warp bubble geometry.

    Args:
        x_in: X coordinate (relative to bubble center)
        y_in: Y coordinate (relative to bubble center)
        size_scale: Sizing factor of the Lentz soliton template

    Returns:
        Tuple[float, float]: (WFX, WFY) warp factors in x and y directions
    """
    x = x_in
    y = abs(y_in)
    WFX = 0.0
    WFY = 0.0

    # Lentz shift vector template - piecewise constant regions
    # Region 1: Right triangle region
    if (x >= size_scale and x <= 2 * size_scale) and (x - size_scale >= y):
        WFX = -2.0
        WFY = 0.0
    # Region 2: Upper right diagonal region
    elif (x > size_scale and x <= 2 * size_scale) and \
         (x - size_scale <= y) and (-y + 3 * size_scale >= x):
        WFX = -1.0
        WFY = 1.0
    # Region 3: Center upper vertical region
    elif (x > 0 and x <= size_scale) and \
         (x + size_scale > y) and (-y + size_scale < x):
        WFX = 0.0
        WFY = 1.0
    # Region 4: Center upper diagonal region
    elif (x > 0 and x <= size_scale) and \
         (x + size_scale <= y) and (-y + 3 * size_scale >= x):
        WFX = -0.5
        WFY = 0.5
    # Region 5: Left upper diagonal region
    elif (x > -size_scale and x <= 0) and \
         (-x + size_scale < y) and (-y + 3 * size_scale >= -x):
        WFX = 0.5
        WFY = 0.5
    # Region 6: Left center region
    elif (x > -size_scale and x <= 0) and \
         (x + size_scale <= y) and (-y + size_scale >= x):
        WFX = 1.0
        WFY = 0.0
    # Region 7: Center horizontal region
    elif (x >= -size_scale and x <= size_scale) and (x + size_scale > y):
        WFX = 1.0
        WFY = 0.0

    # Restore the sign of y to WFY
    WFY = np.sign(y_in) * WFY

    return WFX, WFY


def get_lentz_metric(
    grid_size: List[int],
    world_center: List[float],
    velocity: float,
    scale: Optional[float] = None,
    grid_scale: Optional[List[float]] = None
) -> Tensor:
    """
    Build the Lentz warp drive metric

    The Lentz metric is a soliton-based solution to Einstein's field equations
    that creates a warp bubble using a discontinuous shift vector. The metric
    uses piecewise constant warp factors defined over specific geometric regions,
    creating a simpler but less smooth warp bubble compared to the Alcubierre metric.

    Args:
        grid_size: World size in [t, x, y, z]
        world_center: World center location in [t, x, y, z]
        velocity: Speed of the warp drive in factors of c, along the x direction
        scale: Sizing factor of the Lentz soliton template
               (default: max(grid_size[1:4])/7)
        grid_scale: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

    Returns:
        Tensor: Lentz metric tensor

    Notes:
        The Lentz metric divides space into geometric regions with constant
        shift vector values, creating a discontinuous but analytically tractable
        warp bubble. The shift vector has both x and y components, unlike the
        Alcubierre metric which only has an x component.
    """
    # Handle default arguments
    if scale is None:
        scale = max(grid_size[1:4]) / 7.0

    if grid_scale is None:
        grid_scale = [1.0, 1.0, 1.0, 1.0]

    c = speed_of_light()

    # Start with Minkowski space in 3+1 form
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    # Add the Lentz soliton modifications to the shift vector
    for i in range(grid_size[1]):  # x
        for j in range(grid_size[2]):  # y
            # Calculate spatial coordinates relative to world center
            x = i * grid_scale[1] - world_center[1]
            y = j * grid_scale[2] - world_center[2]

            for k in range(grid_size[3]):  # z (not used in Lentz template)
                for t in range(grid_size[0]):  # time
                    # Determine the x offset of the center of the bubble, centered in time
                    xs = (t * grid_scale[0] - world_center[0]) * velocity * c

                    # Calculate position relative to moving bubble center
                    xp = x - xs

                    # Get Lentz template warp factor values for this position
                    WFX, WFY = get_warp_factor_by_region(xp, y, scale)

                    # Assign dxdt term (x component of shift vector)
                    beta[0][t, i, j, k] = -WFX * velocity

                    # Assign dydt term (y component of shift vector)
                    beta[1][t, i, j, k] = WFY * velocity

    # Build the metric tensor from 3+1 components
    metric_dict = three_plus_one_builder(alpha, beta, gamma)

    # Create tensor object
    metric = Tensor(
        tensor=metric_dict,
        tensor_type="metric",
        name="Lentz",
        index="covariant",
        coords="cartesian",
        scaling=grid_scale,
        params={
            "gridSize": grid_size,
            "worldCenter": world_center,
            "velocity": velocity,
            "scale": scale,
        }
    )

    return metric
