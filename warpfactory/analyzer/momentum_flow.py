"""
Momentum flow line calculations

This module provides functions for calculating and tracing momentum flow lines
from stress-energy tensors. Momentum flow lines visualize the flow of energy
and momentum through spacetime.
"""

import numpy as np
from typing import List, Tuple
from ..core.tensor import Tensor


def _trilinear_interp(F: np.ndarray, pos: np.ndarray) -> float:
    """
    Trilinear interpolation at a given position in a 3D array

    Reference: https://en.wikipedia.org/wiki/Trilinear_interpolation

    Args:
        F: 3D array to interpolate from
        pos: Position [x, y, z] in index coordinates

    Returns:
        Interpolated value at the position
    """
    # Add small epsilon to avoid division by zero when on grid points
    x = pos + 1e-8

    # Get floor and ceiling indices
    x_floor = int(np.floor(x[0]))
    x_ceil = int(np.ceil(x[0]))
    y_floor = int(np.floor(x[1]))
    y_ceil = int(np.ceil(x[1]))
    z_floor = int(np.floor(x[2]))
    z_ceil = int(np.ceil(x[2]))

    # Calculate interpolation weights
    xd = (x[0] - x_floor) / (x_ceil - x_floor) if x_ceil != x_floor else 0
    yd = (x[1] - y_floor) / (y_ceil - y_floor) if y_ceil != y_floor else 0
    zd = (x[2] - z_floor) / (z_ceil - z_floor) if z_ceil != z_floor else 0

    # Interpolate along x-axis
    c00 = F[x_floor, y_floor, z_floor] * (1 - xd) + F[x_ceil, y_floor, z_floor] * xd
    c01 = F[x_floor, y_floor, z_ceil] * (1 - xd) + F[x_ceil, y_floor, z_ceil] * xd
    c10 = F[x_floor, y_ceil, z_floor] * (1 - xd) + F[x_ceil, y_ceil, z_floor] * xd
    c11 = F[x_floor, y_ceil, z_ceil] * (1 - xd) + F[x_ceil, y_ceil, z_ceil] * xd

    # Interpolate along y-axis
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # Interpolate along z-axis
    c = c0 * (1 - zd) + c1 * zd

    return c


def get_momentum_flow_lines(
    energy_tensor: Tensor,
    start_points: Tuple[np.ndarray, np.ndarray, np.ndarray],
    step_size: float,
    max_steps: int,
    scale_factor: float = 1.0
) -> List[np.ndarray]:
    """
    Calculate momentum flow lines from a stress-energy tensor

    Traces paths through spacetime following the momentum density vector field.
    This is useful for visualizing how energy and momentum flow in warp drive
    spacetimes.

    Args:
        energy_tensor: Stress-energy tensor (must be contravariant)
        start_points: Tuple of (X, Y, Z) arrays defining starting positions
                     Each array contains the coordinates of starting points
        step_size: Step size for path integration
        max_steps: Maximum number of steps to propagate each path
        scale_factor: Scaling factor applied to momentum density (default: 1.0)

    Returns:
        List of paths, where each path is an Nx3 array of positions [x, y, z]

    Raises:
        ValueError: If energy tensor is not contravariant

    Example:
        >>> # Create starting points on a grid
        >>> x = np.array([10, 20, 30])
        >>> y = np.array([15, 25, 35])
        >>> z = np.array([20, 30, 40])
        >>> start_points = (x, y, z)
        >>>
        >>> # Calculate flow lines
        >>> paths = get_momentum_flow_lines(
        ...     energy_tensor=T_contravariant,
        ...     start_points=start_points,
        ...     step_size=0.1,
        ...     max_steps=1000,
        ...     scale_factor=1.0
        ... )
    """
    # Check that the energy tensor is contravariant
    if energy_tensor.index.lower() != "contravariant":
        raise ValueError('Energy tensor for momentum flowlines should be contravariant.')

    # Extract momentum density components (T^{0i})
    # These are the spatial components of the energy flux
    # Note: MATLAB indexing starts at 1, Python at 0
    # MATLAB {1,2} = Python (0,1) for x-momentum
    # MATLAB {1,3} = Python (0,2) for y-momentum
    # MATLAB {1,4} = Python (0,3) for z-momentum
    x_mom = np.squeeze(energy_tensor[(0, 1)]) * scale_factor
    y_mom = np.squeeze(energy_tensor[(0, 2)]) * scale_factor
    z_mom = np.squeeze(energy_tensor[(0, 3)]) * scale_factor

    # Get the shape for bounds checking
    # Note: These are 4D arrays (t, x, y, z), but we use the spatial dimensions
    # We'll use the first time slice for momentum field
    if x_mom.ndim == 4:
        x_mom = x_mom[0, :, :, :]
        y_mom = y_mom[0, :, :, :]
        z_mom = z_mom[0, :, :, :]

    # Reshape starting points to 1D arrays
    start_x = np.array(start_points[0]).flatten()
    start_y = np.array(start_points[1]).flatten()
    start_z = np.array(start_points[2]).flatten()

    # Initialize paths list
    paths = []

    # Trace each path
    for j in range(len(start_x)):
        # Initialize position array
        pos = np.zeros((max_steps, 3))
        pos[0, :] = [start_x[j], start_y[j], start_z[j]]

        # Propagate the path
        for i in range(max_steps - 1):
            # Check if position is outside the grid bounds
            # Use 1-based indexing bounds to match MATLAB behavior
            if (np.any(np.isnan(pos[i, :])) or
                np.floor(pos[i, 0]) <= 0 or np.ceil(pos[i, 0]) >= x_mom.shape[0] - 1 or
                np.floor(pos[i, 1]) <= 0 or np.ceil(pos[i, 1]) >= x_mom.shape[1] - 1 or
                np.floor(pos[i, 2]) <= 0 or np.ceil(pos[i, 2]) >= x_mom.shape[2] - 1):
                break

            # Interpolate momentum at current position
            x_momentum = _trilinear_interp(x_mom, pos[i, :])
            y_momentum = _trilinear_interp(y_mom, pos[i, :])
            z_momentum = _trilinear_interp(z_mom, pos[i, :])

            # Propagate position using momentum direction
            pos[i + 1, 0] = pos[i, 0] + x_momentum * step_size
            pos[i + 1, 1] = pos[i, 1] + y_momentum * step_size
            pos[i + 1, 2] = pos[i, 2] + z_momentum * step_size

        # Store path up to where it stopped (excluding the last invalid point)
        # MATLAB: paths{j} = Pos(1:i-1, :)
        # When loop breaks at step i, we want points 0 to i-1 in Python
        if i > 0:
            paths.append(pos[:i, :].copy())
        else:
            # If we broke on first iteration, return empty array
            paths.append(np.empty((0, 3)))

    return paths
