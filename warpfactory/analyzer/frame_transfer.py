"""
Frame transfer module for transforming energy tensors between reference frames

This module provides functionality to transform stress-energy tensors between
different reference frames using explicit Cholesky decomposition.
"""

import numpy as np
import warnings
from typing import Optional
from ..core.tensor import Tensor
from ..core.tensor_ops import verify_tensor, change_tensor_index, get_array_module


def tensor_cell_to_array(tensor: Tensor, use_gpu: bool = False) -> np.ndarray:
    """
    Convert a 4x4 tensor dictionary into a 6D numpy array

    Converts tensor components from dictionary format {(i,j): array_4d}
    to a single array with indexing (mu, nu, t, x, y, z)

    Args:
        tensor: Tensor object with 4x4 components
        use_gpu: Whether to use GPU arrays (CuPy)

    Returns:
        ndarray: 6D array with shape (4, 4, t, x, y, z)
    """
    xp = np
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            warnings.warn("CuPy not available, using NumPy")

    # Get shape from first component
    grid_shape = tensor.shape

    # Create output array
    array_tensor = xp.zeros((4, 4, *grid_shape), dtype=tensor[(0, 0)].dtype)

    # Fill array from tensor dictionary
    for i in range(4):
        for j in range(4):
            if use_gpu and xp != np:
                array_tensor[i, j] = xp.asarray(tensor[(i, j)])
            else:
                array_tensor[i, j] = tensor[(i, j)]

    return array_tensor


def get_eulerian_transformation_matrix(
    metric_array: np.ndarray,
    coords: str
) -> np.ndarray:
    """
    Compute the Eulerian transformation matrix using explicit Cholesky decomposition

    Returns the unique lower triangular matrix M with positive diagonal values
    that transforms the metric tensor to flat (Minkowski) metric:
    M^T . g . M = eta

    Args:
        metric_array: 6D array of metric components (4, 4, t, x, y, z)
        coords: Coordinate system (currently only 'cartesian' supported)

    Returns:
        ndarray: Transformation matrix M with shape (4, 4, t, x, y, z)
                or (4, 4) for simple metrics

    Notes:
        Uses explicit Cholesky decomposition formulas for 4x4 spacetime metrics.
        Warns if transformation is infinite or imaginary (numerical precision issues).
    """
    xp = get_array_module(metric_array)
    g = metric_array

    # Check if simple 4x4 metric or full spacetime array
    if g.ndim == 2:
        # Simple 4x4 matrix case
        if g.shape != (4, 4):
            raise ValueError("Simple metric must be 4x4")

        # Check symmetry
        if not xp.allclose(g, g.T):
            warnings.warn("Metric is not symmetric")

        M = xp.zeros((4, 4), dtype=g.dtype)

        # Explicit Cholesky decomposition factors
        factor0 = g[3, 3]
        factor1 = (-g[2, 3]**2 + g[2, 2] * factor0)
        factor2 = (2 * g[1, 2] * g[1, 3] * g[2, 3] - g[3, 3] * g[1, 2]**2 -
                   g[2, 2] * g[1, 3]**2 + g[1, 1] * factor1)
        factor3 = (-2 * g[3, 3] * g[0, 1] * g[0, 2] * g[1, 2] +
                   2 * g[0, 2] * g[0, 3] * g[1, 2] * g[1, 3] +
                   2 * g[0, 1] * g[0, 2] * g[1, 3] * g[2, 3] +
                   2 * g[0, 1] * g[0, 3] * g[1, 2] * g[2, 3] -
                   g[0, 1]**2 * g[2, 3]**2 - g[0, 2]**2 * g[1, 3]**2 - g[0, 3]**2 * g[1, 2]**2 +
                   g[2, 2] * (-2 * g[0, 1] * g[0, 3] * g[1, 3] + g[3, 3] * g[0, 1]**2) +
                   g[1, 1] * (-2 * g[0, 2] * g[0, 3] * g[2, 3] + g[3, 3] * g[0, 2]**2 +
                              g[2, 2] * g[0, 3]**2) -
                   g[0, 0] * factor2)

        # Compute transformation matrix elements
        M[0, 0] = xp.sqrt(factor2 / factor3)
        M[1, 0] = ((g[0, 1] * g[2, 3]**2 + g[0, 2] * g[1, 2] * g[3, 3] -
                    g[0, 2] * g[1, 3] * g[2, 3] - g[0, 3] * g[1, 2] * g[2, 3] +
                    g[0, 3] * g[1, 3] * g[2, 2] - g[0, 1] * g[2, 2] * g[3, 3]) /
                   xp.sqrt(factor2 * factor3))
        M[2, 0] = ((g[0, 2] * g[1, 3]**2 - g[0, 3] * g[1, 2] * g[1, 3] +
                    g[0, 1] * g[1, 2] * g[3, 3] - g[0, 1] * g[1, 3] * g[2, 3] -
                    g[0, 2] * g[1, 1] * g[3, 3] + g[0, 3] * g[1, 1] * g[2, 3]) /
                   xp.sqrt(factor2 * factor3))
        M[3, 0] = ((g[0, 3] * g[1, 2]**2 - g[0, 2] * g[1, 2] * g[1, 3] -
                    g[0, 1] * g[1, 2] * g[2, 3] + g[0, 1] * g[1, 3] * g[2, 2] +
                    g[0, 2] * g[1, 1] * g[2, 3] - g[0, 3] * g[1, 1] * g[2, 2]) /
                   xp.sqrt(factor2 * factor3))

        M[1, 1] = xp.sqrt(factor1 / factor2)
        M[2, 1] = ((g[1, 3] * g[2, 3] - g[1, 2] * g[3, 3]) /
                   xp.sqrt(factor1 * factor2))
        M[3, 1] = ((g[1, 2] * g[2, 3] - g[1, 3] * g[2, 2]) /
                   xp.sqrt(factor1 * factor2))

        M[2, 2] = xp.sqrt(factor0 / factor1)
        M[3, 2] = -g[2, 3] / xp.sqrt(factor0 * factor1)

        M[3, 3] = xp.sqrt(1.0 / factor0)

    elif g.ndim == 6:
        # Full spacetime array case (4, 4, t, x, y, z)
        if g.shape[:2] != (4, 4):
            raise ValueError("Metric array must have shape (4, 4, t, x, y, z)")

        grid_shape = g.shape[2:]
        M = xp.zeros_like(g)

        # Explicit Cholesky decomposition factors
        factor0 = g[3, 3]
        factor1 = (-g[2, 3]**2 + g[2, 2] * factor0)
        factor2 = (2 * g[1, 2] * g[1, 3] * g[2, 3] - g[3, 3] * g[1, 2]**2 -
                   g[2, 2] * g[1, 3]**2 + g[1, 1] * factor1)
        factor3 = (-2 * g[3, 3] * g[0, 1] * g[0, 2] * g[1, 2] +
                   2 * g[0, 2] * g[0, 3] * g[1, 2] * g[1, 3] +
                   2 * g[0, 1] * g[0, 2] * g[1, 3] * g[2, 3] +
                   2 * g[0, 1] * g[0, 3] * g[1, 2] * g[2, 3] -
                   g[0, 1]**2 * g[2, 3]**2 - g[0, 2]**2 * g[1, 3]**2 - g[0, 3]**2 * g[1, 2]**2 +
                   g[2, 2] * (-2 * g[0, 1] * g[0, 3] * g[1, 3] + g[3, 3] * g[0, 1]**2) +
                   g[1, 1] * (-2 * g[0, 2] * g[0, 3] * g[2, 3] + g[3, 3] * g[0, 2]**2 +
                              g[2, 2] * g[0, 3]**2) -
                   g[0, 0] * factor2)

        # Compute transformation matrix elements
        M[0, 0] = xp.sqrt(factor2 / factor3)
        M[1, 0] = ((g[0, 1] * g[2, 3]**2 + g[0, 2] * g[1, 2] * g[3, 3] -
                    g[0, 2] * g[1, 3] * g[2, 3] - g[0, 3] * g[1, 2] * g[2, 3] +
                    g[0, 3] * g[1, 3] * g[2, 2] - g[0, 1] * g[2, 2] * g[3, 3]) /
                   xp.sqrt(factor2 * factor3))
        M[2, 0] = ((g[0, 2] * g[1, 3]**2 - g[0, 3] * g[1, 2] * g[1, 3] +
                    g[0, 1] * g[1, 2] * g[3, 3] - g[0, 1] * g[1, 3] * g[2, 3] -
                    g[0, 2] * g[1, 1] * g[3, 3] + g[0, 3] * g[1, 1] * g[2, 3]) /
                   xp.sqrt(factor2 * factor3))
        M[3, 0] = ((g[0, 3] * g[1, 2]**2 - g[0, 2] * g[1, 2] * g[1, 3] -
                    g[0, 1] * g[1, 2] * g[2, 3] + g[0, 1] * g[1, 3] * g[2, 2] +
                    g[0, 2] * g[1, 1] * g[2, 3] - g[0, 3] * g[1, 1] * g[2, 2]) /
                   xp.sqrt(factor2 * factor3))

        M[1, 1] = xp.sqrt(factor1 / factor2)
        M[2, 1] = ((g[1, 3] * g[2, 3] - g[1, 2] * g[3, 3]) /
                   xp.sqrt(factor1 * factor2))
        M[3, 1] = ((g[1, 2] * g[2, 3] - g[1, 3] * g[2, 2]) /
                   xp.sqrt(factor1 * factor2))

        M[2, 2] = xp.sqrt(factor0 / factor1)
        M[3, 2] = -g[2, 3] / xp.sqrt(factor0 * factor1)

        M[3, 3] = xp.sqrt(1.0 / factor0)

    else:
        raise ValueError(f"Unrecognized metric array shape: {g.shape}")

    # Check for numerical issues
    if xp.any(xp.isinf(M)):
        warnings.warn("Eulerian Transformation is Infinite - Numerical Precision Insufficient")

    if not xp.all(xp.isreal(M)):
        warnings.warn("Eulerian Transformation is imaginary - Numerical Precision Insufficient")

    return M


def do_frame_transfer(
    metric: Tensor,
    energy_tensor: Tensor,
    frame: str,
    use_gpu: bool = False
) -> Tensor:
    """
    Transform the stress-energy tensor into selected reference frames

    Performs coordinate transformation to move from the coordinate frame to
    the specified observer frame. Currently only 'Eulerian' frame is supported.

    Args:
        metric: Metric tensor (must be verified)
        energy_tensor: Stress-energy tensor (must be verified)
        frame: Target frame ('Eulerian' is currently the only supported option)
        use_gpu: Whether to use GPU computation (requires CuPy)

    Returns:
        Tensor: Transformed stress-energy tensor with updated 'frame' metadata

    Raises:
        ValueError: If tensors are not verified or frame is not supported

    Notes:
        The Eulerian transformation uses explicit Cholesky decomposition to find
        the transformation matrix M such that M^T . g . M = eta (Minkowski metric).
        The stress-energy tensor is then transformed as:
        T_new = M^T . T . M

        The result is returned with contravariant indices (upper indices) where
        T^{0,i} = -T_{0,i} for spatial indices.
    """
    # Verify inputs
    if not verify_tensor(metric, suppress_msgs=True):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")

    if not verify_tensor(energy_tensor, suppress_msgs=True):
        raise ValueError("Stress-energy is not verified. Please verify stress-energy tensor using verify_tensor(energy_tensor).")

    # Initialize output
    transformed_energy_tensor = energy_tensor.copy()

    # Only process if frame is Eulerian and not already in Eulerian frame
    if frame.lower() == "eulerian":
        # Check if already in Eulerian frame
        if hasattr(energy_tensor, 'frame') and energy_tensor.frame.lower() == 'eulerian':
            warnings.warn("Energy tensor is already in Eulerian frame")
            return transformed_energy_tensor

        # Convert to covariant (lower) indices
        energy_tensor_covariant = change_tensor_index(energy_tensor, "covariant", metric)

        # Convert from cell dictionary to array format
        array_energy_tensor = tensor_cell_to_array(energy_tensor_covariant, use_gpu)
        array_metric_tensor = tensor_cell_to_array(metric, use_gpu)

        # Get appropriate array module
        xp = get_array_module(array_energy_tensor)

        # Get Eulerian transformation matrix at each point in space
        M = get_eulerian_transformation_matrix(array_metric_tensor, metric.coords)

        # Permute to put tensor indices last for matrix multiplication
        # From (mu, nu, t, x, y, z) to (t, x, y, z, mu, nu)
        M = xp.moveaxis(M, [0, 1], [-2, -1])
        array_energy_tensor = xp.moveaxis(array_energy_tensor, [0, 1], [-2, -1])

        # Perform transformation: T_transformed = M^T . T . M
        # Using einsum for efficient tensor contraction
        # T'_ij = M_ki * T_kl * M_lj
        transformed_array = xp.einsum('...ki,...kl,...lj->...ij',
                                      M, array_energy_tensor, M)

        # Move tensor indices back to front
        # From (t, x, y, z, mu, nu) to (mu, nu, t, x, y, z)
        transformed_array = xp.moveaxis(transformed_array, [-2, -1], [0, 1])

        # Convert array back to cell dictionary format
        for i in range(4):
            for j in range(4):
                if use_gpu:
                    # Keep on GPU if requested
                    transformed_energy_tensor.tensor[(i, j)] = transformed_array[i, j]
                else:
                    # Ensure on CPU
                    transformed_energy_tensor.tensor[(i, j)] = xp.asnumpy(transformed_array[i, j]) if hasattr(xp, 'asnumpy') else transformed_array[i, j]

        # Transform to contravariant: T^{0,i} = -T_{0,i} for spatial indices
        for i in range(1, 4):
            transformed_energy_tensor.tensor[(0, i)] = -transformed_energy_tensor.tensor[(0, i)]
            transformed_energy_tensor.tensor[(i, 0)] = -transformed_energy_tensor.tensor[(i, 0)]

        # Update tensor metadata
        transformed_energy_tensor.frame = "Eulerian"
        transformed_energy_tensor.index = "contravariant"

    else:
        warnings.warn(f"Frame '{frame}' not found. Only 'Eulerian' is currently supported.")

    return transformed_energy_tensor
