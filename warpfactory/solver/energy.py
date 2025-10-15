"""
Stress-energy tensor calculations

The stress-energy tensor describes the density and flux of energy and
momentum in spacetime. It is computed from the metric via Einstein's
field equations.
"""

import numpy as np
from typing import Dict, List, Optional
from ..core.tensor import Tensor
from ..core.tensor_ops import c4_inv, get_array_module, change_tensor_index, verify_tensor
from ..units.constants import c as speed_of_light, G as gravitational_constant
from .ricci import calculate_ricci_tensor, calculate_ricci_scalar
from .einstein import calculate_einstein_tensor


def metric_to_energy_density(
    gl: Dict[tuple, np.ndarray],
    gu: Dict[tuple, np.ndarray],
    delta: List[float]
) -> Dict[tuple, np.ndarray]:
    """
    Convert metric tensor to stress-energy tensor

    Uses Einstein's field equations:
    T^μν = (c^4 / 8πG) G^μν

    Where G^μν is the Einstein tensor

    Args:
        gl: Covariant metric tensor g_μν (4x4 dictionary)
        gu: Contravariant metric tensor g^μν (4x4 dictionary)
        delta: Grid spacing [dt, dx, dy, dz]

    Returns:
        Dict: Contravariant stress-energy tensor T^μν (4x4 dictionary)
    """
    xp = get_array_module(gl[(0, 0)])
    c = speed_of_light()
    G = gravitational_constant()

    # Calculate the Ricci tensor
    R_munu = calculate_ricci_tensor(gu, gl, delta)

    # Calculate the Ricci scalar
    R = calculate_ricci_scalar(R_munu, gu)

    # Calculate Einstein tensor (covariant form)
    E = calculate_einstein_tensor(R_munu, R, gl)

    # Calculate energy density from Einstein tensor
    # First, convert to contravariant form: T^μν = (c^4 / 8πG) E^μν
    # where E^μν = g^μα g^νβ E_αβ

    energy_density_cov = {}
    for mu in range(4):
        for nu in range(4):
            energy_density_cov[(mu, nu)] = (c**4 / (8 * np.pi * G)) * E[(mu, nu)]

    # Convert to contravariant form
    energy_density = {}
    s = gl[(0, 0)].shape

    for mu in range(4):
        for nu in range(4):
            if xp == np:
                energy_density[(mu, nu)] = np.zeros(s)
            else:
                energy_density[(mu, nu)] = xp.zeros(s, dtype=gl[(0, 0)].dtype)

            for alpha in range(4):
                for beta in range(4):
                    energy_density[(mu, nu)] += (
                        energy_density_cov[(alpha, beta)] *
                        gu[(alpha, mu)] *
                        gu[(beta, nu)]
                    )

    return energy_density


def get_energy_tensor(
    metric: Tensor,
    try_gpu: bool = False,
    diff_order: str = 'fourth'
) -> Tensor:
    """
    Convert a metric tensor to the stress-energy tensor

    Args:
        metric: Metric tensor object
        try_gpu: Whether to use GPU computation (requires CuPy)
        diff_order: Order of finite differences ('fourth' or 'second')

    Returns:
        Tensor: Stress-energy tensor object (contravariant)
    """
    # Verify metric
    if not verify_tensor(metric, suppress_msgs=True):
        raise ValueError("Metric is not verified. Please verify metric using verify_tensor(metric).")

    # Ensure metric is covariant
    if metric.index.lower() != "covariant":
        metric = change_tensor_index(metric, "covariant")
        print(f"Changed metric from {metric.index} index to covariant index")

    # Use GPU if requested
    if try_gpu:
        try:
            import cupy as cp
            metric_gpu = metric.to_gpu()
            gl_gpu = metric_gpu.tensor
            gu_gpu = c4_inv(gl_gpu)

            # Compute on GPU
            energy_dict = metric_to_energy_density(gl_gpu, gu_gpu, metric.scaling)

            # Gather results from GPU
            energy_tensor = {}
            for i in range(4):
                for j in range(4):
                    energy_tensor[(i, j)] = cp.asnumpy(energy_dict[(i, j)])

        except ImportError:
            print("CuPy not installed, falling back to CPU computation")
            try_gpu = False

    if not try_gpu:
        # Compute on CPU
        gl = metric.tensor
        gu = c4_inv(gl)
        energy_tensor = metric_to_energy_density(gl, gu, metric.scaling)

    # Create energy tensor object
    energy = Tensor(
        tensor=energy_tensor,
        tensor_type="stress-energy",
        name=metric.name,
        index="contravariant",
        coords=metric.coords,
        scaling=metric.scaling,
        params={"order": diff_order, "source_metric": metric.name}
    )

    return energy
