"""
Evaluate metric properties at specific points

This module provides a high-level function to evaluate a metric tensor and compute
all core analysis products including:
- Stress-energy tensor (coordinate and Eulerian frames)
- Energy conditions (Null, Weak, Dominant, Strong)
- Kinematic scalars (expansion, shear, vorticity)
"""

import numpy as np
from typing import Dict, Any, Optional
from ..core.tensor import Tensor
from ..solver.energy import get_energy_tensor
from .frame_transfer import do_frame_transfer
from .energy_conditions import get_energy_conditions
from .scalars import get_scalars


def eval_metric(
    metric: Tensor,
    try_gpu: bool = False,
    keep_positive: bool = True,
    num_angular_vec: int = 100,
    num_time_vec: int = 10
) -> Dict[str, Any]:
    """
    Evaluate the metric and return the core analysis products

    This is a convenience function that computes the complete set of
    physical quantities from a spacetime metric, including energy conditions,
    stress-energy tensors, and kinematic scalars.

    Args:
        metric: Metric tensor struct object
        try_gpu: Flag on whether or not to use GPU computation (False=no, True=yes)
        keep_positive: Flag on whether or not to return positive values of the
                      energy conditions (False=no, True=yes). When False, positive
                      values are set to zero.
        num_angular_vec: Number of equally spaced spatial vectors to evaluate
        num_time_vec: Number of equally spaced temporal shells to evaluate

    Returns:
        Dict containing:
            - metric: The input metric tensor
            - energy_tensor: Stress-energy tensor in coordinate frame
            - energy_tensor_eulerian: Stress-energy tensor in Eulerian frame
            - null: Null energy condition values
            - weak: Weak energy condition values
            - strong: Strong energy condition values
            - dominant: Dominant energy condition values
            - expansion: Expansion scalar
            - shear: Shear scalar
            - vorticity: Vorticity scalar

    Notes:
        Energy condition values are typically negative where violated and positive
        where satisfied. When keep_positive=False, positive values are zeroed out
        to highlight only the violations.

    Example:
        >>> from warpfactory.metrics.alcubierre import get_alcubierre_metric
        >>> metric = get_alcubierre_metric([10, 10, 10, 10])
        >>> results = eval_metric(metric, num_angular_vec=50)
        >>> print(f"Min null energy: {results['null'].min()}")
    """
    # Initialize output dictionary
    output = {}

    # Store metric
    output['metric'] = metric

    # Compute energy tensor outputs
    output['energy_tensor'] = get_energy_tensor(metric, try_gpu)
    output['energy_tensor_eulerian'] = do_frame_transfer(
        metric,
        output['energy_tensor'],
        "Eulerian",
        try_gpu
    )

    # Compute energy condition outputs
    # Note: get_energy_conditions returns (map, vec, vector_field)
    # We only need the map (most violating evaluation at each point)
    output['null'], _, _ = get_energy_conditions(
        output['energy_tensor'],
        metric,
        "Null",
        num_angular_vec,
        num_time_vec,
        return_vec=False,
        try_gpu=try_gpu
    )

    output['weak'], _, _ = get_energy_conditions(
        output['energy_tensor'],
        metric,
        "Weak",
        num_angular_vec,
        num_time_vec,
        return_vec=False,
        try_gpu=try_gpu
    )

    output['strong'], _, _ = get_energy_conditions(
        output['energy_tensor'],
        metric,
        "Strong",
        num_angular_vec,
        num_time_vec,
        return_vec=False,
        try_gpu=try_gpu
    )

    output['dominant'], _, _ = get_energy_conditions(
        output['energy_tensor'],
        metric,
        "Dominant",
        num_angular_vec,
        num_time_vec,
        return_vec=False,
        try_gpu=try_gpu
    )

    # If not keeping positive values, zero them out
    if not keep_positive:
        output['null'][output['null'] > 0] = 0
        output['weak'][output['weak'] > 0] = 0
        output['strong'][output['strong'] > 0] = 0
        output['dominant'][output['dominant'] > 0] = 0

    # Compute scalar outputs
    expansion, shear, vorticity = get_scalars(metric)
    output['expansion'] = expansion
    output['shear'] = shear
    output['vorticity'] = vorticity

    return output
