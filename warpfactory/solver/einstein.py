"""
Einstein tensor calculations

The Einstein tensor appears in Einstein's field equations and relates
the curvature of spacetime to the stress-energy tensor.
"""

import numpy as np
from typing import Dict
from ..core.tensor_ops import get_array_module


def calculate_einstein_tensor(
    R_munu: Dict[tuple, np.ndarray],
    R: np.ndarray,
    gl: Dict[tuple, np.ndarray]
) -> Dict[tuple, np.ndarray]:
    """
    Calculate the Einstein tensor G_μν from the Ricci tensor and scalar

    The Einstein tensor is given by:
    G_μν = R_μν - (1/2) g_μν R

    Args:
        R_munu: Ricci tensor R_μν (4x4 dictionary)
        R: Ricci scalar
        gl: Covariant metric tensor g_μν (4x4 dictionary)

    Returns:
        Dict: Einstein tensor G_μν (4x4 dictionary)
    """
    E = {}

    for mu in range(4):
        for nu in range(4):
            E[(mu, nu)] = R_munu[(mu, nu)] - 0.5 * gl[(mu, nu)] * R

    return E
