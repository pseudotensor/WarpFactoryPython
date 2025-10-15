"""
Covariant derivative calculations for vectors in curved spacetime

The covariant derivative extends the concept of directional derivatives to
curved spacetime by accounting for how the coordinate basis vectors change
from point to point using Christoffel symbols.
"""

import numpy as np
from typing import Dict, List
from ..core.tensor_ops import get_array_module
from .christoffel import get_christoffel_symbols
from .finite_differences import take_finite_difference_1


def cov_div(
    gl: Dict[tuple, np.ndarray],
    gu: Dict[tuple, np.ndarray],
    vec_u: Dict[int, np.ndarray],
    vec_d: Dict[int, np.ndarray],
    idx_div: int,
    idx_vec: int,
    delta: List[float],
    stair_sel: int
) -> np.ndarray:
    """
    Calculate the covariant derivative of a vector field

    The covariant derivative ∇_μ V^ν of a contravariant vector is:
        ∇_μ V^ν = ∂_μ V^ν + Γ^ν_μρ V^ρ

    The covariant derivative ∇_μ V_ν of a covariant vector is:
        ∇_μ V_ν = ∂_μ V_ν - Γ^ρ_νμ V_ρ

    Args:
        gl: Covariant metric tensor g_μν (4x4 dictionary with tuple keys)
        gu: Contravariant metric tensor g^μν (4x4 dictionary with tuple keys)
        vec_u: Contravariant vector V^μ (dictionary with int keys 0-3)
        vec_d: Covariant vector V_μ (dictionary with int keys 0-3)
        idx_div: Index of derivative direction (0=t, 1=x, 2=y, 3=z)
        idx_vec: Index of vector component to differentiate
        delta: Grid spacing [dt, dx, dy, dz]
        stair_sel: 0 for covariant vector (lower index), 1 for contravariant vector (upper index)

    Returns:
        ndarray: Covariant derivative ∇_{idx_div} V_{idx_vec} or ∇_{idx_div} V^{idx_vec}

    Raises:
        ValueError: If stair_sel is not 0 or 1
    """
    xp = get_array_module(gl[(0, 0)])
    s = gl[(0, 0)].shape

    # Precalculate metric derivatives
    diff_1_gl = {}

    for i in range(4):
        for j in range(4):
            # Check for phi-phi component in cylindrical/spherical coordinates
            phi_phi_flag = False
            if i == 1 and j == 1 and s[1] == 1:
                phi_phi_flag = True

            for k in range(4):
                diff_1_gl[(i, j, k)] = take_finite_difference_1(
                    gl[(i, j)], k, delta, phi_phi_flag
                )

    # Covariant derivative of covariant vector (lower index)
    if stair_sel == 0:
        # Build gradient operated vector: ∂_{idx_div} V_{idx_vec}
        cd_vec = take_finite_difference_1(vec_d[idx_vec], idx_div, delta, False)

        # Subtract connection term: - Γ^i_{idx_vec,idx_div} V_i
        for i in range(4):
            Gamma = get_christoffel_symbols(gu, diff_1_gl, i, idx_vec, idx_div)
            cd_vec = cd_vec - Gamma * vec_d[i]

    # Covariant derivative of contravariant vector (upper index)
    elif stair_sel == 1:
        # Build gradient operated vector: ∂_{idx_div} V^{idx_vec}
        cd_vec = take_finite_difference_1(vec_u[idx_vec], idx_div, delta, False)

        # Add connection term: + Γ^{idx_vec}_{idx_div,i} V^i
        for i in range(4):
            Gamma = get_christoffel_symbols(gu, diff_1_gl, idx_vec, idx_div, i)
            cd_vec = cd_vec + Gamma * vec_u[i]

    else:
        raise ValueError(f'Invalid variance selected: {stair_sel}. Must be 0 (covariant) or 1 (contravariant)')

    return cd_vec
