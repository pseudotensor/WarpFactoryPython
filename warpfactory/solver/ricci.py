"""
Ricci tensor and Ricci scalar calculations

The Ricci tensor describes the curvature of spacetime and is fundamental
to Einstein's field equations.
"""

import numpy as np
from typing import Dict, List
from ..core.tensor_ops import get_array_module
from ..units.constants import c as speed_of_light
from .finite_differences import take_finite_difference_1, take_finite_difference_2


def calculate_ricci_tensor(
    gu: Dict[tuple, np.ndarray],
    gl: Dict[tuple, np.ndarray],
    delta: List[float]
) -> Dict[tuple, np.ndarray]:
    """
    Calculate the Ricci tensor R_μν from the metric tensor

    The Ricci tensor is computed using the formula:
    R_μν = ∂_ρ Γ^ρ_νμ - ∂_ν Γ^ρ_ρμ + Γ^ρ_ρλ Γ^λ_νμ - Γ^ρ_νλ Γ^λ_ρμ

    Args:
        gu: Contravariant metric tensor g^μν (4x4 dictionary)
        gl: Covariant metric tensor g_μν (4x4 dictionary)
        delta: Grid spacing [dt, dx, dy, dz]

    Returns:
        Dict: Ricci tensor R_μν (4x4 dictionary)
    """
    xp = get_array_module(gu[(0, 0)])
    s = gu[(0, 0)].shape
    c = speed_of_light()

    # Precalculate metric derivatives for speed
    diff_1_gl = {}
    diff_2_gl = {}

    # First derivatives
    for i in range(4):
        for j in range(i, 4):
            phi_phi_flag = False
            for k in range(4):
                diff_1_gl[(i, j, k)] = take_finite_difference_1(gl[(i, j)], k, delta, phi_phi_flag)
                # Adjust for time coordinate
                if k == 0:
                    diff_1_gl[(i, j, k)] = diff_1_gl[(i, j, k)] / c

                # Second derivatives
                for n in range(k, 4):
                    diff_2_gl[(i, j, k, n)] = take_finite_difference_2(gl[(i, j)], k, n, delta, phi_phi_flag)

                    # Adjust for time coordinates
                    if (n == 0 and k != 0) or (n != 0 and k == 0):
                        diff_2_gl[(i, j, k, n)] = diff_2_gl[(i, j, k, n)] / c
                    elif n == 0 and k == 0:
                        diff_2_gl[(i, j, k, n)] = diff_2_gl[(i, j, k, n)] / (c**2)

                    # Symmetry
                    if k != n:
                        diff_2_gl[(i, j, n, k)] = diff_2_gl[(i, j, k, n)]

    # Assign symmetric values for first derivatives
    for k in range(4):
        diff_1_gl[(1, 0, k)] = diff_1_gl[(0, 1, k)]
        diff_1_gl[(2, 0, k)] = diff_1_gl[(0, 2, k)]
        diff_1_gl[(2, 1, k)] = diff_1_gl[(1, 2, k)]
        diff_1_gl[(3, 0, k)] = diff_1_gl[(0, 3, k)]
        diff_1_gl[(3, 1, k)] = diff_1_gl[(1, 3, k)]
        diff_1_gl[(3, 2, k)] = diff_1_gl[(2, 3, k)]

        for n in range(4):
            diff_2_gl[(1, 0, k, n)] = diff_2_gl[(0, 1, k, n)]
            diff_2_gl[(2, 0, k, n)] = diff_2_gl[(0, 2, k, n)]
            diff_2_gl[(2, 1, k, n)] = diff_2_gl[(1, 2, k, n)]
            diff_2_gl[(3, 0, k, n)] = diff_2_gl[(0, 3, k, n)]
            diff_2_gl[(3, 1, k, n)] = diff_2_gl[(1, 3, k, n)]
            diff_2_gl[(3, 2, k, n)] = diff_2_gl[(2, 3, k, n)]

    # Construct Ricci tensor
    R_munu = {}

    for i in range(4):
        for j in range(i, 4):
            # Initialize
            if xp == np:
                R_munu_temp = np.zeros(s)
            else:
                R_munu_temp = xp.zeros(s, dtype=gl[(0, 0)].dtype)

            for a in range(4):
                for b in range(4):
                    if xp == np:
                        R_munu_temp_2 = np.zeros(s)
                    else:
                        R_munu_temp_2 = xp.zeros(s, dtype=gl[(0, 0)].dtype)

                    # First term
                    R_munu_temp_2 = R_munu_temp_2 - (
                        diff_2_gl[(i, j, a, b)] + diff_2_gl[(a, b, i, j)] -
                        diff_2_gl[(i, b, j, a)] - diff_2_gl[(j, b, i, a)]
                    )

                    for r in range(4):
                        if xp == np:
                            R_munu_temp_3 = np.zeros(s)
                            R_munu_temp_4 = np.zeros(s)
                            R_munu_temp_5 = np.zeros(s)
                        else:
                            R_munu_temp_3 = xp.zeros(s, dtype=gl[(0, 0)].dtype)
                            R_munu_temp_4 = xp.zeros(s, dtype=gl[(0, 0)].dtype)
                            R_munu_temp_5 = xp.zeros(s, dtype=gl[(0, 0)].dtype)

                        for d in range(4):
                            # Second term
                            R_munu_temp_3 = R_munu_temp_3 + diff_1_gl[(b, d, j)] * gu[(r, d)]
                            R_munu_temp_4 = R_munu_temp_4 + (diff_1_gl[(j, d, b)] - diff_1_gl[(j, b, d)]) * gu[(r, d)]
                            # Third term
                            R_munu_temp_5 = R_munu_temp_5 - (diff_1_gl[(b, d, a)] + diff_1_gl[(b, d, a)] - diff_1_gl[(a, b, d)]) * gu[(r, d)]

                        R_munu_temp_2 = R_munu_temp_2 + (
                            R_munu_temp_4 * diff_1_gl[(i, r, a)] +
                            0.5 * (R_munu_temp_3 * diff_1_gl[(a, r, i)] +
                                   R_munu_temp_5 * (diff_1_gl[(j, r, i)] + diff_1_gl[(i, r, j)] - diff_1_gl[(j, i, r)]))
                        )

                    R_munu_temp = R_munu_temp + gu[(a, b)] * R_munu_temp_2

            R_munu[(i, j)] = 0.5 * R_munu_temp

    # Assign symmetric values
    R_munu[(1, 0)] = R_munu[(0, 1)]
    R_munu[(2, 0)] = R_munu[(0, 2)]
    R_munu[(2, 1)] = R_munu[(1, 2)]
    R_munu[(3, 0)] = R_munu[(0, 3)]
    R_munu[(3, 1)] = R_munu[(1, 3)]
    R_munu[(3, 2)] = R_munu[(2, 3)]

    return R_munu


def calculate_ricci_scalar(
    R_munu: Dict[tuple, np.ndarray],
    gu: Dict[tuple, np.ndarray]
) -> np.ndarray:
    """
    Calculate the Ricci scalar R from the Ricci tensor

    The Ricci scalar is the trace of the Ricci tensor:
    R = g^μν R_μν

    Args:
        R_munu: Ricci tensor R_μν (4x4 dictionary)
        gu: Contravariant metric tensor g^μν (4x4 dictionary)

    Returns:
        ndarray: Ricci scalar R at each spacetime point
    """
    xp = get_array_module(gu[(0, 0)])
    s = gu[(0, 0)].shape

    R = xp.zeros(s) if xp == np else xp.zeros(s, dtype=gu[(0, 0)].dtype)

    for mu in range(4):
        for nu in range(4):
            R = R + gu[(mu, nu)] * R_munu[(mu, nu)]

    return R
