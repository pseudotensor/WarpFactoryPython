"""
Christoffel symbols calculation

The Christoffel symbols (connection coefficients) describe how coordinates
change as you move through curved spacetime.
"""

import numpy as np
from typing import Dict
from ..core.tensor_ops import get_array_module


def get_christoffel_symbols(
    gu: Dict[tuple, np.ndarray],
    diff_1_gl: Dict[tuple, np.ndarray],
    i: int,
    k: int,
    l: int
) -> np.ndarray:
    """
    Calculate a single Christoffel symbol Γ^i_kl

    The Christoffel symbols are given by:
    Γ^i_kl = (1/2) g^im (∂_k g_ml + ∂_l g_mk - ∂_m g_kl)

    Args:
        gu: Contravariant metric tensor g^μν (4x4 dictionary)
        diff_1_gl: First derivatives of covariant metric ∂_k g_ij (4x4x4 dictionary)
        i: Upper index of Christoffel symbol
        k: First lower index
        l: Second lower index

    Returns:
        ndarray: Christoffel symbol Γ^i_kl at each spacetime point
    """
    xp = get_array_module(gu[(0, 0)])
    s = gu[(0, 0)].shape

    Gamma = xp.zeros(s) if xp == np else xp.zeros(s, dtype=gu[(0, 0)].dtype)

    for m in range(4):
        Gamma = Gamma + 0.5 * gu[(i, m)] * (
            diff_1_gl[(m, k, l)] +
            diff_1_gl[(m, l, k)] -
            diff_1_gl[(k, l, m)]
        )

    return Gamma
