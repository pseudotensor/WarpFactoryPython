"""
Finite difference calculations for spacetime derivatives

Implements 4th-order accurate finite difference schemes for computing
partial derivatives of metric tensors in spacetime.
"""

import numpy as np
from typing import List
from ..core.tensor_ops import get_array_module


def take_finite_difference_1(
    A: np.ndarray,
    k: int,
    delta: List[float],
    phi_phi_flag: bool = False
) -> np.ndarray:
    """
    Take the first partial derivative of a 4D array in coordinate direction k

    Uses 4th-order accurate finite difference stencil:
    f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)

    Args:
        A: 4D array to differentiate
        k: Coordinate direction (0=t, 1=x, 2=y, 3=z)
        delta: Grid spacing in each direction [dt, dx, dy, dz]
        phi_phi_flag: Special handling for phi-phi component (not typically used)

    Returns:
        4D array: Derivative of A in direction k
    """
    xp = get_array_module(A)

    if isinstance(delta, list):
        delta = xp.array(delta)

    s = xp.array(A.shape) if hasattr(xp, 'array') else A.shape
    B = xp.zeros(s) if xp == np else xp.zeros(s, dtype=A.dtype)

    # Only compute if we have enough grid points (need 5 for 4th order)
    if s[k] >= 5:
        if k == 0:  # Time direction
            B[2:-2, :, :, :] = (
                -(A[4:, :, :, :] - A[:-4, :, :, :]) +
                8 * (A[3:-1, :, :, :] - A[1:-3, :, :, :])
            ) / (12 * delta[k])
            # Boundary conditions: copy interior values
            B[0, :, :, :] = B[2, :, :, :]
            B[1, :, :, :] = B[2, :, :, :]
            B[-2, :, :, :] = B[-3, :, :, :]
            B[-1, :, :, :] = B[-3, :, :, :]

        elif k == 1:  # X direction
            B[:, 2:-2, :, :] = (
                -(A[:, 4:, :, :] - A[:, :-4, :, :]) +
                8 * (A[:, 3:-1, :, :] - A[:, 1:-3, :, :])
            ) / (12 * delta[k])
            B[:, 0, :, :] = B[:, 2, :, :]
            B[:, 1, :, :] = B[:, 2, :, :]
            B[:, -2, :, :] = B[:, -3, :, :]
            B[:, -1, :, :] = B[:, -3, :, :]

        elif k == 2:  # Y direction
            B[:, :, 2:-2, :] = (
                -(A[:, :, 4:, :] - A[:, :, :-4, :]) +
                8 * (A[:, :, 3:-1, :] - A[:, :, 1:-3, :])
            ) / (12 * delta[k])

            if phi_phi_flag:
                # Special handling for phi coordinate
                B[:, :, 0, :] = 2 * 4
                B[:, :, 1, :] = 2 * 3
                B[:, :, -2, :] = 2 * (s[2] - 5 - 1)
                B[:, :, -1, :] = 2 * (s[2] - 5)
            else:
                B[:, :, 0, :] = B[:, :, 2, :]
                B[:, :, 1, :] = B[:, :, 2, :]
                B[:, :, -2, :] = B[:, :, -3, :]
                B[:, :, -1, :] = B[:, :, -3, :]

        elif k == 3:  # Z direction
            B[:, :, :, 2:-2] = (
                -(A[:, :, :, 4:] - A[:, :, :, :-4]) +
                8 * (A[:, :, :, 3:-1] - A[:, :, :, 1:-3])
            ) / (12 * delta[k])
            B[:, :, :, 0] = B[:, :, :, 2]
            B[:, :, :, 1] = B[:, :, :, 2]
            B[:, :, :, -2] = B[:, :, :, -3]
            B[:, :, :, -1] = B[:, :, :, -3]

    return B


def take_finite_difference_2(
    A: np.ndarray,
    k1: int,
    k2: int,
    delta: List[float],
    phi_phi_flag: bool = False
) -> np.ndarray:
    """
    Take the second partial derivative of a 4D array

    For same direction (k1==k2), uses:
    f''(x) ≈ [-f(x+2h) - f(x-2h) + 16f(x+h) + 16f(x-h) - 30f(x)] / (12h²)

    For mixed directions (k1!=k2), applies first derivative twice

    Args:
        A: 4D array to differentiate
        k1: First coordinate direction
        k2: Second coordinate direction
        delta: Grid spacing in each direction
        phi_phi_flag: Special handling for phi coordinate

    Returns:
        4D array: Second derivative of A
    """
    xp = get_array_module(A)

    if isinstance(delta, list):
        delta = xp.array(delta)

    s = xp.array(A.shape) if hasattr(xp, 'array') else A.shape
    B = xp.zeros(s) if xp == np else xp.zeros(s, dtype=A.dtype)

    if s[k1] >= 5 and s[k2] >= 5:
        if k1 == k2:  # Second derivative in same direction
            if k1 == 0:  # Time
                B[2:-2, :, :, :] = (
                    -(A[4:, :, :, :] + A[:-4, :, :, :]) +
                    16 * (A[3:-1, :, :, :] + A[1:-3, :, :, :]) -
                    30 * A[2:-2, :, :, :]
                ) / (12 * delta[k1]**2)
                B[0, :, :, :] = B[2, :, :, :]
                B[1, :, :, :] = B[2, :, :, :]
                B[-2, :, :, :] = B[-3, :, :, :]
                B[-1, :, :, :] = B[-3, :, :, :]

            elif k1 == 1:  # X
                B[:, 2:-2, :, :] = (
                    -(A[:, 4:, :, :] + A[:, :-4, :, :]) +
                    16 * (A[:, 3:-1, :, :] + A[:, 1:-3, :, :]) -
                    30 * A[:, 2:-2, :, :]
                ) / (12 * delta[k1]**2)
                B[:, 0, :, :] = B[:, 2, :, :]
                B[:, 1, :, :] = B[:, 2, :, :]
                B[:, -2, :, :] = B[:, -3, :, :]
                B[:, -1, :, :] = B[:, -3, :, :]

            elif k1 == 2:  # Y
                B[:, :, 2:-2, :] = (
                    -(A[:, :, 4:, :] + A[:, :, :-4, :]) +
                    16 * (A[:, :, 3:-1, :] + A[:, :, 1:-3, :]) -
                    30 * A[:, :, 2:-2, :]
                ) / (12 * delta[k1]**2)

                if phi_phi_flag:
                    B[:, :, 0, :] = -2
                    B[:, :, 1, :] = -2
                    B[:, :, -2, :] = 2
                    B[:, :, -1, :] = 2
                else:
                    B[:, :, 0, :] = B[:, :, 2, :]
                    B[:, :, 1, :] = B[:, :, 2, :]
                    B[:, :, -2, :] = B[:, :, -3, :]
                    B[:, :, -1, :] = B[:, :, -3, :]

            elif k1 == 3:  # Z
                B[:, :, :, 2:-2] = (
                    -(A[:, :, :, 4:] + A[:, :, :, :-4]) +
                    16 * (A[:, :, :, 3:-1] + A[:, :, :, 1:-3]) -
                    30 * A[:, :, :, 2:-2]
                ) / (12 * delta[k1]**2)
                B[:, :, :, 0] = B[:, :, :, 2]
                B[:, :, :, 1] = B[:, :, :, 2]
                B[:, :, :, -2] = B[:, :, :, -3]
                B[:, :, :, -1] = B[:, :, :, -3]

        else:  # Mixed partial derivative
            kL = max(k1, k2)
            kS = min(k1, k2)

            # Index ranges for 4th order stencil
            x2 = slice(4, s[kS])
            x1 = slice(3, s[kS] - 1)
            x0 = slice(2, s[kS] - 2)
            x_1 = slice(1, s[kS] - 3)
            x_2 = slice(0, s[kS] - 4)

            y2 = slice(4, s[kL])
            y1 = slice(3, s[kL] - 1)
            y0 = slice(2, s[kL] - 2)
            y_1 = slice(1, s[kL] - 3)
            y_2 = slice(0, s[kL] - 4)

            # Compute mixed derivative using finite difference formula
            if kS == 0:  # Time with...
                if kL == 1:  # X
                    B[x0, y0, :, :] = (
                        -(-(A[x2, y2, :, :] - A[x_2, y2, :, :]) + 8 * (A[x1, y2, :, :] - A[x_1, y2, :, :])) +
                        (-(A[x2, y_2, :, :] - A[x_2, y_2, :, :]) + 8 * (A[x1, y_2, :, :] - A[x_1, y_2, :, :])) +
                        8 * (-(A[x2, y1, :, :] - A[x_2, y1, :, :]) + 8 * (A[x1, y1, :, :] - A[x_1, y1, :, :])) -
                        8 * (-(A[x2, y_1, :, :] - A[x_2, y_1, :, :]) + 8 * (A[x1, y_1, :, :] - A[x_1, y_1, :, :]))
                    ) / (144 * delta[kL] * delta[kS])
                elif kL == 2:  # Y
                    B[x0, :, y0, :] = (
                        -(-(A[x2, :, y2, :] - A[x_2, :, y2, :]) + 8 * (A[x1, :, y2, :] - A[x_1, :, y2, :])) +
                        (-(A[x2, :, y_2, :] - A[x_2, :, y_2, :]) + 8 * (A[x1, :, y_2, :] - A[x_1, :, y_2, :])) +
                        8 * (-(A[x2, :, y1, :] - A[x_2, :, y1, :]) + 8 * (A[x1, :, y1, :] - A[x_1, :, y1, :])) -
                        8 * (-(A[x2, :, y_1, :] - A[x_2, :, y_1, :]) + 8 * (A[x1, :, y_1, :] - A[x_1, :, y_1, :]))
                    ) / (144 * delta[kL] * delta[kS])
                elif kL == 3:  # Z
                    B[x0, :, :, y0] = (
                        -(-(A[x2, :, :, y2] - A[x_2, :, :, y2]) + 8 * (A[x1, :, :, y2] - A[x_1, :, :, y2])) +
                        (-(A[x2, :, :, y_2] - A[x_2, :, :, y_2]) + 8 * (A[x1, :, :, y_2] - A[x_1, :, :, y_2])) +
                        8 * (-(A[x2, :, :, y1] - A[x_2, :, :, y1]) + 8 * (A[x1, :, :, y1] - A[x_1, :, :, y1])) -
                        8 * (-(A[x2, :, :, y_1] - A[x_2, :, :, y_1]) + 8 * (A[x1, :, :, y_1] - A[x_1, :, :, y_1]))
                    ) / (144 * delta[kL] * delta[kS])

            elif kS == 1:  # X with...
                if kL == 2:  # Y
                    B[:, x0, y0, :] = (
                        -(-(A[:, x2, y2, :] - A[:, x_2, y2, :]) + 8 * (A[:, x1, y2, :] - A[:, x_1, y2, :])) +
                        (-(A[:, x2, y_2, :] - A[:, x_2, y_2, :]) + 8 * (A[:, x1, y_2, :] - A[:, x_1, y_2, :])) +
                        8 * (-(A[:, x2, y1, :] - A[:, x_2, y1, :]) + 8 * (A[:, x1, y1, :] - A[:, x_1, y1, :])) -
                        8 * (-(A[:, x2, y_1, :] - A[:, x_2, y_1, :]) + 8 * (A[:, x1, y_1, :] - A[:, x_1, y_1, :]))
                    ) / (144 * delta[kL] * delta[kS])
                elif kL == 3:  # Z
                    B[:, x0, :, y0] = (
                        -(-(A[:, x2, :, y2] - A[:, x_2, :, y2]) + 8 * (A[:, x1, :, y2] - A[:, x_1, :, y2])) +
                        (-(A[:, x2, :, y_2] - A[:, x_2, :, y_2]) + 8 * (A[:, x1, :, y_2] - A[:, x_1, :, y_2])) +
                        8 * (-(A[:, x2, :, y1] - A[:, x_2, :, y1]) + 8 * (A[:, x1, :, y1] - A[:, x_1, :, y1])) -
                        8 * (-(A[:, x2, :, y_1] - A[:, x_2, :, y_1]) + 8 * (A[:, x1, :, y_1] - A[:, x_1, :, y_1]))
                    ) / (144 * delta[kL] * delta[kS])

            elif kS == 2 and kL == 3:  # Y with Z
                B[:, :, x0, y0] = (
                    -(-(A[:, :, x2, y2] - A[:, :, x_2, y2]) + 8 * (A[:, :, x1, y2] - A[:, :, x_1, y2])) +
                    (-(A[:, :, x2, y_2] - A[:, :, x_2, y_2]) + 8 * (A[:, :, x1, y_2] - A[:, :, x_1, y_2])) +
                    8 * (-(A[:, :, x2, y1] - A[:, :, x_2, y1]) + 8 * (A[:, :, x1, y1] - A[:, :, x_1, y1])) -
                    8 * (-(A[:, :, x2, y_1] - A[:, :, x_2, y_1]) + 8 * (A[:, :, x1, y_1] - A[:, :, x_1, y_1]))
                ) / (144 * delta[kL] * delta[kS])

    return B
