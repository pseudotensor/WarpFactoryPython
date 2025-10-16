"""
Utility functions for warp shell metric calculations

These functions support the comoving warp shell metric computation,
including TOV equations, sigmoid functions, and coordinate transformations.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from typing import Tuple
from ...units.constants import c as speed_of_light, G as gravitational_constant


def tov_const_density(R: float, M: np.ndarray, rho: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Tolman-Oppenheimer-Volkoff equation for constant density

    Computes the pressure profile for a constant density spherical shell
    using the TOV equation solution.

    Args:
        R: Outer radius of the shell
        M: Mass profile array as function of radius
        rho: Density profile array
        r: Radial coordinate array

    Returns:
        Pressure profile array
    """
    c = speed_of_light()
    G = gravitational_constant()

    M_end = M[-1]

    # TOV solution for constant density
    numerator = R * np.sqrt(R - 2*G*M_end/c**2) - np.sqrt(R**3 - 2*G*M_end*r**2/c**2)
    denominator = np.sqrt(R**3 - 2*G*M_end*r**2/c**2) - 3*R*np.sqrt(R - 2*G*M_end/c**2)

    P = c**2 * rho * (numerator / denominator) * (r < R)

    return P


def compact_sigmoid(r: np.ndarray, R1: float, R2: float, sigma: float, Rbuff: float) -> np.ndarray:
    """
    Compact sigmoid function for smooth transitions

    Creates a smooth transition function between R1 and R2 with buffer zones.

    Args:
        r: Radial coordinate array
        R1: Inner radius
        R2: Outer radius
        sigma: Sharpness parameter
        Rbuff: Buffer distance from walls

    Returns:
        Sigmoid function values

    Raises:
        ValueError: If function returns non-numeric values
    """
    # Calculate the sigmoid with buffer zones
    exponent = ((R2 - R1 - 2*Rbuff) * (sigma + 2) / 2 *
                (1.0 / (r - R2 + Rbuff) + 1.0 / (r - R1 - Rbuff)))

    f = np.abs(1.0 / (np.exp(exponent) + 1) *
               (r > R1 + Rbuff) * (r < R2 - Rbuff) +
               (r >= R2 - Rbuff) - 1)

    if np.any(np.isinf(f)) or np.any(~np.isreal(f)):
        raise ValueError('compact_sigmoid returns non-numeric values!')

    return f


def alpha_numeric_solver(M: np.ndarray, P: np.ndarray, R: float, r: np.ndarray) -> np.ndarray:
    """
    Numerical solver for the alpha metric function

    Solves for the alpha function using the TOV equation via trapezoidal integration.

    Args:
        M: Mass profile array
        P: Pressure profile array
        R: Maximum radius with non-zero density
        r: Radial coordinate array

    Returns:
        Alpha function values
    """
    c = speed_of_light()
    G = gravitational_constant()

    # Calculate derivative of alpha
    dalpha = (G*M/c**2 + 4*np.pi*G*r**3*P/c**4) / (r**2 - 2*G*M*r/c**2)
    dalpha[0] = 0

    # Integrate using trapezoidal method
    alpha_temp = cumulative_trapezoid(dalpha, r, initial=0)

    # Apply boundary condition at outer radius
    C = 0.5 * np.log(1 - 2*G*M[-1]/r[-1]/c**2)
    offset = C - alpha_temp[-1]
    alpha = alpha_temp + offset

    return alpha


def legendre_radial_interp(input_array: np.ndarray, r: float) -> float:
    """
    3rd order Legendre polynomial interpolation

    Performs cubic interpolation using Legendre polynomials for radial data.

    Args:
        input_array: Array of values to interpolate
        r: Position to interpolate at (can be non-integer)

    Returns:
        Interpolated value at position r
    """
    r_scale = 1

    # Get neighboring indices
    x0 = int(np.floor(r/r_scale - 1))
    x1 = int(np.floor(r/r_scale))
    x2 = int(np.ceil(r/r_scale))
    x3 = int(np.ceil(r/r_scale + 1))

    # Get values (with bounds checking) - FIX: Convert from 1-based MATLAB to 0-based Python
    y0 = input_array[max(x0 - 1, 0)]
    y1 = input_array[max(x1 - 1, 0)]
    y2 = input_array[min(max(x2 - 1, 0), len(input_array) - 1)]
    y3 = input_array[min(x3 - 1, len(input_array) - 1)]

    x = r

    # Scale indices back
    x0 = x0 * r_scale
    x1 = x1 * r_scale
    x2 = x2 * r_scale
    x3 = x3 * r_scale

    # Lagrange interpolation formula
    output_value = (y0 * (x - x1) * (x - x2) * (x - x3) / ((x0 - x1) * (x0 - x2) * (x0 - x3)) +
                   y1 * (x - x0) * (x - x2) * (x - x3) / ((x1 - x0) * (x1 - x2) * (x1 - x3)) +
                   y2 * (x - x0) * (x - x1) * (x - x3) / ((x2 - x0) * (x2 - x1) * (x2 - x3)) +
                   y3 * (x - x0) * (x - x1) * (x - x2) / ((x3 - x0) * (x3 - x1) * (x3 - x2)))

    return output_value


def sph2cart_diag(theta: float, phi: float, g11_sph: float, g22_sph: float) -> Tuple[float, float, float, float, float, float, float]:
    """
    Convert diagonal spherical metric components to Cartesian components

    Transforms a diagonal spherical metric (with only g_tt and g_rr non-zero)
    into Cartesian coordinates, producing both diagonal and off-diagonal terms.

    Args:
        theta: Polar angle (from z-axis)
        phi: Azimuthal angle (from x-axis)
        g11_sph: Spherical g_tt component
        g22_sph: Spherical g_rr component

    Returns:
        Tuple of (g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart)
        representing the Cartesian metric components
    """
    g11_cart = g11_sph

    E = g22_sph

    # Handle special angles to avoid numerical issues
    if abs(phi) == np.pi/2:
        cos_phi = 0.0
    else:
        cos_phi = np.cos(phi)

    if abs(theta) == np.pi/2:
        cos_theta = 0.0
    else:
        cos_theta = np.cos(theta)

    sin_theta = np.sin(theta)
    sin_phi = np.sin(phi)

    # Diagonal components
    g22_cart = E * cos_phi**2 * sin_theta**2 + cos_phi**2 * cos_theta**2 + sin_phi**2
    g33_cart = E * sin_phi**2 * sin_theta**2 + cos_theta**2 * sin_phi**2 + cos_phi**2
    g44_cart = E * cos_theta**2 + sin_theta**2

    # Off-diagonal components
    g23_cart = (E * cos_phi * sin_phi * sin_theta**2 +
                cos_phi * cos_theta**2 * sin_phi -
                cos_phi * sin_phi)
    g24_cart = (E * cos_phi * cos_theta * sin_theta -
                cos_phi * cos_theta * sin_theta)
    g34_cart = (E * cos_theta * sin_phi * sin_theta -
                cos_theta * sin_phi * sin_theta)

    return g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart


def smooth_array(arr: np.ndarray, smooth_factor: float, iterations: int = 4) -> np.ndarray:
    """
    Smooth array using moving average (matches MATLAB smooth())

    FIXED: Now uses moving average filter instead of Savitzky-Golay
    FIXED: Removed double 1.79 multiplication (caller already applies it)

    Args:
        arr: Input array to smooth
        smooth_factor: Smoothing window span (used directly)
        iterations: Number of smoothing passes (default: 4)

    Returns:
        Smoothed array
    """
    from scipy.ndimage import uniform_filter1d

    result = arr.copy()

    # FIX: Use smooth_factor directly (no 1.79 multiplication here)
    span = int(smooth_factor)
    if span < 1:
        return result
    if span % 2 == 0:
        span += 1  # Must be odd for symmetry

    # Ensure window isn't larger than array
    span = min(span, len(result))
    if span < 3:
        return result

    # FIX: Use moving average (uniform filter) to match MATLAB smooth()
    for _ in range(iterations):
        if len(result) >= span:
            result = uniform_filter1d(result, size=span, mode='nearest')

    return result
