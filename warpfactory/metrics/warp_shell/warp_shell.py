"""
Warp Shell Comoving metric implementation

Implements the comoving warp shell metric as described in:
https://iopscience.iop.org/article/10.1088/1361-6382/ad26aa

This metric describes a spherical shell of matter that can create a warp effect
for objects within the shell.
"""

import numpy as np
from typing import List, Optional
from scipy.integrate import cumulative_trapezoid
from ...core.tensor import Tensor
from ...units.constants import c as speed_of_light, G as gravitational_constant
from .utils import (
    tov_const_density,
    compact_sigmoid,
    alpha_numeric_solver,
    legendre_radial_interp,
    sph2cart_diag,
    smooth_array
)


def get_warp_shell_comoving_metric(
    grid_size: List[int],
    world_center: List[float],
    m: float,
    R1: float,
    R2: float,
    Rbuff: float = 0.0,
    sigma: float = 0.0,
    smooth_factor: float = 1.0,
    v_warp: float = 0.0,
    do_warp: bool = False,
    grid_scaling: Optional[List[float]] = None
) -> Tensor:
    """
    Build the Warp Shell metric in a comoving frame

    The warp shell metric describes a spherical shell of matter with specific
    density profile that can create a warp effect inside the shell.

    Args:
        grid_size: World size in [t, x, y, z]
        world_center: World center location in [t, x, y, z]
        m: Total mass of the warp shell
        R1: Inner radius of the shell
        R2: Outer radius of the shell
        Rbuff: Buffer distance between shell wall and where shift starts to change (default: 0)
        sigma: Sharpness parameter of the shift sigmoid (default: 0)
        smooth_factor: Factor by which to smooth the walls of the shell (default: 1)
        v_warp: Speed of the warp drive in factors of c, along x direction (default: 0)
        do_warp: Whether to create the warp effect inside the shell (default: False)
        grid_scaling: Scaling of the grid in [t, x, y, z] (default: [1,1,1,1])

    Returns:
        Tensor: Warp shell metric tensor in covariant form

    Notes:
        - The metric is built in spherical coordinates and then transformed to Cartesian
        - Uses TOV equations to compute pressure profile for the shell
        - The warp effect is applied via a shift vector if do_warp is True
    """
    if grid_scaling is None:
        grid_scaling = [1.0, 1.0, 1.0, 1.0]

    c = speed_of_light()
    G = gravitational_constant()

    # Calculate world size for radius sampling
    world_size = np.sqrt(
        (grid_size[1] * grid_scaling[1] - world_center[1])**2 +
        (grid_size[2] * grid_scaling[2] - world_center[2])**2 +
        (grid_size[3] * grid_scaling[3] - world_center[3])**2
    )

    # Create high-resolution radius array for profile computation
    r_sample_res = 100000
    rsample = np.linspace(0, world_size * 1.2, r_sample_res)

    # Construct density profile (constant density shell)
    shell_volume = 4.0/3.0 * np.pi * (R2**3 - R1**3)
    rho = np.zeros(len(rsample)) + m / shell_volume * ((rsample > R1) & (rsample < R2))

    # Find maximum radius with non-zero density
    density_diff = np.diff(rho > 0)
    if np.any(density_diff):
        max_r_idx = np.argmin(density_diff)
        max_r = rsample[max_r_idx]
    else:
        max_r = R2

    # Construct mass profile using integration
    M = cumulative_trapezoid(4 * np.pi * rho * rsample**2, rsample, initial=0)

    # Construct pressure profile using TOV equation
    P = tov_const_density(R2, M, rho, rsample)

    # Smooth the density and pressure profiles
    rho_smooth = smooth_array(rho, 1.79 * smooth_factor, iterations=4)
    P_smooth = smooth_array(P, smooth_factor, iterations=4)

    # Reconstruct mass profile with smoothed density
    M = cumulative_trapezoid(4 * np.pi * rho_smooth * rsample**2, rsample, initial=0)
    M[M < 0] = np.max(M)

    # Construct shift radial vector
    shift_radial_vector = compact_sigmoid(rsample, R1, R2, sigma, Rbuff)
    shift_radial_vector = smooth_array(shift_radial_vector, smooth_factor, iterations=2)

    # Solve for metric functions B and A
    # B component (radial)
    B = 1.0 / (1.0 - 2*G*M / (rsample * c**2))
    B[0] = 1.0

    # Solve for alpha function
    a = alpha_numeric_solver(M, P_smooth, max_r, rsample)

    # A component (temporal) from alpha
    A = -np.exp(2.0 * a)

    # Initialize metric tensor components
    metric_dict = {}
    for mu in range(4):
        for nu in range(4):
            metric_dict[(mu, nu)] = np.zeros(grid_size)

    # Initialize shift matrix
    shift_matrix = np.zeros(grid_size)

    # Small offset to handle r = 0
    epsilon = 1e-10

    # Loop through spatial grid and compute metric at each point
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                # Convert grid indices to coordinates
                # MATLAB uses 1-based indexing, Python uses 0-based
                x = (i + 1) * grid_scaling[1] - world_center[1]
                y = (j + 1) * grid_scaling[2] - world_center[2]
                z = (k + 1) * grid_scaling[3] - world_center[3]

                # Convert to spherical coordinates
                # ref: Catalog of Spacetimes, Eq. (1.6.2)
                r = np.sqrt(x**2 + y**2 + z**2) + epsilon
                theta = np.arctan2(np.sqrt(x**2 + y**2), z)
                phi = np.arctan2(y, x)

                # Find nearest sample point for interpolation
                min_idx = np.argmin(np.abs(rsample - r))
                if rsample[min_idx] > r and min_idx > 0:
                    min_idx = min_idx - 1

                # Fractional index for interpolation
                if min_idx < len(rsample) - 1:
                    frac_idx = min_idx + (r - rsample[min_idx]) / (rsample[min_idx + 1] - rsample[min_idx])
                else:
                    frac_idx = min_idx

                # Interpolate metric functions at this radius
                g11_sph = legendre_radial_interp(A, frac_idx)
                g22_sph = legendre_radial_interp(B, frac_idx)

                # Convert from spherical to Cartesian coordinates
                g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart = \
                    sph2cart_diag(theta, phi, g11_sph, g22_sph)

                # Assign to metric tensor (only time slice t=0)
                t = 0
                metric_dict[(0, 0)][t, i, j, k] = g11_cart
                metric_dict[(1, 1)][t, i, j, k] = g22_cart
                metric_dict[(1, 2)][t, i, j, k] = g23_cart
                metric_dict[(2, 1)][t, i, j, k] = g23_cart
                metric_dict[(1, 3)][t, i, j, k] = g24_cart
                metric_dict[(3, 1)][t, i, j, k] = g24_cart
                metric_dict[(2, 2)][t, i, j, k] = g33_cart
                metric_dict[(2, 3)][t, i, j, k] = g34_cart
                metric_dict[(3, 2)][t, i, j, k] = g34_cart
                metric_dict[(3, 3)][t, i, j, k] = g44_cart

                # Store shift function value
                shift_matrix[t, i, j, k] = legendre_radial_interp(shift_radial_vector, frac_idx)

    # Add warp effect if requested
    if do_warp:
        # Modify g_tx component (metric_dict[(0, 1)])
        metric_dict[(0, 1)] = metric_dict[(0, 1)] - metric_dict[(0, 1)] * shift_matrix - shift_matrix * v_warp
        metric_dict[(1, 0)] = metric_dict[(0, 1)]

    # Create tensor object
    metric = Tensor(
        tensor=metric_dict,
        tensor_type="metric",
        name="Comoving Warp Shell",
        index="covariant",
        coords="cartesian",
        scaling=grid_scaling,
        params={
            "gridSize": grid_size,
            "worldCenter": world_center,
            "m": m,
            "R1": R1,
            "R2": R2,
            "Rbuff": Rbuff,
            "sigma": sigma,
            "smoothFactor": smooth_factor,
            "vWarp": v_warp,
            "doWarp": do_warp,
            "rho": rho,
            "rhoSmooth": rho_smooth,
            "P": P,
            "PSmooth": P_smooth,
            "M": M,
            "rVec": rsample,
            "A": A,
            "B": B,
        }
    )

    return metric
