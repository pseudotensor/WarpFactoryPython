"""
Calculate kinematic scalars from spacetime metrics

This module computes the expansion, shear, and vorticity scalars from a metric tensor
using the 3+1 decomposition and covariant derivatives.

Scalars:
- Expansion scalar (θ): Trace of expansion tensor, measures volume change
- Shear scalar (σ²): Measures distortion without volume change
- Vorticity scalar (ω²): Measures rotation of spacetime
"""

import numpy as np
from typing import Tuple
from ..core.tensor import Tensor
from ..core.tensor_ops import c4_inv, change_tensor_index, get_array_module
from ..metrics.three_plus_one import three_plus_one_decomposer
from .utils import get_trace


def get_scalars(metric: Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate expansion, shear, and vorticity scalars from a metric tensor

    The scalars characterize the kinematic properties of the spacetime:
    - Expansion: Rate of volume change along worldlines
    - Shear: Distortion without volume change
    - Vorticity: Rotation of the congruence

    Args:
        metric: Metric tensor object (must be 4x4 tensor)

    Returns:
        Tuple of (expansion_scalar, shear_scalar, vorticity_scalar):
            - expansion_scalar: Expansion θ at each spacetime point
            - shear_scalar: Shear σ² at each spacetime point
            - vorticity_scalar: Vorticity ω² at each spacetime point
    """
    # Get array module (numpy or cupy)
    xp = get_array_module(metric[(0, 0)])
    s = metric.shape

    # Get 3+1 decomposition components
    alpha, beta_down, gamma_down, beta_up, gamma_up = three_plus_one_decomposer(metric)

    # Construct 4-velocity field
    # u^μ = (1/α)[1, -β^i]
    # u_μ = g_μν u^ν
    u_up = {}
    u_down = {}

    # Time component: u^0 = 1/α
    u_up[0] = 1.0 / alpha

    # Spatial components: u^i = -β^i/α
    for i in range(3):
        u_up[i+1] = -beta_up[i] / alpha

    # Lower the index using the metric to get u_μ
    # Ensure metric is covariant
    metric = change_tensor_index(metric, "covariant")

    for mu in range(4):
        u_down[mu] = xp.zeros(s)
        for nu in range(4):
            u_down[mu] = u_down[mu] + metric[(mu, nu)] * u_up[nu]

    # Calculate covariant derivative ∇_μ u_ν
    # NOTE: This is a simplified placeholder for covariant derivative
    # A full implementation would use Christoffel symbols and finite differences
    del_u = _compute_covariant_derivative(metric, u_up, u_down)

    # Calculate projection tensor P_μν = g_μν + u_μ u_ν
    # and mixed version P^μ_ν = δ^μ_ν + u^μ u_ν
    P = {}
    P_mix = {}

    for mu in range(4):
        for nu in range(4):
            # Kronecker delta
            k_delta = 1 if mu == nu else 0

            # P^μ_ν = δ^μ_ν + u^μ u_ν
            P_mix[(mu, nu)] = k_delta + u_up[mu] * u_down[nu]

            # P_μν = g_μν + u_μ u_ν
            P[(mu, nu)] = metric[(mu, nu)] + u_down[mu] * u_down[nu]

    # Build expansion tensor θ_μν = P^α_μ P^β_ν ∇_(α u_β)
    # θ_μν = (1/2)(∇_μ u_ν + ∇_ν u_μ) projected
    theta_tensor = {}
    for mu in range(4):
        for nu in range(4):
            theta_tensor[(mu, nu)] = xp.zeros(s)
            for alpha in range(4):
                for beta in range(4):
                    # Symmetric part: (∇_α u_β + ∇_β u_α)/2
                    symmetric_part = 0.5 * (del_u[(alpha, beta)] + del_u[(beta, alpha)])
                    theta_tensor[(mu, nu)] = (theta_tensor[(mu, nu)] +
                                             P_mix[(alpha, mu)] * P_mix[(beta, nu)] * symmetric_part)

    # Build vorticity tensor ω_μν = P^α_μ P^β_ν ∇_[α u_β]
    # ω_μν = (1/2)(∇_μ u_ν - ∇_ν u_μ) projected
    omega_tensor = {}
    for mu in range(4):
        for nu in range(4):
            omega_tensor[(mu, nu)] = xp.zeros(s)
            for alpha in range(4):
                for beta in range(4):
                    # Antisymmetric part: (∇_α u_β - ∇_β u_α)/2
                    antisymmetric_part = 0.5 * (del_u[(alpha, beta)] - del_u[(beta, alpha)])
                    omega_tensor[(mu, nu)] = (omega_tensor[(mu, nu)] +
                                             P_mix[(alpha, mu)] * P_mix[(beta, nu)] * antisymmetric_part)

    # Create tensor objects for trace calculation
    theta = Tensor(
        tensor=theta_tensor,
        tensor_type="tensor",
        index="covariant",
        coords=metric.coords
    )

    omega = Tensor(
        tensor=omega_tensor,
        tensor_type="tensor",
        index="covariant",
        coords=metric.coords
    )

    # Calculate expansion scalar: θ = Tr(θ_μν)
    expansion_scalar = get_trace(theta, metric)

    # Calculate vorticity scalar: ω² = (1/2) ω^μν ω_μν
    omega_up = change_tensor_index(omega, "contravariant", metric)
    vorticity_scalar = xp.zeros(s)
    for mu in range(4):
        for nu in range(4):
            vorticity_scalar = vorticity_scalar + 0.5 * omega_up[(mu, nu)] * omega[(mu, nu)]

    # Calculate shear tensor: σ_μν = θ_μν - (θ/3) P_μν
    shear_tensor = {}
    for mu in range(4):
        for nu in range(4):
            shear_tensor[(mu, nu)] = theta_tensor[(mu, nu)] - (expansion_scalar / 3.0) * P[(mu, nu)]

    # Create shear tensor object
    shear = Tensor(
        tensor=shear_tensor,
        tensor_type="tensor",
        index="covariant",
        coords=metric.coords
    )

    # Calculate shear scalar: σ² = (1/2) σ^μν σ_μν
    shear_up = change_tensor_index(shear, "contravariant", metric)
    shear_scalar = xp.zeros(s)
    for mu in range(4):
        for nu in range(4):
            shear_scalar = shear_scalar + 0.5 * shear_up[(mu, nu)] * shear[(mu, nu)]

    return expansion_scalar, shear_scalar, vorticity_scalar


def _compute_covariant_derivative(
    metric: Tensor,
    u_up: dict,
    u_down: dict
) -> dict:
    """
    Compute covariant derivative of 4-velocity vector

    Implements the full covariant derivative using Christoffel symbols:
    ∇_μ u_ν = ∂_μ u_ν - Γ^α_μν u_α

    Args:
        metric: Metric tensor
        u_up: Contravariant 4-velocity components
        u_down: Covariant 4-velocity components

    Returns:
        Dictionary with (mu, nu) keys containing ∇_μ u_ν
    """
    from ..solver.christoffel import get_christoffel_symbols
    from ..solver.finite_differences import take_finite_difference_1

    xp = get_array_module(metric[(0, 0)])
    s = metric.shape

    # Initialize covariant derivative tensor
    del_u = {}

    # Get grid spacing from metric (use scaling if available, otherwise default to 1)
    delta = metric.scaling if hasattr(metric, 'scaling') and metric.scaling is not None else [1.0, 1.0, 1.0, 1.0]

    # Compute metric derivatives
    # diff_1_gl[(i, j, k)] = ∂_k g_ij
    diff_1_gl = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                diff_1_gl[(i, j, k)] = take_finite_difference_1(
                    metric[(i, j)], k, delta, phi_phi_flag=False
                )

    # Get contravariant metric (inverse)
    metric_up = c4_inv(metric.tensor)

    # Compute covariant derivative for each component
    # ∇_μ u_ν = ∂_μ u_ν - Γ^α_μν u_α
    for mu in range(4):
        for nu in range(4):
            # Compute ordinary derivative ∂_μ u_ν
            del_u[(mu, nu)] = take_finite_difference_1(u_down[nu], mu, delta, phi_phi_flag=False)

            # Subtract connection term: Γ^α_μν u_α
            for alpha in range(4):
                Gamma = get_christoffel_symbols(metric_up, diff_1_gl, alpha, mu, nu)
                del_u[(mu, nu)] = del_u[(mu, nu)] - Gamma * u_down[alpha]

    return del_u
