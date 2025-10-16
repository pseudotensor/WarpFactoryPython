"""
Verify the theoretical correctness of the spherical to Cartesian transformation

This tests that the transformation correctly converts a diagonal spherical metric
(with only g_tt, g_rr, g_θθ, g_φφ non-zero) to Cartesian coordinates.

The coordinate transformation is:
  x = r sin(θ) cos(φ)
  y = r sin(θ) sin(φ)
  z = r cos(θ)

The metric transforms as:
  g_μν = ∂x^α/∂x'^μ * ∂x^β/∂x'^ν * g'_αβ
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell.utils import sph2cart_diag


def compute_jacobian(r, theta, phi):
    """
    Compute Jacobian matrix for spherical to Cartesian transformation

    Coordinates: (t, x, y, z) in Cartesian, (t, r, θ, φ) in spherical

    Returns: 4x4 Jacobian matrix ∂(t,x,y,z)/∂(t,r,θ,φ)
    """
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Special angle handling
    if abs(theta) == np.pi/2:
        cos_theta = 0.0
    if abs(phi) == np.pi/2:
        cos_phi = 0.0

    # Jacobian matrix
    J = np.array([
        [1, 0, 0, 0],  # t doesn't change
        [0, sin_theta*cos_phi, r*cos_theta*cos_phi, -r*sin_theta*sin_phi],  # ∂x/∂(t,r,θ,φ)
        [0, sin_theta*sin_phi, r*cos_theta*sin_phi,  r*sin_theta*cos_phi],  # ∂y/∂(t,r,θ,φ)
        [0, cos_theta,        -r*sin_theta,          0]                      # ∂z/∂(t,r,θ,φ)
    ])

    return J


def transform_metric_theory(g_sph, J):
    """
    Transform metric using tensor transformation law:
    g_cart = J^T * g_sph * J

    Args:
        g_sph: 4x4 spherical metric matrix
        J: 4x4 Jacobian matrix

    Returns:
        g_cart: 4x4 Cartesian metric matrix
    """
    return J.T @ g_sph @ J


def test_transformation_theory():
    """
    Verify the sph2cart_diag function against theoretical transformation
    """
    print("=" * 80)
    print("THEORETICAL VERIFICATION: Metric Transformation")
    print("=" * 80)

    # Test parameters
    r = 10.0
    M = 1.0

    # Schwarzschild metric components
    g_tt = -(1 - 2*M/r)
    g_rr = 1 / (1 - 2*M/r)

    print(f"\nSchwarschild metric at r = {r}, M = {M}:")
    print(f"  g_tt = {g_tt:.6f}")
    print(f"  g_rr = {g_rr:.6f}")
    print(f"  g_θθ = r² = {r**2:.6f}")
    print(f"  g_φφ = r²sin²θ")

    # Test cases
    test_cases = [
        (np.pi/4, np.pi/4, "θ=π/4, φ=π/4"),
        (np.pi/2, 0.0, "θ=π/2, φ=0"),
        (np.pi/2, np.pi/2, "θ=π/2, φ=π/2"),
        (np.pi/3, np.pi/6, "θ=π/3, φ=π/6"),
    ]

    for theta, phi, description in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Test: {description}")

        # Spherical metric (diagonal)
        g_sph = np.diag([g_tt, g_rr, r**2, r**2 * np.sin(theta)**2])

        # Compute Jacobian
        J = compute_jacobian(r, theta, phi)

        # Transform using theory
        g_cart_theory = transform_metric_theory(g_sph, J)

        # Transform using function (which assumes diagonal spherical metric)
        g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)

        # Note: The function assumes g_θθ = 1 and g_φφ = 1 in the metric,
        # and returns components that need to be multiplied by r²
        # Actually, looking at the code, E = g_rr_sph represents the radial metric component

        print(f"\nTheoretical transformation (full tensor law):")
        print(f"  g_tt = {g_cart_theory[0,0]:.6f}")
        print(f"  g_xx = {g_cart_theory[1,1]:.6f}")
        print(f"  g_xy = {g_cart_theory[1,2]:.6f}")
        print(f"  g_xz = {g_cart_theory[1,3]:.6f}")
        print(f"  g_yy = {g_cart_theory[2,2]:.6f}")
        print(f"  g_yz = {g_cart_theory[2,3]:.6f}")
        print(f"  g_zz = {g_cart_theory[3,3]:.6f}")

        print(f"\nFunction output (sph2cart_diag):")
        print(f"  g_tt = {g11:.6f}")
        print(f"  g_xx = {g22:.6f}")
        print(f"  g_xy = {g23:.6f}")
        print(f"  g_xz = {g24:.6f}")
        print(f"  g_yy = {g33:.6f}")
        print(f"  g_yz = {g34:.6f}")
        print(f"  g_zz = {g44:.6f}")

        # Check agreement
        tol = 1e-10
        match_tt = abs(g_cart_theory[0,0] - g11) < tol
        match_xx = abs(g_cart_theory[1,1] - g22) < tol
        match_xy = abs(g_cart_theory[1,2] - g23) < tol
        match_xz = abs(g_cart_theory[1,3] - g24) < tol
        match_yy = abs(g_cart_theory[2,2] - g33) < tol
        match_yz = abs(g_cart_theory[2,3] - g34) < tol
        match_zz = abs(g_cart_theory[3,3] - g44) < tol

        if all([match_tt, match_xx, match_xy, match_xz, match_yy, match_yz, match_zz]):
            print("\n✓ MATCH: Function agrees with theoretical transformation")
        else:
            print("\n✗ MISMATCH: Differences detected:")
            if not match_tt: print(f"  g_tt: {abs(g_cart_theory[0,0] - g11):.2e}")
            if not match_xx: print(f"  g_xx: {abs(g_cart_theory[1,1] - g22):.2e}")
            if not match_xy: print(f"  g_xy: {abs(g_cart_theory[1,2] - g23):.2e}")
            if not match_xz: print(f"  g_xz: {abs(g_cart_theory[1,3] - g24):.2e}")
            if not match_yy: print(f"  g_yy: {abs(g_cart_theory[2,2] - g33):.2e}")
            if not match_yz: print(f"  g_yz: {abs(g_cart_theory[2,3] - g34):.2e}")
            if not match_zz: print(f"  g_zz: {abs(g_cart_theory[3,3] - g44):.2e}")


def test_simplified_metric():
    """
    Test with simplified metric where g_θθ = g_φφ = E (constant)
    This is what the function actually assumes
    """
    print(f"\n{'=' * 80}")
    print("SIMPLIFIED METRIC TEST")
    print("=" * 80)
    print("\nThe function assumes: ds² = g_tt dt² + g_rr (dr² + r²dΩ²)")
    print("where dΩ² = dθ² + sin²θ dφ² with coefficient absorbed into g_rr")

    # Test with unit r to see pure angular effects
    r = 1.0
    g_tt = -1.0
    E = 1.5  # g_rr

    theta = np.pi/4
    phi = np.pi/4

    print(f"\nTest with r = {r}, E = {E}, θ = {theta:.4f}, φ = {phi:.4f}")

    # Build spherical metric with g_θθ = E*r², g_φφ = E*r²sin²θ
    g_sph = np.diag([g_tt, E, E * r**2, E * r**2 * np.sin(theta)**2])

    # Compute Jacobian
    J = compute_jacobian(r, theta, phi)

    # Transform
    g_cart_theory = transform_metric_theory(g_sph, J)

    # Function output
    g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, E)

    print(f"\nTheoretical:")
    print(f"  g_xx = {g_cart_theory[1,1]:.6f}")
    print(f"  g_xy = {g_cart_theory[1,2]:.6f}")
    print(f"  g_xz = {g_cart_theory[1,3]:.6f}")
    print(f"  g_yy = {g_cart_theory[2,2]:.6f}")
    print(f"  g_yz = {g_cart_theory[2,3]:.6f}")
    print(f"  g_zz = {g_cart_theory[3,3]:.6f}")

    print(f"\nFunction:")
    print(f"  g_xx = {g22:.6f}")
    print(f"  g_xy = {g23:.6f}")
    print(f"  g_xz = {g24:.6f}")
    print(f"  g_yy = {g33:.6f}")
    print(f"  g_yz = {g34:.6f}")
    print(f"  g_zz = {g44:.6f}")

    tol = 1e-10
    if (abs(g_cart_theory[1,1] - g22) < tol and
        abs(g_cart_theory[1,2] - g23) < tol and
        abs(g_cart_theory[1,3] - g24) < tol and
        abs(g_cart_theory[2,2] - g33) < tol and
        abs(g_cart_theory[2,3] - g34) < tol and
        abs(g_cart_theory[3,3] - g44) < tol):
        print("\n✓ MATCH: Function output agrees with theory")
    else:
        print("\n✗ MISMATCH detected")


if __name__ == "__main__":
    test_transformation_theory()
    test_simplified_metric()

    print(f"\n{'=' * 80}")
    print("VERIFICATION COMPLETE")
    print("=" * 80)
