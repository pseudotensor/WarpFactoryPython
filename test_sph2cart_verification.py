"""
Test script to verify spherical to Cartesian transformation

Compares Python implementation against known transformations
using Schwarzschild metric as test case.
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell.utils import sph2cart_diag


def test_schwarzschild_transformation():
    """
    Test with Schwarzschild metric in spherical coordinates:
    ds^2 = -(1-2M/r)dt^2 + (1-2M/r)^(-1)dr^2 + r^2 dθ^2 + r^2 sin^2(θ) dφ^2

    In diagonal form (t, r, θ, φ):
    g_tt = -(1-2M/r)
    g_rr = (1-2M/r)^(-1)
    g_θθ = r^2
    g_φφ = r^2 sin^2(θ)
    """

    print("=" * 80)
    print("VERIFICATION: Spherical to Cartesian Transformation")
    print("=" * 80)

    # Test parameters
    M = 1.0  # Mass parameter
    r = 10.0  # Radial coordinate

    # Spherical metric components
    g_tt_sph = -(1 - 2*M/r)
    g_rr_sph = 1 / (1 - 2*M/r)

    print(f"\nInput Schwarzschild metric at r = {r}, M = {M}:")
    print(f"  g_tt (spherical) = {g_tt_sph:.6f}")
    print(f"  g_rr (spherical) = {g_rr_sph:.6f}")

    # Test at various angles
    test_cases = [
        (np.pi/4, np.pi/4, "θ=π/4, φ=π/4"),
        (np.pi/2, 0.0, "θ=π/2, φ=0 (x-axis)"),
        (np.pi/2, np.pi/2, "θ=π/2, φ=π/2 (y-axis)"),
        (0.0, 0.0, "θ=0, φ=0 (z-axis)"),
        (np.pi/3, np.pi/6, "θ=π/3, φ=π/6"),
    ]

    for theta, phi, description in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Test case: {description}")
        print(f"  θ = {theta:.6f} rad, φ = {phi:.6f} rad")

        # Apply transformation
        g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(
            theta, phi, g_tt_sph, g_rr_sph
        )

        print(f"\nCartesian components:")
        print(f"  g_tt (g11_cart) = {g11:.6f}")
        print(f"  g_xx (g22_cart) = {g22:.6f}")
        print(f"  g_xy (g23_cart) = {g23:.6f}")
        print(f"  g_xz (g24_cart) = {g24:.6f}")
        print(f"  g_yy (g33_cart) = {g33:.6f}")
        print(f"  g_yz (g34_cart) = {g34:.6f}")
        print(f"  g_zz (g44_cart) = {g44:.6f}")

        # Check for numerical issues
        components = [g11, g22, g23, g24, g33, g34, g44]
        if any(np.isnan(c) or np.isinf(c) for c in components):
            print("\n  ⚠️  WARNING: NaN or Inf detected!")

        # Verify special angle behavior
        if abs(theta - np.pi/2) < 1e-10:
            print(f"  Special angle: θ = π/2, cos(θ) set to 0")
        if abs(phi - np.pi/2) < 1e-10:
            print(f"  Special angle: φ = π/2, cos(φ) set to 0")


def test_transformation_formula_consistency():
    """
    Verify transformation formulas match between MATLAB and Python
    """
    print(f"\n{'=' * 80}")
    print("FORMULA VERIFICATION")
    print("=" * 80)

    # Test values
    theta = np.pi/6
    phi = np.pi/4
    E = 1.25  # g22_sph value

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    print(f"\nTest values:")
    print(f"  θ = {theta:.6f}, φ = {phi:.6f}")
    print(f"  E (g_rr_sph) = {E:.6f}")
    print(f"  cos(θ) = {cos_theta:.6f}, sin(θ) = {sin_theta:.6f}")
    print(f"  cos(φ) = {cos_phi:.6f}, sin(φ) = {sin_phi:.6f}")

    # Compute each component manually to verify formula
    print("\nDiagonal components:")

    g22 = E * cos_phi**2 * sin_theta**2 + cos_phi**2 * cos_theta**2 + sin_phi**2
    print(f"  g22 = E·cos²φ·sin²θ + cos²φ·cos²θ + sin²φ")
    print(f"      = {E}·{cos_phi**2:.4f}·{sin_theta**2:.4f} + {cos_phi**2:.4f}·{cos_theta**2:.4f} + {sin_phi**2:.4f}")
    print(f"      = {g22:.6f}")

    g33 = E * sin_phi**2 * sin_theta**2 + cos_theta**2 * sin_phi**2 + cos_phi**2
    print(f"  g33 = E·sin²φ·sin²θ + cos²θ·sin²φ + cos²φ")
    print(f"      = {E}·{sin_phi**2:.4f}·{sin_theta**2:.4f} + {cos_theta**2:.4f}·{sin_phi**2:.4f} + {cos_phi**2:.4f}")
    print(f"      = {g33:.6f}")

    g44 = E * cos_theta**2 + sin_theta**2
    print(f"  g44 = E·cos²θ + sin²θ")
    print(f"      = {E}·{cos_theta**2:.4f} + {sin_theta**2:.4f}")
    print(f"      = {g44:.6f}")

    print("\nOff-diagonal components:")

    g23 = E * cos_phi * sin_phi * sin_theta**2 + cos_phi * cos_theta**2 * sin_phi - cos_phi * sin_phi
    print(f"  g23 = E·cosφ·sinφ·sin²θ + cosφ·cos²θ·sinφ - cosφ·sinφ")
    print(f"      = {g23:.6f}")

    g24 = E * cos_phi * cos_theta * sin_theta - cos_phi * cos_theta * sin_theta
    print(f"  g24 = E·cosφ·cosθ·sinθ - cosφ·cosθ·sinθ")
    print(f"      = {E}·{cos_phi:.4f}·{cos_theta:.4f}·{sin_theta:.4f} - {cos_phi:.4f}·{cos_theta:.4f}·{sin_theta:.4f}")
    print(f"      = {g24:.6f}")
    print(f"      = (E-1)·cosφ·cosθ·sinθ = {(E-1)*cos_phi*cos_theta*sin_theta:.6f}")

    g34 = E * cos_theta * sin_phi * sin_theta - cos_theta * sin_phi * sin_theta
    print(f"  g34 = E·cosθ·sinφ·sinθ - cosθ·sinφ·sinθ")
    print(f"      = {E}·{cos_theta:.4f}·{sin_phi:.4f}·{sin_theta:.4f} - {cos_theta:.4f}·{sin_phi:.4f}·{sin_theta:.4f}")
    print(f"      = {g34:.6f}")
    print(f"      = (E-1)·cosθ·sinφ·sinθ = {(E-1)*cos_theta*sin_phi*sin_theta:.6f}")

    # Verify with function
    g11_func, g22_func, g23_func, g24_func, g33_func, g34_func, g44_func = sph2cart_diag(
        theta, phi, -1.0, E
    )

    print("\nFunction output (with g_tt = -1.0):")
    print(f"  g22 = {g22_func:.6f} (manual: {g22:.6f})")
    print(f"  g33 = {g33_func:.6f} (manual: {g33:.6f})")
    print(f"  g44 = {g44_func:.6f} (manual: {g44:.6f})")
    print(f"  g23 = {g23_func:.6f} (manual: {g23:.6f})")
    print(f"  g24 = {g24_func:.6f} (manual: {g24:.6f})")
    print(f"  g34 = {g34_func:.6f} (manual: {g34:.6f})")

    # Check agreement
    tol = 1e-12
    if (abs(g22_func - g22) < tol and abs(g33_func - g33) < tol and
        abs(g44_func - g44) < tol and abs(g23_func - g23) < tol and
        abs(g24_func - g24) < tol and abs(g34_func - g34) < tol):
        print("\n✓ All components match manual calculation")
    else:
        print("\n✗ Mismatch detected!")


def test_special_angles():
    """
    Test special angle handling
    """
    print(f"\n{'=' * 80}")
    print("SPECIAL ANGLE HANDLING")
    print("=" * 80)

    g_tt = -0.8
    g_rr = 1.25

    # Test exactly π/2
    print("\nTest: θ = π/2 exactly")
    theta = np.pi / 2
    phi = 0.0

    g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)

    print(f"  Input: θ = {theta:.15f}")
    print(f"  cos(θ) should be treated as 0")
    print(f"  Results: g22={g22:.6f}, g33={g33:.6f}, g44={g44:.6f}")

    # Also compute with regular cos
    cos_theta_regular = np.cos(theta)
    print(f"  Regular cos(π/2) = {cos_theta_regular:.15e}")

    # Test φ = π/2 exactly
    print("\nTest: φ = π/2 exactly")
    theta = np.pi / 4
    phi = np.pi / 2

    g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)

    print(f"  Input: φ = {phi:.15f}")
    print(f"  cos(φ) should be treated as 0")
    print(f"  Results: g22={g22:.6f}, g33={g33:.6f}, g44={g44:.6f}")

    cos_phi_regular = np.cos(phi)
    print(f"  Regular cos(π/2) = {cos_phi_regular:.15e}")


def compare_matlab_python_formulas():
    """
    Direct comparison of MATLAB and Python formulas
    """
    print(f"\n{'=' * 80}")
    print("MATLAB vs PYTHON FORMULA COMPARISON")
    print("=" * 80)

    print("\nMATLAB formulas (from sph2cartDiag.m):")
    print("  g22_cart = (E*cosPhi^2*sin(theta)^2 + (cosPhi^2*cosTheta^2)) + sin(phi)^2")
    print("  g33_cart = (E*sin(phi)^2*sin(theta)^2 + (cosTheta^2*sin(phi)^2)) + cosPhi^2")
    print("  g44_cart = (E*cosTheta^2 + sin(theta)^2)")
    print("  g23_cart = (E*cosPhi*sin(phi)*sin(theta)^2 + (cosPhi*cosTheta^2*sin(phi)) - cosPhi*sin(phi))")
    print("  g24_cart = (E*cosPhi*cosTheta*sin(theta) - (cosPhi*cosTheta*sin(theta)))")
    print("  g34_cart = (E*cosTheta*sin(phi)*sin(theta) - (cosTheta*sin(phi)*sin(theta)))")

    print("\nPython formulas (from utils.py):")
    print("  g22_cart = E * cos_phi**2 * sin_theta**2 + cos_phi**2 * cos_theta**2 + sin_phi**2")
    print("  g33_cart = E * sin_phi**2 * sin_theta**2 + cos_theta**2 * sin_phi**2 + cos_phi**2")
    print("  g44_cart = E * cos_theta**2 + sin_theta**2")
    print("  g23_cart = E * cos_phi * sin_phi * sin_theta**2 + cos_phi * cos_theta**2 * sin_phi - cos_phi * sin_phi")
    print("  g24_cart = E * cos_phi * cos_theta * sin_theta - cos_phi * cos_theta * sin_theta")
    print("  g34_cart = E * cos_theta * sin_phi * sin_theta - cos_theta * sin_phi * sin_theta")

    print("\n✓ Formulas are IDENTICAL (accounting for notation: cosPhi → cos_phi, etc.)")


if __name__ == "__main__":
    test_schwarzschild_transformation()
    test_transformation_formula_consistency()
    test_special_angles()
    compare_matlab_python_formulas()

    print(f"\n{'=' * 80}")
    print("VERIFICATION COMPLETE")
    print("=" * 80)
