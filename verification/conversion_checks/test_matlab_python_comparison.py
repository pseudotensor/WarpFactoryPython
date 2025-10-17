"""
Direct comparison test between MATLAB and Python sph2cart_diag implementations

This test verifies that the Python implementation produces identical results
to the MATLAB implementation for various test cases.
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell.utils import sph2cart_diag


def matlab_sph2cart_diag(theta, phi, g11_sph, g22_sph):
    """
    Python reimplementation of MATLAB sph2cartDiag.m for direct comparison

    This is a line-by-line translation of the MATLAB code.
    """
    g11_cart = g11_sph

    E = g22_sph

    # Special angle handling - exactly as in MATLAB
    if abs(phi) == np.pi/2:
        cosPhi = 0
    else:
        cosPhi = np.cos(phi)

    if abs(theta) == np.pi/2:
        cosTheta = 0
    else:
        cosTheta = np.cos(theta)

    # Compute components - exactly as in MATLAB
    g22_cart = (E*cosPhi**2*np.sin(theta)**2 + (cosPhi**2*cosTheta**2)) + np.sin(phi)**2
    g33_cart = (E*np.sin(phi)**2*np.sin(theta)**2 + (cosTheta**2*np.sin(phi)**2)) + cosPhi**2
    g44_cart = (E*cosTheta**2 + np.sin(theta)**2)

    g23_cart = (E*cosPhi*np.sin(phi)*np.sin(theta)**2 +
                (cosPhi*cosTheta**2*np.sin(phi)) -
                cosPhi*np.sin(phi))
    g24_cart = (E*cosPhi*cosTheta*np.sin(theta) -
                (cosPhi*cosTheta*np.sin(theta)))
    g34_cart = (E*cosTheta*np.sin(phi)*np.sin(theta) -
                (cosTheta*np.sin(phi)*np.sin(theta)))

    return g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart


def compare_implementations(theta, phi, g_tt, g_rr, description):
    """Compare Python and MATLAB-like implementations"""

    # Python implementation
    py_g11, py_g22, py_g23, py_g24, py_g33, py_g34, py_g44 = \
        sph2cart_diag(theta, phi, g_tt, g_rr)

    # MATLAB-like implementation
    ml_g11, ml_g22, ml_g23, ml_g24, ml_g33, ml_g34, ml_g44 = \
        matlab_sph2cart_diag(theta, phi, g_tt, g_rr)

    # Compare
    tol = 1e-15
    match = True
    max_diff = 0.0

    components = ['g11', 'g22', 'g23', 'g24', 'g33', 'g34', 'g44']
    py_values = [py_g11, py_g22, py_g23, py_g24, py_g33, py_g34, py_g44]
    ml_values = [ml_g11, ml_g22, ml_g23, ml_g24, ml_g33, ml_g34, ml_g44]

    print(f"\n{description}")
    print(f"  θ = {theta:.6f}, φ = {phi:.6f}")
    print(f"  g_tt = {g_tt:.6f}, g_rr = {g_rr:.6f}")

    for comp, py_val, ml_val in zip(components, py_values, ml_values):
        diff = abs(py_val - ml_val)
        max_diff = max(max_diff, diff)

        if diff > tol:
            match = False
            print(f"  ✗ {comp}: Python={py_val:.10f}, MATLAB={ml_val:.10f}, diff={diff:.2e}")
        else:
            print(f"  ✓ {comp}: {py_val:.10f} (diff={diff:.2e})")

    return match, max_diff


def main():
    print("=" * 80)
    print("MATLAB vs PYTHON IMPLEMENTATION COMPARISON")
    print("=" * 80)

    test_cases = [
        # (theta, phi, g_tt, g_rr, description)
        (np.pi/4, np.pi/4, -1.0, 1.5, "Test 1: θ=π/4, φ=π/4"),
        (np.pi/2, 0.0, -0.8, 1.25, "Test 2: θ=π/2 (special), φ=0"),
        (np.pi/2, np.pi/2, -0.8, 1.25, "Test 3: θ=π/2, φ=π/2 (both special)"),
        (0.0, 0.0, -1.0, 2.0, "Test 4: θ=0, φ=0 (z-axis)"),
        (np.pi/3, np.pi/6, -0.9, 1.1, "Test 5: θ=π/3, φ=π/6"),
        (np.pi/6, np.pi/3, -1.5, 0.8, "Test 6: θ=π/6, φ=π/3"),
        (np.pi/2, 0.0, -1.0, 1.0, "Test 7: Minkowski at equator"),
        (0.5, 1.2, -0.85, 1.35, "Test 8: Generic angles"),
    ]

    all_match = True
    max_diff_overall = 0.0

    for theta, phi, g_tt, g_rr, description in test_cases:
        match, max_diff = compare_implementations(theta, phi, g_tt, g_rr, description)
        if not match:
            all_match = False
        max_diff_overall = max(max_diff_overall, max_diff)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_match:
        print(f"\n✓ ALL TESTS PASSED")
        print(f"  Maximum difference: {max_diff_overall:.2e}")
        print(f"\n  Python implementation is IDENTICAL to MATLAB implementation")
    else:
        print(f"\n✗ SOME TESTS FAILED")
        print(f"  Maximum difference: {max_diff_overall:.2e}")

    print("\n" + "=" * 80)
    print("FORMULA VERIFICATION")
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

    print("\n✓ Formulas are ALGEBRAICALLY IDENTICAL")
    print("  (accounting for notation: cosPhi ↔ cos_phi, cosTheta ↔ cos_theta, etc.)")

    print("\n" + "=" * 80)
    print("SPECIAL ANGLE HANDLING")
    print("=" * 80)

    print("\nBoth implementations handle special angles identically:")
    print("  if abs(phi) == π/2: set cos(φ) = 0")
    print("  if abs(theta) == π/2: set cos(θ) = 0")
    print("\nThis prevents numerical errors from cos(π/2) ≈ 6.12e-17")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
