"""
Test script to verify Ricci tensor calculation

Tests the Python Ricci tensor implementation against known exact solutions:
1. Minkowski spacetime (should give zero Ricci tensor)
2. Schwarzschild spacetime (should give zero Ricci tensor in vacuum)
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.solver.ricci import calculate_ricci_tensor, calculate_ricci_scalar
from warpfactory.units.constants import c as speed_of_light


def setup_minkowski_metric(nx=11, ny=11, nz=11, nt=11):
    """
    Create Minkowski metric: η_μν = diag(-1, 1, 1, 1)

    Returns:
        gl: Covariant metric tensor (dictionary)
        gu: Contravariant metric tensor (dictionary)
        delta: Grid spacing
    """
    # Grid spacing
    dx = 1.0
    dy = 1.0
    dz = 1.0
    dt = 1.0
    delta = [dt, dx, dy, dz]

    # Create flat spacetime metric
    gl = {}
    gu = {}

    for i in range(4):
        for j in range(4):
            gl[(i, j)] = np.zeros((nt, nx, ny, nz))
            gu[(i, j)] = np.zeros((nt, nx, ny, nz))

            if i == j:
                if i == 0:  # Time component
                    gl[(i, j)] = -np.ones((nt, nx, ny, nz))
                    gu[(i, j)] = -np.ones((nt, nx, ny, nz))
                else:  # Space components
                    gl[(i, j)] = np.ones((nt, nx, ny, nz))
                    gu[(i, j)] = np.ones((nt, nx, ny, nz))

    return gl, gu, delta


def setup_schwarzschild_metric(nx=11, ny=11, nz=11, nt=11, M=1.0):
    """
    Create Schwarzschild metric in Schwarzschild coordinates (t, r, theta, phi)

    In isotropic coordinates (which can map to Cartesian):
    ds² = -(1-M/2r)²/(1+M/2r)² dt² + (1+M/2r)⁴ (dx² + dy² + dz²)

    For simplicity, we'll test a weak field approximation or just use
    a region far from the singularity.

    Returns:
        gl: Covariant metric tensor (dictionary)
        gu: Contravariant metric tensor (dictionary)
        delta: Grid spacing
    """
    # Grid spacing
    dx = 1.0
    dy = 1.0
    dz = 1.0
    dt = 1.0
    delta = [dt, dx, dy, dz]

    # Create coordinate grids (far from black hole, r >> M)
    x = np.linspace(10, 20, nx)
    y = np.linspace(10, 20, ny)
    z = np.linspace(10, 20, nz)
    t = np.linspace(0, 10, nt)

    T, X, Y, Z = np.meshgrid(t, x, y, z, indexing='ij')

    # Radius from center
    r = np.sqrt(X**2 + Y**2 + Z**2)

    # Schwarzschild metric in isotropic coordinates
    psi = 1 + M / (2 * r)

    gl = {}
    gu = {}

    # Initialize all components to zero
    for i in range(4):
        for j in range(4):
            gl[(i, j)] = np.zeros((nt, nx, ny, nz))
            gu[(i, j)] = np.zeros((nt, nx, ny, nz))

    # Metric components
    # g_00 = -(1-M/2r)²/(1+M/2r)²
    gl[(0, 0)] = -((1 - M/(2*r)) / psi)**2
    # g_11 = g_22 = g_33 = (1+M/2r)⁴
    conformal_factor = psi**4
    gl[(1, 1)] = conformal_factor
    gl[(2, 2)] = conformal_factor
    gl[(3, 3)] = conformal_factor

    # Contravariant components (inverse)
    gu[(0, 0)] = 1.0 / gl[(0, 0)]
    gu[(1, 1)] = 1.0 / gl[(1, 1)]
    gu[(2, 2)] = 1.0 / gl[(2, 2)]
    gu[(3, 3)] = 1.0 / gl[(3, 3)]

    return gl, gu, delta


def test_minkowski():
    """Test that Minkowski spacetime gives zero Ricci tensor"""
    print("=" * 60)
    print("Testing Minkowski Spacetime")
    print("=" * 60)

    gl, gu, delta = setup_minkowski_metric(nx=11, ny=11, nz=11, nt=11)

    print("\nMetric components (should be constant):")
    print(f"g_00 = {gl[(0, 0)][5, 5, 5, 5]}")
    print(f"g_11 = {gl[(1, 1)][5, 5, 5, 5]}")
    print(f"g_22 = {gl[(2, 2)][5, 5, 5, 5]}")
    print(f"g_33 = {gl[(3, 3)][5, 5, 5, 5]}")

    print("\nCalculating Ricci tensor...")
    R_munu = calculate_ricci_tensor(gu, gl, delta)

    print("\nRicci tensor components (should all be ~0):")
    max_error = 0.0
    for i in range(4):
        for j in range(i, 4):
            component = R_munu[(i, j)]
            max_val = np.max(np.abs(component))
            mean_val = np.mean(np.abs(component))
            max_error = max(max_error, max_val)
            print(f"R_{i}{j}: max={max_val:.2e}, mean={mean_val:.2e}")

    # Calculate Ricci scalar
    print("\nCalculating Ricci scalar...")
    R = calculate_ricci_scalar(R_munu, gu)
    print(f"Ricci scalar: max={np.max(np.abs(R)):.2e}, mean={np.mean(np.abs(R)):.2e}")

    # Check if test passes
    tolerance = 1e-10
    if max_error < tolerance:
        print(f"\n✓ TEST PASSED: All components < {tolerance}")
        return True
    else:
        print(f"\n✗ TEST FAILED: Max error {max_error:.2e} exceeds tolerance {tolerance}")
        return False


def test_schwarzschild():
    """Test that Schwarzschild spacetime gives zero Ricci tensor (vacuum solution)"""
    print("\n" + "=" * 60)
    print("Testing Schwarzschild Spacetime")
    print("=" * 60)

    gl, gu, delta = setup_schwarzschild_metric(nx=11, ny=11, nz=11, nt=11, M=1.0)

    print("\nMetric components at center point:")
    print(f"g_00 = {gl[(0, 0)][5, 5, 5, 5]}")
    print(f"g_11 = {gl[(1, 1)][5, 5, 5, 5]}")
    print(f"g_22 = {gl[(2, 2)][5, 5, 5, 5]}")
    print(f"g_33 = {gl[(3, 3)][5, 5, 5, 5]}")

    print("\nCalculating Ricci tensor...")
    R_munu = calculate_ricci_tensor(gu, gl, delta)

    print("\nRicci tensor components (should all be ~0 for vacuum):")
    max_error = 0.0
    for i in range(4):
        for j in range(i, 4):
            component = R_munu[(i, j)]
            max_val = np.max(np.abs(component))
            mean_val = np.mean(np.abs(component))
            max_error = max(max_error, max_val)
            print(f"R_{i}{j}: max={max_val:.2e}, mean={mean_val:.2e}")

    # Calculate Ricci scalar
    print("\nCalculating Ricci scalar...")
    R = calculate_ricci_scalar(R_munu, gu)
    print(f"Ricci scalar: max={np.max(np.abs(R)):.2e}, mean={np.mean(np.abs(R)):.2e}")

    # For Schwarzschild, expect small errors due to finite differences
    tolerance = 1e-3  # More lenient for curved spacetime
    if max_error < tolerance:
        print(f"\n✓ TEST PASSED: All components < {tolerance}")
        return True
    else:
        print(f"\n✗ TEST WARNING: Max error {max_error:.2e} exceeds tolerance {tolerance}")
        print("(This may be acceptable for finite difference approximation)")
        return False


def detailed_component_analysis():
    """
    Detailed analysis of specific Ricci tensor computation steps
    """
    print("\n" + "=" * 60)
    print("Detailed Component Analysis")
    print("=" * 60)

    # Small test case for detailed inspection
    gl, gu, delta = setup_minkowski_metric(nx=7, ny=7, nz=7, nt=7)

    print("\nTesting finite difference operations...")
    print(f"Speed of light c = {speed_of_light()}")
    print(f"Grid spacing: dt={delta[0]}, dx={delta[1]}, dy={delta[2]}, dz={delta[3]}")

    # Test a simple derivative
    from warpfactory.solver.finite_differences import take_finite_difference_1, take_finite_difference_2

    # Create a linear function in x direction
    test_field = np.zeros((7, 7, 7, 7))
    for i in range(7):
        test_field[:, i, :, :] = i * 1.0

    df_dx = take_finite_difference_1(test_field, 1, delta, False)
    print(f"\nTest field (linear in x): f = x")
    print(f"df/dx (should be ~1.0): {df_dx[3, 3, 3, 3]}")

    d2f_dx2 = take_finite_difference_2(test_field, 1, 1, delta, False)
    print(f"d²f/dx² (should be ~0.0): {d2f_dx2[3, 3, 3, 3]}")


if __name__ == "__main__":
    print("Ricci Tensor Verification Test Suite")
    print("=" * 60)

    # Run detailed analysis first
    detailed_component_analysis()

    # Test Minkowski
    test1_passed = test_minkowski()

    # Test Schwarzschild
    test2_passed = test_schwarzschild()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Minkowski test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Schwarzschild test: {'PASSED/WARNING' if test2_passed else 'FAILED'}")

    if test1_passed:
        print("\n✓ Core implementation appears correct for flat spacetime")
    else:
        print("\n✗ Issues detected in implementation")
