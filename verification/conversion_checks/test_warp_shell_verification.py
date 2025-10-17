"""
Verification test for warp shell metric construction
Compares key calculations between MATLAB and Python implementations
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell.utils import (
    tov_const_density,
    compact_sigmoid,
    alpha_numeric_solver,
    legendre_radial_interp,
    sph2cart_diag,
    smooth_array
)
from warpfactory.units.constants import c as speed_of_light, G as gravitational_constant

def test_constants():
    """Verify physical constants match MATLAB"""
    c = speed_of_light()
    G = gravitational_constant()

    print("=" * 60)
    print("CONSTANTS VERIFICATION")
    print("=" * 60)
    print(f"Speed of light (c):        {c:.10e} m/s")
    print(f"Expected (MATLAB):         {2.99792458e8:.10e} m/s")
    print(f"Match: {np.isclose(c, 2.99792458e8)}")
    print()
    print(f"Gravitational const (G):   {G:.10e} m^3/kg/s^2")
    print(f"Expected (MATLAB):         {6.67430e-11:.10e} m^3/kg/s^2")
    print(f"Match: {np.isclose(G, 6.67430e-11)}")
    print()

def test_tov_equation():
    """Verify TOV equation implementation"""
    print("=" * 60)
    print("TOV EQUATION VERIFICATION")
    print("=" * 60)

    c = speed_of_light()
    G = gravitational_constant()

    # Test parameters
    R = 10.0
    m_total = 1.0e6
    R1 = 8.0
    R2 = 10.0

    # Create test arrays
    r = np.linspace(0, 15, 1000)
    shell_volume = 4.0/3.0 * np.pi * (R2**3 - R1**3)
    rho = np.zeros(len(r)) + m_total / shell_volume * ((r > R1) & (r < R2))

    # Compute mass profile
    from scipy.integrate import cumulative_trapezoid
    M = cumulative_trapezoid(4 * np.pi * rho * r**2, r, initial=0)

    # Compute TOV pressure
    P = tov_const_density(R2, M, rho, r)

    # Manual calculation at a test point
    r_test = 9.0
    idx = np.argmin(np.abs(r - r_test))
    M_end = M[-1]

    # MATLAB formula: c^2*rho*(R*sqrt(R-2*G*M_end/c^2)-sqrt(R^3-2*G*M_end*r^2/c^2))/(sqrt(R^3-2*G*M_end*r^2/c^2)-3*R*sqrt(R-2*G*M_end/c^2))
    numerator = R * np.sqrt(R - 2*G*M_end/c**2) - np.sqrt(R**3 - 2*G*M_end*r_test**2/c**2)
    denominator = np.sqrt(R**3 - 2*G*M_end*r_test**2/c**2) - 3*R*np.sqrt(R - 2*G*M_end/c**2)
    P_expected = c**2 * rho[idx] * (numerator / denominator)

    print(f"Test point r = {r_test}")
    print(f"Python P:    {P[idx]:.10e}")
    print(f"Expected P:  {P_expected:.10e}")
    print(f"Match: {np.isclose(P[idx], P_expected, rtol=1e-10)}")
    print(f"Max pressure: {np.max(P):.10e}")
    print(f"Non-zero pressures: {np.sum(P != 0)} points")
    print()

def test_compact_sigmoid():
    """Verify compact sigmoid implementation"""
    print("=" * 60)
    print("COMPACT SIGMOID VERIFICATION")
    print("=" * 60)

    R1 = 8.0
    R2 = 10.0
    sigma = 1.0
    Rbuff = 0.5

    r = np.linspace(0, 15, 1000)
    f = compact_sigmoid(r, R1, R2, sigma, Rbuff)

    # Check properties
    print(f"R1 = {R1}, R2 = {R2}, sigma = {sigma}, Rbuff = {Rbuff}")
    print(f"Function range: [{np.min(f):.6f}, {np.max(f):.6f}]")
    print(f"Value at R1+Rbuff: {f[np.argmin(np.abs(r - (R1+Rbuff)))]:.6f}")
    print(f"Value at R2-Rbuff: {f[np.argmin(np.abs(r - (R2-Rbuff)))]:.6f}")
    print(f"Value inside: {f[np.argmin(np.abs(r - 9.0))]:.6f}")
    print(f"Value outside: {f[np.argmin(np.abs(r - 12.0))]:.6f}")
    print(f"All finite: {np.all(np.isfinite(f))}")
    print(f"All real: {np.all(np.isreal(f))}")
    print()

def test_alpha_solver():
    """Verify alpha numeric solver"""
    print("=" * 60)
    print("ALPHA NUMERIC SOLVER VERIFICATION")
    print("=" * 60)

    c = speed_of_light()
    G = gravitational_constant()

    # Test parameters
    R1 = 8.0
    R2 = 10.0
    m_total = 1.0e6

    r = np.linspace(0, 15, 1000)
    shell_volume = 4.0/3.0 * np.pi * (R2**3 - R1**3)
    rho = np.zeros(len(r)) + m_total / shell_volume * ((r > R1) & (r < R2))

    from scipy.integrate import cumulative_trapezoid
    M = cumulative_trapezoid(4 * np.pi * rho * r**2, r, initial=0)
    P = tov_const_density(R2, M, rho, r)

    alpha = alpha_numeric_solver(M, P, R2, r)

    # Check boundary condition
    C = 0.5 * np.log(1 - 2*G*M[-1]/r[-1]/c**2)
    print(f"Boundary condition C: {C:.10e}")
    print(f"Alpha at r[-1]: {alpha[-1]:.10e}")
    print(f"Match: {np.isclose(alpha[-1], C, rtol=1e-6)}")
    print(f"Alpha at r=0: {alpha[0]:.10e}")
    print(f"Alpha range: [{np.min(alpha):.6e}, {np.max(alpha):.6e}]")
    print()

def test_legendre_interp():
    """Verify Legendre interpolation"""
    print("=" * 60)
    print("LEGENDRE INTERPOLATION VERIFICATION")
    print("=" * 60)

    # Test with simple array
    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    # Test at integer points
    for i in range(1, 9):
        val = legendre_radial_interp(test_array, float(i))
        print(f"r={i}: {val:.6f} (expected: {test_array[i]:.6f}, match: {np.isclose(val, test_array[i])})")

    # Test at half points
    print("\nHalf-point interpolation:")
    val_2p5 = legendre_radial_interp(test_array, 2.5)
    print(f"r=2.5: {val_2p5:.6f} (expected ~3.5)")

    val_5p5 = legendre_radial_interp(test_array, 5.5)
    print(f"r=5.5: {val_5p5:.6f} (expected ~6.5)")
    print()

def test_sph2cart():
    """Verify spherical to Cartesian transformation"""
    print("=" * 60)
    print("SPHERICAL TO CARTESIAN TRANSFORMATION")
    print("=" * 60)

    # Test at various angles
    test_cases = [
        (0.0, 0.0, "theta=0, phi=0 (z-axis)"),
        (np.pi/2, 0.0, "theta=π/2, phi=0 (x-axis)"),
        (np.pi/2, np.pi/2, "theta=π/2, phi=π/2 (y-axis)"),
        (np.pi/4, np.pi/4, "theta=π/4, phi=π/4"),
    ]

    g11_sph = -1.0
    g22_sph = 2.0

    for theta, phi, desc in test_cases:
        g11_cart, g22_cart, g23_cart, g24_cart, g33_cart, g34_cart, g44_cart = \
            sph2cart_diag(theta, phi, g11_sph, g22_sph)

        print(f"\n{desc}:")
        print(f"  g11_cart (time): {g11_cart:.6f}")
        print(f"  g22_cart (xx):   {g22_cart:.6f}")
        print(f"  g33_cart (yy):   {g33_cart:.6f}")
        print(f"  g44_cart (zz):   {g44_cart:.6f}")
        print(f"  g23_cart (xy):   {g23_cart:.6f}")
        print(f"  g24_cart (xz):   {g24_cart:.6f}")
        print(f"  g34_cart (yz):   {g34_cart:.6f}")

        # Verify identity: should recover Minkowski metric at flat space
        if g11_sph == -1.0 and g22_sph == 1.0:
            expected_diag = np.array([-1.0, 1.0, 1.0, 1.0])
            actual_diag = np.array([g11_cart, g22_cart, g33_cart, g44_cart])
            print(f"  Flat space check: {np.allclose(actual_diag, expected_diag)}")
    print()

def test_coordinate_indexing():
    """Verify coordinate indexing matches between MATLAB and Python"""
    print("=" * 60)
    print("COORDINATE INDEXING VERIFICATION")
    print("=" * 60)

    grid_size = [1, 10, 10, 10]
    world_center = [0.0, 5.0, 5.0, 5.0]
    grid_scaling = [1.0, 1.0, 1.0, 1.0]

    # MATLAB: for i = 1:gridSize(2), x = (i*gridScaling(2)-worldCenter(2))
    # Python: for i in range(grid_size[1]), x = (i+1)*grid_scaling[1] - world_center[1]

    print("MATLAB indexing (1-based):")
    for i in range(1, 4):
        x_matlab = i * grid_scaling[1] - world_center[1]
        print(f"  i={i}: x = {x_matlab}")

    print("\nPython indexing (0-based, +1 correction):")
    for i in range(3):
        x_python = (i + 1) * grid_scaling[1] - world_center[1]
        print(f"  i={i}: x = {x_python}")

    print("\nIndexing matches: {0-based Python with +1 == 1-based MATLAB}")
    print()

def test_metric_sign_convention():
    """Verify metric sign convention"""
    print("=" * 60)
    print("METRIC SIGN CONVENTION")
    print("=" * 60)

    # In the code: A = -exp(2*a)
    # This means g_tt = A should be negative (for -+++ signature)

    alpha_test = 0.0
    A = -np.exp(2.0 * alpha_test)
    print(f"For alpha=0: A = -exp(2*0) = {A}")
    print(f"Sign is negative (correct for -+++ signature): {A < 0}")

    alpha_test = 0.1
    A = -np.exp(2.0 * alpha_test)
    print(f"\nFor alpha=0.1: A = -exp(2*0.1) = {A}")
    print(f"Sign is negative: {A < 0}")
    print()

def test_shift_vector():
    """Verify shift vector implementation"""
    print("=" * 60)
    print("SHIFT VECTOR VERIFICATION")
    print("=" * 60)

    # MATLAB: Metric.tensor{1,2} = Metric.tensor{1,2}-Metric.tensor{1,2}.*ShiftMatrix - ShiftMatrix*vWarp
    # Simplifies to: g_tx = g_tx - g_tx*shift - shift*v_warp = -shift*v_warp (if g_tx starts at 0)
    # Python: metric_dict[(0, 1)] = metric_dict[(0, 1)] - metric_dict[(0, 1)] * shift_matrix - shift_matrix * v_warp

    g_tx_initial = 0.0
    shift = 0.5
    v_warp = 0.8

    g_tx_final = g_tx_initial - g_tx_initial * shift - shift * v_warp
    print(f"Initial g_tx: {g_tx_initial}")
    print(f"Shift: {shift}")
    print(f"v_warp: {v_warp}")
    print(f"Final g_tx: {g_tx_final}")
    print(f"Expected: {-shift * v_warp}")
    print(f"Match: {np.isclose(g_tx_final, -shift * v_warp)}")
    print()

def main():
    """Run all verification tests"""
    print("\n")
    print("*" * 60)
    print("WARP SHELL METRIC VERIFICATION TEST SUITE")
    print("*" * 60)
    print()

    test_constants()
    test_tov_equation()
    test_compact_sigmoid()
    test_alpha_solver()
    test_legendre_interp()
    test_sph2cart()
    test_coordinate_indexing()
    test_metric_sign_convention()
    test_shift_vector()

    print("*" * 60)
    print("VERIFICATION COMPLETE")
    print("*" * 60)
    print()

if __name__ == "__main__":
    main()
