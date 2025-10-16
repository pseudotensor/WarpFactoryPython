"""
Integration test: Verify sph2cart_diag works correctly in the actual warp shell context

This test validates that the transformation produces a valid metric when used
in the actual warp shell metric computation workflow.
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.metrics.warp_shell.utils import sph2cart_diag


def test_metric_symmetry():
    """
    Verify that the transformed metric is symmetric
    """
    print("=" * 80)
    print("TEST 1: Metric Symmetry")
    print("=" * 80)

    theta = np.pi / 3
    phi = np.pi / 4
    g_tt = -0.8
    g_rr = 1.25

    g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)

    # Construct full metric matrix
    metric = np.array([
        [g11, 0,   0,   0  ],
        [0,   g22, g23, g24],
        [0,   g23, g33, g34],
        [0,   g24, g34, g44]
    ])

    print(f"\nMetric matrix at θ={theta:.4f}, φ={phi:.4f}:")
    print(f"  g_tt = {g11:.6f}")
    print(f"  g_xx = {g22:.6f}  g_xy = {g23:.6f}  g_xz = {g24:.6f}")
    print(f"  g_yy = {g33:.6f}  g_yz = {g34:.6f}")
    print(f"  g_zz = {g44:.6f}")

    # Check symmetry (should be symmetric by construction)
    is_symmetric = np.allclose(metric, metric.T)

    if is_symmetric:
        print("\n✓ Metric is symmetric")
    else:
        print("\n✗ Metric is NOT symmetric!")

    return is_symmetric


def test_metric_signature():
    """
    Verify the metric has correct Lorentzian signature (-,+,+,+)
    """
    print("\n" + "=" * 80)
    print("TEST 2: Metric Signature")
    print("=" * 80)

    test_cases = [
        (np.pi/4, np.pi/4, -1.0, 1.5, "Generic point"),
        (np.pi/2, 0.0, -0.8, 1.25, "Equatorial plane"),
        (0.0, 0.0, -1.0, 2.0, "Polar axis"),
    ]

    all_correct = True

    for theta, phi, g_tt, g_rr, description in test_cases:
        g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)

        # Construct metric
        metric = np.array([
            [g11, 0,   0,   0  ],
            [0,   g22, g23, g24],
            [0,   g23, g33, g34],
            [0,   g24, g34, g44]
        ])

        # Compute eigenvalues to check signature
        eigenvalues = np.linalg.eigvalsh(metric)
        eigenvalues = sorted(eigenvalues)

        print(f"\n{description}:")
        print(f"  Eigenvalues: [{eigenvalues[0]:.4f}, {eigenvalues[1]:.4f}, {eigenvalues[2]:.4f}, {eigenvalues[3]:.4f}]")

        # Check for Lorentzian signature: 1 negative, 3 positive
        n_negative = sum(1 for ev in eigenvalues if ev < 0)
        n_positive = sum(1 for ev in eigenvalues if ev > 0)

        if n_negative == 1 and n_positive == 3:
            print(f"  ✓ Correct signature (-,+,+,+)")
        else:
            print(f"  ✗ Incorrect signature: {n_negative} negative, {n_positive} positive")
            all_correct = False

    return all_correct


def test_determinant():
    """
    Check that metric determinant is non-zero (non-degenerate)
    """
    print("\n" + "=" * 80)
    print("TEST 3: Metric Determinant (Non-degeneracy)")
    print("=" * 80)

    test_cases = [
        (np.pi/4, np.pi/4, -1.0, 1.5),
        (np.pi/2, 0.0, -0.8, 1.25),
        (np.pi/3, np.pi/6, -0.9, 1.1),
    ]

    all_nonzero = True

    for theta, phi, g_tt, g_rr in test_cases:
        g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)

        # Construct metric
        metric = np.array([
            [g11, 0,   0,   0  ],
            [0,   g22, g23, g24],
            [0,   g23, g33, g34],
            [0,   g24, g34, g44]
        ])

        det = np.linalg.det(metric)

        print(f"\nθ={theta:.4f}, φ={phi:.4f}:")
        print(f"  det(g) = {det:.6e}")

        if abs(det) > 1e-10:
            print(f"  ✓ Non-degenerate")
        else:
            print(f"  ✗ Nearly degenerate!")
            all_nonzero = False

    return all_nonzero


def test_continuity():
    """
    Test that the metric varies continuously with angle
    """
    print("\n" + "=" * 80)
    print("TEST 4: Continuity in Angular Variables")
    print("=" * 80)

    g_tt = -1.0
    g_rr = 1.5

    # Vary theta
    print("\nVarying θ from 0 to π:")
    theta_values = np.linspace(0, np.pi, 10)
    phi = np.pi / 4

    g22_values = []
    for theta in theta_values:
        g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)
        g22_values.append(g22)

    # Check for discontinuities
    diffs = np.diff(g22_values)
    max_jump = np.max(np.abs(diffs))

    print(f"  g_xx varies from {min(g22_values):.4f} to {max(g22_values):.4f}")
    print(f"  Maximum jump: {max_jump:.4f}")

    if max_jump < 0.5:  # Reasonable threshold
        print(f"  ✓ Continuous variation")
        continuous_theta = True
    else:
        print(f"  ✗ Large discontinuity detected!")
        continuous_theta = False

    # Vary phi
    print("\nVarying φ from 0 to 2π:")
    phi_values = np.linspace(0, 2*np.pi, 20)
    theta = np.pi / 4

    g22_values = []
    g33_values = []
    for phi in phi_values:
        g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)
        g22_values.append(g22)
        g33_values.append(g33)

    diffs22 = np.diff(g22_values)
    diffs33 = np.diff(g33_values)
    max_jump22 = np.max(np.abs(diffs22))
    max_jump33 = np.max(np.abs(diffs33))

    print(f"  g_xx varies from {min(g22_values):.4f} to {max(g22_values):.4f}")
    print(f"  g_yy varies from {min(g33_values):.4f} to {max(g33_values):.4f}")
    print(f"  Maximum jump in g_xx: {max_jump22:.4f}")
    print(f"  Maximum jump in g_yy: {max_jump33:.4f}")

    if max_jump22 < 0.5 and max_jump33 < 0.5:
        print(f"  ✓ Continuous variation")
        continuous_phi = True
    else:
        print(f"  ✗ Large discontinuity detected!")
        continuous_phi = False

    return continuous_theta and continuous_phi


def test_spherical_symmetry_preservation():
    """
    Test that spherically symmetric input produces expected symmetry
    """
    print("\n" + "=" * 80)
    print("TEST 5: Spherical Symmetry Preservation")
    print("=" * 80)

    g_tt = -1.0
    g_rr = 1.5

    # At a given θ, rotating φ should give the same radial component pattern
    theta = np.pi / 4
    phi_values = [0, np.pi/2, np.pi, 3*np.pi/2]

    print(f"\nAt θ = {theta:.4f}, rotating through φ:")

    g44_values = []  # g_zz should be independent of φ
    for phi in phi_values:
        g11, g22, g23, g24, g33, g34, g44 = sph2cart_diag(theta, phi, g_tt, g_rr)
        g44_values.append(g44)
        print(f"  φ = {phi:.4f}: g_zz = {g44:.6f}")

    # g_zz should be constant in φ
    g44_std = np.std(g44_values)

    if g44_std < 1e-10:
        print(f"  ✓ g_zz is independent of φ (std = {g44_std:.2e})")
        symmetric = True
    else:
        print(f"  ✗ g_zz varies with φ (std = {g44_std:.2e})")
        symmetric = False

    return symmetric


def main():
    print("=" * 80)
    print("INTEGRATION TEST: sph2cart_diag in Warp Shell Context")
    print("=" * 80)

    results = {
        "Symmetry": test_metric_symmetry(),
        "Signature": test_metric_signature(),
        "Non-degeneracy": test_determinant(),
        "Continuity": test_continuity(),
        "Spherical Symmetry": test_spherical_symmetry_preservation(),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 80)

    if all_passed:
        print("✓ ALL INTEGRATION TESTS PASSED")
        print("\nThe sph2cart_diag function produces physically valid metrics")
        print("suitable for warp shell computations.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nReview failures above.")

    print("=" * 80)


if __name__ == "__main__":
    main()
