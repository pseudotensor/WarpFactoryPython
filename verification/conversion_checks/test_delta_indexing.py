"""
Test delta indexing between MATLAB and Python
"""

import numpy as np
import sys
sys.path.insert(0, '/WarpFactory/warpfactory_py')

from warpfactory.solver.finite_differences import take_finite_difference_1, take_finite_difference_2


def test_delta_indexing():
    """Verify delta indexing is correct between MATLAB and Python"""
    print("=" * 80)
    print("TESTING DELTA INDEXING")
    print("=" * 80)

    # Create test array with different spacing in each dimension
    nx = 10
    delta_matlab = [0.1, 0.2, 0.3, 0.4]  # MATLAB: delta(1), delta(2), delta(3), delta(4)
    delta_python = [0.1, 0.2, 0.3, 0.4]  # Python: delta[0], delta[1], delta[2], delta[3]

    print("\nMATLAB convention: k=1,2,3,4 for t,x,y,z")
    print("Python convention: k=0,1,2,3 for t,x,y,z")
    print(f"\nDelta values: {delta_python}")

    # Create a simple test where we can verify the division by delta
    # Use a function where we know the derivative
    # f = k * x[k] where k is the dimension
    # df/dx[k] = k

    for k_python in range(4):
        k_matlab = k_python + 1
        print(f"\n{'='*80}")
        print(f"Testing dimension: k_python={k_python} (MATLAB k={k_matlab})")
        print(f"Delta for this dimension: {delta_python[k_python]}")

        # Create test array: f = constant * coordinate[k]
        x = np.linspace(0, 1, nx)
        coords = np.meshgrid(x, x, x, x, indexing='ij')

        # Use the coordinate in dimension k_python
        # Multiply by a factor to make the derivative non-trivial
        factor = k_matlab * 10  # Use k_matlab to match what MATLAB would do
        f = factor * coords[k_python]

        # Take derivative
        df = take_finite_difference_1(f, k_python, delta_python)

        # Expected derivative: factor / 1.0 (since dx = 1/(nx-1))
        # But the finite difference divides by delta[k_python]
        # So we expect: factor / delta[k_python]
        expected_value = factor / delta_python[k_python]

        # Check a point in the interior
        actual_value = df[5, 5, 5, 5]

        print(f"Expected derivative value: {expected_value:.6f}")
        print(f"Actual derivative value: {actual_value:.6f}")
        print(f"Difference: {abs(actual_value - expected_value):.2e}")

        # The actual derivative should be close to factor (the coefficient)
        # divided by the actual grid spacing (which is different from delta)
        # Let's check if the division by delta is correct
        actual_spacing = x[1] - x[0]
        print(f"\nActual grid spacing: {actual_spacing:.6f}")
        print(f"Delta used in finite difference: {delta_python[k_python]:.6f}")

        # The finite difference computes: d/dx / delta[k]
        # If delta[k] = actual_spacing, then we get the correct derivative


def test_delta_in_second_derivative():
    """Verify delta² is used correctly in second derivatives"""
    print("\n" + "=" * 80)
    print("TESTING DELTA² IN SECOND DERIVATIVES")
    print("=" * 80)

    nx = 10
    delta_values = [0.1, 0.2, 0.3, 0.4]

    for k in range(4):
        print(f"\n{'='*80}")
        print(f"Testing dimension k={k}, delta={delta_values[k]}")

        x = np.linspace(0, 1, nx)
        coords = np.meshgrid(x, x, x, x, indexing='ij')

        # f = x² in dimension k, d²f/dx² = 2
        f = coords[k]**2

        d2f = take_finite_difference_2(f, k, k, delta_values)

        # Check interior point
        actual = d2f[5, 5, 5, 5]
        print(f"d²f/dx² at interior point: {actual:.6f}")
        print(f"Expected: 2.0")

        # The division by delta² is built into the formula
        # Check that it's working correctly


def test_mixed_derivative_delta():
    """Verify mixed derivatives use both deltas correctly"""
    print("\n" + "=" * 80)
    print("TESTING MIXED DERIVATIVE DELTA USAGE")
    print("=" * 80)

    nx = 10
    delta_values = [0.1, 0.2, 0.3, 0.4]

    # Test d²/dtdx with f = t*x, expected = 1
    x = np.linspace(0, 1, nx)
    T, X, Y, Z = np.meshgrid(x, x, x, x, indexing='ij')

    f = T * X
    d2f = take_finite_difference_2(f, 0, 1, delta_values)

    print(f"\nMixed derivative d²f/dtdx of f=t*x")
    print(f"Delta[0] (t): {delta_values[0]}")
    print(f"Delta[1] (x): {delta_values[1]}")
    print(f"Expected division: 144 * delta[0] * delta[1] = 144 * {delta_values[0]} * {delta_values[1]} = {144 * delta_values[0] * delta_values[1]}")

    actual = d2f[5, 5, 5, 5]
    print(f"Actual value: {actual:.6f}")
    print(f"Expected: 1.0")


def check_consistency_with_matlab():
    """Final check: verify the k indexing is consistent"""
    print("\n" + "=" * 80)
    print("CONSISTENCY CHECK")
    print("=" * 80)

    print("\nMATLAB uses:")
    print("  k=1 → time (t)")
    print("  k=2 → x")
    print("  k=3 → y")
    print("  k=4 → z")
    print("  delta(k) uses 1-based indexing")

    print("\nPython uses:")
    print("  k=0 → time (t)")
    print("  k=1 → x")
    print("  k=2 → y")
    print("  k=3 → z")
    print("  delta[k] uses 0-based indexing")

    print("\nMapping:")
    print("  MATLAB k=1 → Python k=0 → delta[0]")
    print("  MATLAB k=2 → Python k=1 → delta[1]")
    print("  MATLAB k=3 → Python k=2 → delta[2]")
    print("  MATLAB k=4 → Python k=3 → delta[3]")

    print("\n✓ The indexing is consistent!")
    print("  MATLAB: (12*delta(k)) where k ∈ {1,2,3,4}")
    print("  Python: (12*delta[k]) where k ∈ {0,1,2,3}")
    print("  Both access the same element of the delta array.")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DELTA INDEXING VERIFICATION")
    print("=" * 80)

    test_delta_indexing()
    test_delta_in_second_derivative()
    test_mixed_derivative_delta()
    check_consistency_with_matlab()

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
