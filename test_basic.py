#!/usr/bin/env python3
"""
Basic test script to verify WarpFactory Python package installation and core functionality
"""

import numpy as np
import sys

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        import warpfactory
        from warpfactory import units, metrics, core
        from warpfactory.units import c, G
        from warpfactory.core import Tensor, verify_tensor, c4_inv
        from warpfactory.metrics import three_plus_one, minkowski, alcubierre
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_constants():
    """Test physical constants"""
    print("\nTesting physical constants...")
    from warpfactory.units import c, G

    speed_of_light = c()
    grav_constant = G()

    print(f"  Speed of light: {speed_of_light:.2e} m/s")
    print(f"  Gravitational constant: {grav_constant:.2e} m³/kg/s²")

    assert abs(speed_of_light - 2.99792458e8) < 1e-6, "Speed of light incorrect"
    assert abs(grav_constant - 6.67430e-11) < 1e-15, "G constant incorrect"
    print("✓ Physical constants correct")
    return True


def test_minkowski_metric():
    """Test Minkowski metric creation"""
    print("\nTesting Minkowski metric...")
    from warpfactory.metrics.minkowski import get_minkowski_metric
    from warpfactory.core import verify_tensor

    grid_size = [5, 10, 10, 10]
    metric = get_minkowski_metric(grid_size)

    print(f"  Grid size: {grid_size}")
    print(f"  Metric type: {metric.type}")
    print(f"  Metric name: {metric.name}")
    print(f"  Metric index: {metric.index}")
    print(f"  Tensor shape: {metric.shape}")

    # Verify g_00 = -1 everywhere
    assert np.all(metric[(0, 0)] == -1), "g_00 should be -1"

    # Verify g_11 = g_22 = g_33 = 1 everywhere
    assert np.all(metric[(1, 1)] == 1), "g_11 should be 1"
    assert np.all(metric[(2, 2)] == 1), "g_22 should be 1"
    assert np.all(metric[(3, 3)] == 1), "g_33 should be 1"

    # Verify off-diagonal terms are zero
    for i in range(4):
        for j in range(4):
            if i != j:
                assert np.all(metric[(i, j)] == 0), f"g_{i}{j} should be 0"

    # Verify tensor structure
    is_valid = verify_tensor(metric, suppress_msgs=True)
    assert is_valid, "Metric tensor validation failed"

    print("✓ Minkowski metric correct")
    return True


def test_alcubierre_metric():
    """Test Alcubierre metric creation"""
    print("\nTesting Alcubierre metric...")
    from warpfactory.metrics.alcubierre import get_alcubierre_metric
    from warpfactory.core import verify_tensor

    grid_size = [5, 10, 10, 10]
    world_center = [2.5, 5.0, 5.0, 5.0]
    velocity = 0.5
    radius = 2.0
    sigma = 0.5

    metric = get_alcubierre_metric(
        grid_size, world_center, velocity, radius, sigma
    )

    print(f"  Grid size: {grid_size}")
    print(f"  Warp velocity: {velocity}c")
    print(f"  Bubble radius: {radius}")
    print(f"  Metric name: {metric.name}")
    print(f"  Tensor shape: {metric.shape}")

    # Verify tensor structure
    is_valid = verify_tensor(metric, suppress_msgs=True)
    assert is_valid, "Alcubierre metric validation failed"

    # Check that metric is modified from Minkowski
    # The shift vector beta should be non-zero in some regions
    beta_x = metric[(0, 1)]
    assert not np.all(beta_x == 0), "Shift vector should be non-zero"

    print(f"  Max shift vector magnitude: {np.abs(beta_x).max():.4f}")
    print("✓ Alcubierre metric correct")
    return True


def test_tensor_operations():
    """Test tensor operations"""
    print("\nTesting tensor operations...")
    from warpfactory.core import c3_inv, c4_inv, c_det

    # Test 3x3 inversion with identity matrix
    identity_3 = {
        (0, 0): np.ones((5, 5, 5, 5)),
        (0, 1): np.zeros((5, 5, 5, 5)),
        (0, 2): np.zeros((5, 5, 5, 5)),
        (1, 0): np.zeros((5, 5, 5, 5)),
        (1, 1): np.ones((5, 5, 5, 5)),
        (1, 2): np.zeros((5, 5, 5, 5)),
        (2, 0): np.zeros((5, 5, 5, 5)),
        (2, 1): np.zeros((5, 5, 5, 5)),
        (2, 2): np.ones((5, 5, 5, 5)),
    }

    inv_3 = c3_inv(identity_3)

    # Check that inverse of identity is identity
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert np.allclose(inv_3[(i, j)], expected), f"3x3 inv failed at ({i},{j})"

    print("  ✓ 3x3 matrix inversion works")

    # Test 4x4 inversion with Minkowski metric
    from warpfactory.metrics.minkowski import get_minkowski_metric
    metric = get_minkowski_metric([5, 5, 5, 5])

    # Invert the metric
    inv_metric = c4_inv(metric.tensor)

    # For Minkowski, inverse should flip signs on diagonal
    assert np.allclose(inv_metric[(0, 0)], -1), "Inverse g^00 should be -1"
    assert np.allclose(inv_metric[(1, 1)], 1), "Inverse g^11 should be 1"

    print("  ✓ 4x4 matrix inversion works")

    # Test determinant
    det = c_det(metric.tensor)
    assert np.allclose(det, -1), "Minkowski determinant should be -1"

    print("  ✓ 4x4 determinant calculation works")
    print("✓ All tensor operations work correctly")
    return True


def test_3plus1_decomposition():
    """Test 3+1 decomposition"""
    print("\nTesting 3+1 decomposition...")
    from warpfactory.metrics.three_plus_one import (
        set_minkowski_three_plus_one,
        three_plus_one_builder,
        three_plus_one_decomposer
    )
    from warpfactory.metrics.minkowski import get_minkowski_metric
    from warpfactory.core import Tensor

    grid_size = [5, 5, 5, 5]

    # Create Minkowski in 3+1 form
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    # Check values
    assert np.all(alpha == 1), "Lapse should be 1 for Minkowski"
    for i in range(3):
        assert np.all(beta[i] == 0), f"Shift vector {i} should be 0"
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert np.allclose(gamma[(i, j)], expected), f"Gamma_{i}{j} incorrect"

    print("  ✓ 3+1 Minkowski decomposition correct")

    # Build metric from 3+1 components
    metric_dict = three_plus_one_builder(alpha, beta, gamma)

    # Verify it matches Minkowski
    assert np.allclose(metric_dict[(0, 0)], -1), "Built g_00 incorrect"
    assert np.allclose(metric_dict[(1, 1)], 1), "Built g_11 incorrect"

    print("  ✓ 3+1 builder works correctly")

    print("✓ 3+1 decomposition works correctly")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("WarpFactory Python Package - Basic Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_constants,
        test_minkowski_metric,
        test_alcubierre_metric,
        test_tensor_operations,
        test_3plus1_decomposition,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 All tests passed! WarpFactory Python is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
