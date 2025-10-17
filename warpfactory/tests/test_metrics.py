"""
Unit tests for warpfactory.metrics module

Tests all metric implementations including Minkowski, Alcubierre, Schwarzschild, etc.
"""

import pytest
import numpy as np
from warpfactory.metrics.minkowski import get_minkowski_metric
from warpfactory.metrics.alcubierre import (
    get_alcubierre_metric,
    shape_function_alcubierre
)
from warpfactory.metrics.schwarzschild import get_schwarzschild_metric
from warpfactory.metrics.three_plus_one import (
    set_minkowski_three_plus_one,
    three_plus_one_builder,
    three_plus_one_decomposer
)
from warpfactory.core.tensor_ops import c_det, verify_tensor


class TestMinkowskiMetric:
    """Test Minkowski (flat spacetime) metric"""

    def test_minkowski_creation(self, medium_grid_size, grid_scale):
        """Test basic Minkowski metric creation"""
        metric = get_minkowski_metric(medium_grid_size, grid_scale)

        assert metric.name == "Minkowski", "Name should be 'Minkowski'"
        assert metric.type == "metric", "Type should be 'metric'"
        assert metric.index == "covariant", "Index should be 'covariant'"
        assert metric.shape == tuple(medium_grid_size), "Shape should match grid size"

    def test_minkowski_signature(self, minkowski_metric_small):
        """Test Minkowski signature (-,+,+,+)"""
        # Time component should be -1
        assert np.all(minkowski_metric_small[(0, 0)] == -1), "g_00 should be -1"

        # Spatial components should be +1
        assert np.all(minkowski_metric_small[(1, 1)] == 1), "g_11 should be 1"
        assert np.all(minkowski_metric_small[(2, 2)] == 1), "g_22 should be 1"
        assert np.all(minkowski_metric_small[(3, 3)] == 1), "g_33 should be 1"

    def test_minkowski_off_diagonal_zero(self, minkowski_metric_small):
        """Test that Minkowski off-diagonal components are zero"""
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert np.all(minkowski_metric_small[(i, j)] == 0), \
                        f"Off-diagonal ({i},{j}) should be zero"

    def test_minkowski_determinant(self, minkowski_metric_small):
        """Test Minkowski determinant"""
        det = c_det(minkowski_metric_small.tensor)
        assert np.allclose(det, -1.0), "Minkowski determinant should be -1"

    def test_minkowski_verification(self, minkowski_metric_small):
        """Test that Minkowski metric verifies correctly"""
        assert verify_tensor(minkowski_metric_small, suppress_msgs=True), \
            "Minkowski metric should verify"

    def test_minkowski_params(self, minkowski_metric_small, small_grid_size):
        """Test Minkowski metric parameters"""
        assert "gridSize" in minkowski_metric_small.params, \
            "Should have gridSize parameter"
        assert minkowski_metric_small.params["gridSize"] == small_grid_size, \
            "gridSize should match input"

    def test_minkowski_scaling(self):
        """Test Minkowski with custom scaling"""
        grid_size = [5, 5, 5, 5]
        custom_scale = [2.0, 3.0, 4.0, 5.0]
        metric = get_minkowski_metric(grid_size, custom_scale)

        assert metric.scaling == custom_scale, "Should store custom scaling"


class TestThreePlusOneDecomposition:
    """Test 3+1 decomposition utilities"""

    def test_minkowski_three_plus_one(self, small_grid_size):
        """Test 3+1 decomposition of Minkowski space"""
        alpha, beta, gamma = set_minkowski_three_plus_one(small_grid_size)

        # Lapse should be 1 everywhere
        assert np.all(alpha == 1), "Minkowski lapse should be 1"

        # Shift should be zero
        for i in range(3):
            assert np.all(beta[i] == 0), f"Minkowski shift[{i}] should be 0"

        # Spatial metric should be identity
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert np.all(gamma[(i, j)] == 1), \
                        f"Minkowski gamma[{i},{j}] should be 1"
                else:
                    assert np.all(gamma[(i, j)] == 0), \
                        f"Minkowski gamma[{i},{j}] should be 0"

    def test_three_plus_one_builder(self, small_grid_size):
        """Test building metric from 3+1 components"""
        alpha, beta, gamma = set_minkowski_three_plus_one(small_grid_size)
        metric_dict = three_plus_one_builder(alpha, beta, gamma)

        # Should have 16 components
        assert len(metric_dict) == 16, "Should have 16 metric components"

        # Check Minkowski signature
        assert np.allclose(metric_dict[(0, 0)], -1), "g_00 should be -1"
        for i in range(1, 4):
            assert np.allclose(metric_dict[(i, i)], 1), f"g_{i}{i} should be 1"

    def test_three_plus_one_roundtrip(self, minkowski_metric_small):
        """Test decompose -> rebuild gives same metric"""
        # Decompose
        alpha, beta_down, gamma_down, beta_up, gamma_up = \
            three_plus_one_decomposer(minkowski_metric_small)

        # Rebuild
        rebuilt_dict = three_plus_one_builder(alpha, beta_down, gamma_down)

        # Compare
        for i in range(4):
            for j in range(4):
                assert np.allclose(
                    rebuilt_dict[(i, j)],
                    minkowski_metric_small[(i, j)],
                    rtol=1e-5
                ), f"Roundtrip failed for component ({i},{j})"

    def test_three_plus_one_decomposer_minkowski(self, minkowski_metric_small):
        """Test decomposing Minkowski metric"""
        alpha, beta_down, gamma_down, beta_up, gamma_up = \
            three_plus_one_decomposer(minkowski_metric_small)

        # Check results match expected Minkowski values
        assert np.allclose(alpha, 1.0), "Lapse should be 1"

        for i in range(3):
            assert np.allclose(beta_down[i], 0.0), f"Shift_down[{i}] should be 0"
            assert np.allclose(beta_up[i], 0.0), f"Shift_up[{i}] should be 0"


class TestAlcubierreMetric:
    """Test Alcubierre warp drive metric"""

    def test_alcubierre_shape_function(self):
        """Test Alcubierre shape function"""
        r = np.linspace(0, 10, 100)
        R = 2.0
        sigma = 5.0

        f = shape_function_alcubierre(r, R, sigma)

        # Shape function should be between 0 and 1
        assert np.all(f >= 0), "Shape function should be non-negative"
        assert np.all(f <= 1), "Shape function should be <= 1"

        # At r=0 (center), shape function should be near 1
        f_center = shape_function_alcubierre(0, R, sigma)
        assert f_center > 0.9, "Shape function at center should be near 1"

    def test_alcubierre_creation(
        self, medium_grid_size, world_center, alcubierre_params, grid_scale
    ):
        """Test Alcubierre metric creation"""
        metric = get_alcubierre_metric(
            medium_grid_size,
            world_center,
            alcubierre_params['velocity'],
            alcubierre_params['radius'],
            alcubierre_params['sigma'],
            grid_scale
        )

        assert metric.name == "Alcubierre", "Name should be 'Alcubierre'"
        assert metric.type == "metric", "Type should be 'metric'"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Alcubierre metric should verify"

    def test_alcubierre_params_stored(
        self, medium_grid_size, world_center, alcubierre_params, grid_scale
    ):
        """Test that Alcubierre parameters are stored"""
        metric = get_alcubierre_metric(
            medium_grid_size,
            world_center,
            alcubierre_params['velocity'],
            alcubierre_params['radius'],
            alcubierre_params['sigma'],
            grid_scale
        )

        assert "velocity" in metric.params, "Should store velocity"
        assert "R" in metric.params, "Should store radius"
        assert "sigma" in metric.params, "Should store sigma"

    def test_alcubierre_reduces_to_minkowski_far_away(
        self, alcubierre_params, grid_scale
    ):
        """Test that Alcubierre reduces to Minkowski far from bubble"""
        # Create small grid with bubble at center
        grid_size = [5, 11, 11, 11]
        world_center = [2.5, 5.0, 5.0, 5.0]

        metric = get_alcubierre_metric(
            grid_size,
            world_center,
            alcubierre_params['velocity'],
            alcubierre_params['radius'],
            alcubierre_params['sigma'],
            grid_scale
        )

        # Check corner point (far from center)
        # Should be approximately Minkowski
        t, x, y, z = 0, 0, 0, 0

        # Time component should be close to -1
        assert metric[(0, 0)][t, x, y, z] < -0.5, \
            "Far from bubble, g_00 should be close to -1"

        # Spatial diagonal should be close to 1
        assert metric[(2, 2)][t, x, y, z] > 0.5, \
            "Far from bubble, g_22 should be close to 1"

    def test_alcubierre_different_velocities(
        self, world_center, alcubierre_params, grid_scale
    ):
        """Test Alcubierre with different velocities"""
        grid_size = [5, 10, 10, 10]

        for velocity in [0.5, 1.0, 2.0]:
            metric = get_alcubierre_metric(
                grid_size,
                world_center,
                velocity,
                alcubierre_params['radius'],
                alcubierre_params['sigma'],
                grid_scale
            )
            assert verify_tensor(metric, suppress_msgs=True), \
                f"Alcubierre metric should verify for velocity={velocity}"


class TestSchwarzschildMetric:
    """Test Schwarzschild (black hole) metric"""

    def test_schwarzschild_creation(self):
        """Test Schwarzschild metric creation"""
        grid_size = [1, 10, 10, 10]  # Time must be 1
        world_center = [0.5, 5.0, 5.0, 5.0]
        rs = 2.0  # Schwarzschild radius

        metric = get_schwarzschild_metric(grid_size, world_center, rs)

        assert metric.name == "Schwarzschild", "Name should be 'Schwarzschild'"
        assert metric.type == "metric", "Type should be 'metric'"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Schwarzschild metric should verify"

    def test_schwarzschild_time_grid_restriction(self):
        """Test that Schwarzschild requires time grid = 1"""
        grid_size = [5, 10, 10, 10]  # Time > 1, should fail
        world_center = [2.5, 5.0, 5.0, 5.0]
        rs = 2.0

        with pytest.raises(ValueError, match="time grid is greater than 1"):
            get_schwarzschild_metric(grid_size, world_center, rs)

    def test_schwarzschild_params_stored(self):
        """Test that Schwarzschild parameters are stored"""
        grid_size = [1, 10, 10, 10]
        world_center = [0.5, 5.0, 5.0, 5.0]
        rs = 2.0

        metric = get_schwarzschild_metric(grid_size, world_center, rs)

        assert "rs" in metric.params, "Should store Schwarzschild radius"
        assert metric.params["rs"] == rs, "Should store correct rs value"

    def test_schwarzschild_signature(self):
        """Test Schwarzschild metric signature"""
        grid_size = [1, 10, 10, 10]
        world_center = [0.5, 5.0, 5.0, 5.0]
        rs = 1.0

        metric = get_schwarzschild_metric(grid_size, world_center, rs)

        # Far from event horizon, should approach Minkowski
        # At a corner point
        t, x, y, z = 0, 0, 0, 0
        g00 = metric[(0, 0)][t, x, y, z]

        # Should be negative (timelike)
        assert g00 < 0, "g_00 should be negative"

    def test_schwarzschild_symmetry(self):
        """Test Schwarzschild spherical symmetry"""
        grid_size = [1, 10, 10, 10]
        world_center = [0.5, 5.0, 5.0, 5.0]
        rs = 1.0

        metric = get_schwarzschild_metric(grid_size, world_center, rs)

        # Off-diagonal time-space terms should be zero (no rotation)
        t = 0
        for x in [0, 5, 9]:
            for y in [0, 5, 9]:
                for z in [0, 5, 9]:
                    assert np.isclose(metric[(0, 1)][t, x, y, z], 0.0), \
                        "g_0x should be zero"
                    assert np.isclose(metric[(0, 2)][t, x, y, z], 0.0), \
                        "g_0y should be zero"
                    assert np.isclose(metric[(0, 3)][t, x, y, z], 0.0), \
                        "g_0z should be zero"

    def test_schwarzschild_different_radii(self):
        """Test Schwarzschild with different Schwarzschild radii"""
        grid_size = [1, 10, 10, 10]
        world_center = [0.5, 5.0, 5.0, 5.0]

        for rs in [0.5, 1.0, 2.0, 5.0]:
            metric = get_schwarzschild_metric(grid_size, world_center, rs)
            assert verify_tensor(metric, suppress_msgs=True), \
                f"Schwarzschild metric should verify for rs={rs}"

    def test_schwarzschild_reduces_to_minkowski(self):
        """Test that Schwarzschild reduces to Minkowski as rs -> 0"""
        grid_size = [1, 10, 10, 10]
        world_center = [0.5, 5.0, 5.0, 5.0]
        rs_small = 0.01  # Very small rs

        metric = get_schwarzschild_metric(grid_size, world_center, rs_small)
        minkowski = get_minkowski_metric(grid_size)

        # Far from center, should be close to Minkowski
        t, x, y, z = 0, 0, 0, 0

        # Should be close to Minkowski values
        assert abs(metric[(0, 0)][t, x, y, z] - minkowski[(0, 0)][t, x, y, z]) < 0.1, \
            "Small rs should be close to Minkowski"


class TestMetricProperties:
    """Test general metric properties"""

    def test_metric_determinants_nonzero(
        self, minkowski_metric_small, tolerance
    ):
        """Test that metric determinants are non-zero"""
        det = c_det(minkowski_metric_small.tensor)
        assert np.all(np.abs(det) > tolerance), \
            "Metric determinant should be non-zero everywhere"

    def test_metric_symmetry(self, minkowski_metric_small):
        """Test that metrics are symmetric (g_ij = g_ji)"""
        for i in range(4):
            for j in range(4):
                assert np.allclose(
                    minkowski_metric_small[(i, j)],
                    minkowski_metric_small[(j, i)]
                ), f"Metric should be symmetric: g_{i}{j} = g_{j}{i}"

    def test_metric_scaling_effect(self):
        """Test effect of grid scaling on metrics"""
        grid_size = [5, 5, 5, 5]
        scale1 = [1.0, 1.0, 1.0, 1.0]
        scale2 = [2.0, 2.0, 2.0, 2.0]

        metric1 = get_minkowski_metric(grid_size, scale1)
        metric2 = get_minkowski_metric(grid_size, scale2)

        # Metrics should be identical for Minkowski (flat space)
        for i in range(4):
            for j in range(4):
                assert np.allclose(metric1[(i, j)], metric2[(i, j)]), \
                    "Minkowski should be independent of scaling"


class TestMetricEdgeCases:
    """Test edge cases for metrics"""

    def test_single_point_metric(self):
        """Test metric on single point grid"""
        grid_size = [1, 1, 1, 1]
        metric = get_minkowski_metric(grid_size)

        assert metric.shape == (1, 1, 1, 1), "Should handle single point"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Single point metric should verify"

    def test_small_grid_metrics(self):
        """Test metrics on small grids"""
        grid_size = [3, 3, 3, 3]
        world_center = [1.5, 1.5, 1.5, 1.5]

        # Minkowski
        minkowski = get_minkowski_metric(grid_size)
        assert verify_tensor(minkowski, suppress_msgs=True)

        # Alcubierre
        alcubierre = get_alcubierre_metric(
            grid_size, world_center, 1.0, 1.0, 5.0
        )
        assert verify_tensor(alcubierre, suppress_msgs=True)

        # Schwarzschild
        schwarzschild = get_schwarzschild_metric(
            [1, 3, 3, 3], [0.5, 1.5, 1.5, 1.5], 0.5
        )
        assert verify_tensor(schwarzschild, suppress_msgs=True)

    def test_metric_with_zero_velocity(self, world_center, grid_scale):
        """Test Alcubierre with zero velocity (should be like Minkowski)"""
        grid_size = [5, 10, 10, 10]

        metric = get_alcubierre_metric(
            grid_size,
            world_center,
            velocity=0.0,  # Zero velocity
            radius=2.0,
            sigma=5.0,
            grid_scale=grid_scale
        )

        # Should verify
        assert verify_tensor(metric, suppress_msgs=True), \
            "Zero velocity Alcubierre should verify"

        # Should be close to Minkowski (beta=0)
        minkowski = get_minkowski_metric(grid_size, grid_scale)

        # Check time-space components are zero (like Minkowski)
        for i in range(1, 4):
            assert np.allclose(metric[(0, i)], 0.0, atol=1e-10), \
                "Zero velocity should have zero shift"

    def test_shape_function_edge_cases(self):
        """Test Alcubierre shape function edge cases"""
        R = 2.0
        sigma = 5.0

        # At r = 0
        f_zero = shape_function_alcubierre(0.0, R, sigma)
        assert 0 <= f_zero <= 1, "Shape function at r=0 should be in [0,1]"

        # At very large r
        r_large = np.array([1000.0])
        f_large = shape_function_alcubierre(r_large, R, sigma)
        assert np.all(f_large < 0.1), "Shape function should decay far from bubble"

        # Test with array input
        r_array = np.array([0, 1, 2, 3, 4, 5])
        f_array = shape_function_alcubierre(r_array, R, sigma)
        assert len(f_array) == len(r_array), "Should handle array input"
        assert np.all((f_array >= 0) & (f_array <= 1)), \
            "All values should be in [0,1]"

    def test_schwarzschild_at_origin(self):
        """Test Schwarzschild behavior near origin (with epsilon)"""
        grid_size = [1, 5, 5, 5]
        world_center = [0.5, 2.5, 2.5, 2.5]  # Center of grid
        rs = 1.0

        metric = get_schwarzschild_metric(grid_size, world_center, rs)

        # At grid center point (closest to actual origin)
        t, x, y, z = 0, 2, 2, 2

        # Should have valid (finite) values due to epsilon
        assert np.isfinite(metric[(0, 0)][t, x, y, z]), \
            "Metric should be finite at origin"
        assert np.isfinite(metric[(1, 1)][t, x, y, z]), \
            "Metric should be finite at origin"


class TestLentzMetric:
    """Test Lentz warp drive metric"""

    def test_lentz_creation(self):
        """Test Lentz metric creation with default parameters"""
        from warpfactory.metrics.lentz.lentz import get_lentz_metric

        grid_size = [5, 21, 21, 11]
        world_center = [2.5, 10.5, 10.5, 5.5]
        velocity = 0.5
        scale = 3.0

        metric = get_lentz_metric(grid_size, world_center, velocity, scale)

        assert metric.name == "Lentz", "Name should be 'Lentz'"
        assert metric.type == "metric", "Type should be 'metric'"
        assert metric.index == "covariant", "Index should be 'covariant'"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Lentz metric should verify"

    def test_lentz_params_stored(self):
        """Test that Lentz parameters are stored"""
        from warpfactory.metrics.lentz.lentz import get_lentz_metric

        grid_size = [5, 21, 21, 11]
        world_center = [2.5, 10.5, 10.5, 5.5]
        velocity = 0.5
        scale = 3.0

        metric = get_lentz_metric(grid_size, world_center, velocity, scale)

        assert "velocity" in metric.params, "Should store velocity"
        assert "scale" in metric.params, "Should store scale"
        assert metric.params["velocity"] == velocity, "Should store correct velocity"
        assert metric.params["scale"] == scale, "Should store correct scale"

    def test_lentz_warp_factors(self):
        """Test that Lentz warp factors are correctly identified"""
        from warpfactory.metrics.lentz.lentz import get_warp_factor_by_region

        scale = 1.0

        # Region 1: Right triangle region (x >= scale, x <= 2*scale, x-scale >= y)
        WFX, WFY = get_warp_factor_by_region(1.5, 0.0, scale)
        assert WFX == -2.0, "Region 1 should have WFX = -2.0"
        assert WFY == 0.0, "Region 1 should have WFY = 0.0"

        # Region 2: Upper right diagonal
        WFX, WFY = get_warp_factor_by_region(1.5, 1.0, scale)
        assert WFX == -1.0, "Region 2 should have WFX = -1.0"
        assert WFY == 1.0, "Region 2 should have WFY = 1.0"

        # Region 3: Center upper vertical (x > 0, x <= scale, x+scale > y, -y+scale < x)
        # For x=0.3, y=0.2: x+scale > y → 1.3 > 0.2 ✓, -y+scale < x → 0.8 < 0.3 ✗
        # Actually Region 7 is triggered for most points in center
        # Let's test Region 7 instead
        WFX, WFY = get_warp_factor_by_region(0.5, 0.0, scale)
        assert WFX == 1.0, "Region 7 should have WFX = 1.0"
        assert WFY == 0.0, "Region 7 should have WFY = 0.0"

        # Test another clear case in Region 7
        WFX, WFY = get_warp_factor_by_region(0.0, 0.0, scale)
        assert WFX == 1.0, "Region 7 center should have WFX = 1.0"
        assert WFY == 0.0, "Region 7 center should have WFY = 0.0"

    def test_lentz_shift_vector_components(self):
        """Test Lentz shift vector has both x and y components"""
        from warpfactory.metrics.lentz.lentz import get_lentz_metric

        # Use centered configuration so bubble is at origin at t=0
        grid_size = [1, 21, 21, 11]  # Single time slice
        world_center = [0.0, 10.5, 10.5, 5.5]  # t=0 so bubble is centered
        velocity = 0.5

        metric = get_lentz_metric(grid_size, world_center, velocity)

        # Check g_01 and g_02 components directly (shift vector components)
        g_tx = metric[(0, 1)]
        g_ty = metric[(0, 2)]

        # Shift should have non-zero x and y components in some regions
        # Check that at least some points are non-zero
        has_nonzero_x_shift = np.any(np.abs(g_tx) > 1e-10)
        has_nonzero_y_shift = np.any(np.abs(g_ty) > 1e-10)

        assert has_nonzero_x_shift, \
            "Lentz should have non-zero shift in x direction in some regions"
        assert has_nonzero_y_shift, \
            "Lentz should have non-zero shift in y direction in some regions"

    def test_lentz_symmetry(self):
        """Test Lentz metric symmetry properties"""
        from warpfactory.metrics.lentz.lentz import get_lentz_metric

        grid_size = [3, 15, 15, 9]
        world_center = [1.5, 7.5, 7.5, 4.5]
        velocity = 0.3

        metric = get_lentz_metric(grid_size, world_center, velocity)

        # Metric should be symmetric
        for i in range(4):
            for j in range(4):
                assert np.allclose(metric[(i, j)], metric[(j, i)], atol=1e-10), \
                    f"Metric should be symmetric: g_{i}{j} = g_{j}{i}"

    def test_lentz_default_scale(self):
        """Test Lentz metric with default scale parameter"""
        from warpfactory.metrics.lentz.lentz import get_lentz_metric

        grid_size = [5, 21, 21, 11]
        world_center = [2.5, 10.5, 10.5, 5.5]
        velocity = 0.5

        # Should use default scale = max(grid_size[1:4]) / 7
        metric = get_lentz_metric(grid_size, world_center, velocity)

        expected_scale = max(grid_size[1:4]) / 7.0
        assert metric.params["scale"] == expected_scale, \
            "Default scale should be max(grid_size[1:4]) / 7"

    def test_lentz_edge_cases(self):
        """Test Lentz metric edge cases and boundary regions"""
        from warpfactory.metrics.lentz.lentz import get_warp_factor_by_region

        scale = 1.0

        # Test points at boundaries between regions
        # At y = 0 (center horizontal line)
        WFX, WFY = get_warp_factor_by_region(0.0, 0.0, scale)
        assert WFY == 0.0, "On horizontal centerline, WFY should be 0"

        # Test symmetry in y
        WFX_pos, WFY_pos = get_warp_factor_by_region(0.5, 0.5, scale)
        WFX_neg, WFY_neg = get_warp_factor_by_region(0.5, -0.5, scale)
        assert WFX_pos == WFX_neg, "WFX should be symmetric in y"
        assert WFY_pos == -WFY_neg, "WFY should be antisymmetric in y"


class TestVanDenBroeckMetric:
    """Test Van Den Broeck warp drive metric"""

    def test_van_den_broeck_creation(self):
        """Test Van Den Broeck metric creation"""
        from warpfactory.metrics.van_den_broeck.van_den_broeck import \
            get_van_den_broeck_metric

        grid_size = [5, 10, 10, 10]
        world_center = [2.5, 5.0, 5.0, 5.0]
        v = 0.5
        R1 = 2.0
        sigma1 = 5.0
        R2 = 2.0
        sigma2 = 5.0
        A = 2.0

        metric = get_van_den_broeck_metric(
            grid_size, world_center, v, R1, sigma1, R2, sigma2, A
        )

        assert metric.name == "Van Den Broeck", \
            "Name should be 'Van Den Broeck'"
        assert metric.type == "metric", "Type should be 'metric'"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Van Den Broeck metric should verify"

    def test_van_den_broeck_params_stored(self):
        """Test that Van Den Broeck parameters are stored"""
        from warpfactory.metrics.van_den_broeck.van_den_broeck import \
            get_van_den_broeck_metric

        grid_size = [5, 10, 10, 10]
        world_center = [2.5, 5.0, 5.0, 5.0]
        v = 0.5
        R1 = 2.0
        sigma1 = 5.0
        R2 = 2.0
        sigma2 = 5.0
        A = 2.0

        metric = get_van_den_broeck_metric(
            grid_size, world_center, v, R1, sigma1, R2, sigma2, A
        )

        assert "R1" in metric.params, "Should store R1"
        assert "R2" in metric.params, "Should store R2"
        assert "A" in metric.params, "Should store A"
        assert metric.params["A"] == A, "Should store correct A value"

    def test_van_den_broeck_expansion_factor(self):
        """Test spatial expansion factor B(r)"""
        from warpfactory.metrics.van_den_broeck.van_den_broeck import \
            get_van_den_broeck_metric

        grid_size = [1, 9, 9, 9]
        world_center = [0.0, 4.5, 4.5, 4.5]  # Centered at t=0
        v = 0.5
        R1 = 2.0
        sigma1 = 5.0
        R2 = 2.0
        sigma2 = 5.0
        A = 2.0

        metric = get_van_den_broeck_metric(
            grid_size, world_center, v, R1, sigma1, R2, sigma2, A
        )

        # At grid center (bubble center at t=0), B should be > 1
        t, x, y, z = 0, 4, 4, 4  # Center point
        g_xx = metric[(1, 1)][t, x, y, z]

        # g_xx = B^2, so B = sqrt(g_xx)
        B_center = np.sqrt(g_xx)

        # At center, B should be influenced by expansion factor A
        # B = 1 + f*A where f is shape function (near 1 at center)
        assert B_center >= 1.0, "Expansion factor should be >= 1"
        # For small radius and high sigma, shape function is near 1 at center
        assert B_center <= 1 + A + 1.0, \
            f"Expansion factor at center should be reasonable, got {B_center}"

    def test_van_den_broeck_effective_velocity(self):
        """Test that effective velocity is v_eff = v(1+A)^2"""
        from warpfactory.metrics.van_den_broeck.van_den_broeck import \
            get_van_den_broeck_metric

        grid_size = [5, 10, 10, 10]
        world_center = [2.5, 5.0, 5.0, 5.0]
        v = 0.5
        R1 = 2.0
        sigma1 = 5.0
        R2 = 2.0
        sigma2 = 5.0
        A = 2.0

        metric = get_van_den_broeck_metric(
            grid_size, world_center, v, R1, sigma1, R2, sigma2, A
        )

        # Effective velocity should be stored
        expected_velocity = v * (1 + A)**2
        assert "velocity" in metric.params, "Should store velocity"
        assert np.isclose(metric.params["velocity"], expected_velocity), \
            f"Velocity should be v(1+A)^2 = {expected_velocity}"

    def test_van_den_broeck_metric_signature(self):
        """Test Van Den Broeck metric signature"""
        from warpfactory.metrics.van_den_broeck.van_den_broeck import \
            get_van_den_broeck_metric

        grid_size = [3, 9, 9, 9]
        world_center = [1.5, 4.5, 4.5, 4.5]
        v = 0.1  # Small velocity
        R1 = 2.0
        sigma1 = 5.0
        R2 = 2.0
        sigma2 = 5.0
        A = 1.0

        metric = get_van_den_broeck_metric(
            grid_size, world_center, v, R1, sigma1, R2, sigma2, A
        )

        # Time component should be negative
        assert np.all(metric[(0, 0)] < 0), "g_00 should be negative (timelike)"

        # Spatial components should be positive
        assert np.all(metric[(1, 1)] > 0), "g_11 should be positive"
        assert np.all(metric[(2, 2)] > 0), "g_22 should be positive"
        assert np.all(metric[(3, 3)] > 0), "g_33 should be positive"

    def test_van_den_broeck_different_expansions(self):
        """Test Van Den Broeck with different expansion factors"""
        from warpfactory.metrics.van_den_broeck.van_den_broeck import \
            get_van_den_broeck_metric

        grid_size = [3, 9, 9, 9]
        world_center = [1.5, 4.5, 4.5, 4.5]
        v = 0.5
        R1 = 2.0
        sigma1 = 5.0
        R2 = 2.0
        sigma2 = 5.0

        for A in [0.5, 1.0, 2.0, 5.0]:
            metric = get_van_den_broeck_metric(
                grid_size, world_center, v, R1, sigma1, R2, sigma2, A
            )
            assert verify_tensor(metric, suppress_msgs=True), \
                f"Van Den Broeck metric should verify for A={A}"

    def test_van_den_broeck_shift_function(self):
        """Test shift function in Van Den Broeck metric"""
        from warpfactory.metrics.van_den_broeck.van_den_broeck import \
            get_van_den_broeck_metric

        grid_size = [1, 9, 9, 9]
        world_center = [0.0, 4.5, 4.5, 4.5]  # Centered at t=0
        v = 0.5
        R1 = 2.0
        sigma1 = 5.0
        R2 = 2.0
        sigma2 = 5.0
        A = 2.0

        metric = get_van_den_broeck_metric(
            grid_size, world_center, v, R1, sigma1, R2, sigma2, A
        )

        # Shift vector creates g_tx component
        # Check that we have non-zero shift somewhere in the grid
        g_tx = metric[(0, 1)]

        has_nonzero_shift = np.any(np.abs(g_tx) > 1e-10)
        assert has_nonzero_shift, "Shift vector should create non-zero g_tx in the bubble"


class TestModifiedTimeMetric:
    """Test Modified Time warp drive metric"""

    def test_modified_time_creation(self):
        """Test Modified Time metric creation"""
        from warpfactory.metrics.modified_time.modified_time import \
            get_modified_time_metric

        grid_size = [5, 10, 10, 10]
        world_center = [2.5, 5.0, 5.0, 5.0]
        velocity = 1.0
        radius = 2.0
        sigma = 5.0
        A = 2.0

        metric = get_modified_time_metric(
            grid_size, world_center, velocity, radius, sigma, A
        )

        assert metric.name == "Modified Time", \
            "Name should be 'Modified Time'"
        assert metric.type == "metric", "Type should be 'metric'"
        assert verify_tensor(metric, suppress_msgs=True), \
            "Modified Time metric should verify"

    def test_modified_time_params_stored(self):
        """Test that Modified Time parameters are stored"""
        from warpfactory.metrics.modified_time.modified_time import \
            get_modified_time_metric

        grid_size = [5, 10, 10, 10]
        world_center = [2.5, 5.0, 5.0, 5.0]
        velocity = 1.0
        radius = 2.0
        sigma = 5.0
        A = 2.0

        metric = get_modified_time_metric(
            grid_size, world_center, velocity, radius, sigma, A
        )

        assert "velocity" in metric.params, "Should store velocity"
        assert "R" in metric.params, "Should store radius"
        assert "sigma" in metric.params, "Should store sigma"
        assert "A" in metric.params, "Should store A"
        assert metric.params["A"] == A, "Should store correct A value"

    def test_modified_time_lapse_modification(self):
        """Test lapse function modification"""
        from warpfactory.metrics.modified_time.modified_time import \
            get_modified_time_metric
        from warpfactory.metrics.three_plus_one import three_plus_one_decomposer

        grid_size = [1, 9, 9, 9]
        world_center = [0.0, 4.5, 4.5, 4.5]  # Centered at t=0
        velocity = 0.5
        radius = 2.0
        sigma = 5.0
        A = 2.0

        metric = get_modified_time_metric(
            grid_size, world_center, velocity, radius, sigma, A
        )

        # Decompose to get lapse function
        alpha, beta_down, gamma_down, beta_up, gamma_up = \
            three_plus_one_decomposer(metric)

        # Lapse should be modified: alpha = (1-fs) + fs/A
        # At bubble center at t=0, fs should be significant
        # Check that lapse varies across grid (not all 1.0)
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)

        assert alpha_max > 0.0, "Lapse should be positive"
        # If A > 1, lapse should be < 1 somewhere in bubble
        if A > 1.0:
            assert alpha_min < 1.0 or alpha_max > 1.0, \
                "Lapse should vary from Minkowski value when modified"

    def test_modified_time_three_plus_one_properties(self):
        """Test 3+1 decomposition properties"""
        from warpfactory.metrics.modified_time.modified_time import \
            get_modified_time_metric
        from warpfactory.metrics.three_plus_one import three_plus_one_decomposer

        grid_size = [5, 10, 10, 10]
        world_center = [2.5, 5.0, 5.0, 5.0]
        velocity = 1.0
        radius = 2.0
        sigma = 5.0
        A = 2.0

        metric = get_modified_time_metric(
            grid_size, world_center, velocity, radius, sigma, A
        )

        # Should be able to decompose
        alpha, beta_down, gamma_down, beta_up, gamma_up = \
            three_plus_one_decomposer(metric)

        # All components should be finite
        assert np.all(np.isfinite(alpha)), "Lapse should be finite"
        for i in range(3):
            assert np.all(np.isfinite(beta_down[i])), \
                f"Shift[{i}] should be finite"

    def test_modified_time_metric_symmetry(self):
        """Test metric symmetry"""
        from warpfactory.metrics.modified_time.modified_time import \
            get_modified_time_metric

        grid_size = [3, 9, 9, 9]
        world_center = [1.5, 4.5, 4.5, 4.5]
        velocity = 0.5
        radius = 2.0
        sigma = 5.0
        A = 2.0

        metric = get_modified_time_metric(
            grid_size, world_center, velocity, radius, sigma, A
        )

        # Metric should be symmetric
        for i in range(4):
            for j in range(4):
                assert np.allclose(metric[(i, j)], metric[(j, i)], atol=1e-10), \
                    f"Metric should be symmetric: g_{i}{j} = g_{j}{i}"

    def test_modified_time_different_lapse_rates(self):
        """Test Modified Time with different lapse rate parameters"""
        from warpfactory.metrics.modified_time.modified_time import \
            get_modified_time_metric

        grid_size = [3, 9, 9, 9]
        world_center = [1.5, 4.5, 4.5, 4.5]
        velocity = 0.5
        radius = 2.0
        sigma = 5.0

        for A in [0.5, 1.0, 2.0, 5.0]:
            metric = get_modified_time_metric(
                grid_size, world_center, velocity, radius, sigma, A
            )
            assert verify_tensor(metric, suppress_msgs=True), \
                f"Modified Time metric should verify for A={A}"

    def test_modified_time_vs_alcubierre(self):
        """Test that Modified Time differs from Alcubierre"""
        from warpfactory.metrics.modified_time.modified_time import \
            get_modified_time_metric
        from warpfactory.metrics.alcubierre import get_alcubierre_metric

        grid_size = [1, 9, 9, 9]
        world_center = [0.0, 4.5, 4.5, 4.5]  # Centered at t=0
        velocity = 0.5
        radius = 2.0
        sigma = 5.0
        A = 2.0

        modified = get_modified_time_metric(
            grid_size, world_center, velocity, radius, sigma, A
        )
        alcubierre = get_alcubierre_metric(
            grid_size, world_center, velocity, radius, sigma
        )

        # g_00 should differ somewhere in the grid due to lapse modification
        # Check if they differ anywhere (not necessarily at grid center)
        g00_modified = modified[(0, 0)]
        g00_alcubierre = alcubierre[(0, 0)]

        # They should differ somewhere if A != 1
        max_diff = np.max(np.abs(g00_modified - g00_alcubierre))
        assert max_diff > 1e-10 or A == 1.0, \
            f"Modified Time g_00 should differ from Alcubierre when A={A}, max diff={max_diff}"
