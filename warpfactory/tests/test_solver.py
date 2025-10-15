"""
Unit tests for warpfactory.solver module

Tests finite differences, Christoffel symbols, Ricci tensor, and Einstein tensor.
"""

import pytest
import numpy as np
from warpfactory.solver.finite_differences import (
    take_finite_difference_1,
    take_finite_difference_2
)
from warpfactory.solver.christoffel import get_christoffel_symbols
from warpfactory.solver.ricci import calculate_ricci_tensor, calculate_ricci_scalar
from warpfactory.solver.einstein import calculate_einstein_tensor
from warpfactory.core.tensor_ops import c4_inv


class TestFiniteDifferences:
    """Test finite difference calculations"""

    def test_first_derivative_constant(self, small_grid_size, grid_scale):
        """Test first derivative of constant function is zero"""
        A = np.ones(small_grid_size)

        for k in range(4):
            dA = take_finite_difference_1(A, k, grid_scale)
            assert np.allclose(dA, 0, atol=1e-10), \
                f"Derivative of constant in direction {k} should be zero"

    def test_first_derivative_linear(self):
        """Test first derivative of linear function"""
        grid_size = [10, 10, 10, 10]
        delta = [1.0, 1.0, 1.0, 1.0]

        # Create linear function in x direction: f(x) = x
        A = np.zeros(grid_size)
        for i in range(grid_size[1]):
            A[:, i, :, :] = i

        # Derivative in x direction should be 1
        dA_dx = take_finite_difference_1(A, 1, delta)

        # Check interior points (not boundaries)
        assert np.allclose(dA_dx[:, 3:-3, :, :], 1.0, atol=0.1), \
            "Derivative of linear function should be constant"

    def test_first_derivative_quadratic(self):
        """Test first derivative of quadratic function"""
        grid_size = [10, 10, 10, 10]
        delta = [1.0, 1.0, 1.0, 1.0]

        # Create quadratic function: f(x) = x^2
        A = np.zeros(grid_size)
        for i in range(grid_size[1]):
            A[:, i, :, :] = i**2

        # Derivative should be 2x
        dA_dx = take_finite_difference_1(A, 1, delta)

        # Check a specific interior point
        x_val = 5
        expected_derivative = 2 * x_val

        # Check with some tolerance for numerical errors
        assert abs(dA_dx[0, x_val, 0, 0] - expected_derivative) < 0.5, \
            "Derivative of quadratic should match 2x"

    def test_second_derivative_constant(self, small_grid_size, grid_scale):
        """Test second derivative of constant is zero"""
        A = np.ones(small_grid_size) * 5.0

        for k1 in range(4):
            for k2 in range(4):
                d2A = take_finite_difference_2(A, k1, k2, grid_scale)
                assert np.allclose(d2A, 0, atol=1e-10), \
                    f"Second derivative of constant in directions {k1},{k2} should be zero"

    def test_second_derivative_linear(self):
        """Test second derivative of linear function is zero"""
        grid_size = [10, 10, 10, 10]
        delta = [1.0, 1.0, 1.0, 1.0]

        # Create linear function
        A = np.zeros(grid_size)
        for i in range(grid_size[1]):
            A[:, i, :, :] = i * 2.0

        # Second derivative should be zero
        d2A = take_finite_difference_2(A, 1, 1, delta)

        # Check interior points
        assert np.allclose(d2A[:, 3:-3, :, :], 0, atol=0.1), \
            "Second derivative of linear function should be zero"

    def test_second_derivative_quadratic(self):
        """Test second derivative of quadratic function"""
        grid_size = [10, 10, 10, 10]
        delta = [1.0, 1.0, 1.0, 1.0]

        # Create quadratic function: f(x) = x^2
        A = np.zeros(grid_size)
        for i in range(grid_size[1]):
            A[:, i, :, :] = i**2

        # Second derivative should be 2
        d2A = take_finite_difference_2(A, 1, 1, delta)

        # Check interior points
        assert np.allclose(d2A[:, 3:-3, :, :], 2.0, atol=0.5), \
            "Second derivative of x^2 should be 2"

    def test_mixed_derivative(self):
        """Test mixed partial derivatives"""
        grid_size = [10, 10, 10, 10]
        delta = [1.0, 1.0, 1.0, 1.0]

        # Create function f(x,y) = x*y
        A = np.zeros(grid_size)
        for i in range(grid_size[1]):
            for j in range(grid_size[2]):
                A[:, i, j, :] = i * j

        # Mixed derivative d²f/dxdy should be 1
        d2A_dxdy = take_finite_difference_2(A, 1, 2, delta)

        # Check interior points
        assert np.allclose(d2A_dxdy[:, 3:-3, 3:-3, :], 1.0, atol=0.5), \
            "Mixed derivative of x*y should be 1"

    def test_derivative_different_directions(self, grid_scale):
        """Test derivatives in different coordinate directions"""
        grid_size = [10, 10, 10, 10]

        for k in range(4):
            # Create function that varies in direction k
            A = np.zeros(grid_size)
            if k == 0:
                for t in range(grid_size[0]):
                    A[t, :, :, :] = t
            elif k == 1:
                for x in range(grid_size[1]):
                    A[:, x, :, :] = x
            elif k == 2:
                for y in range(grid_size[2]):
                    A[:, :, y, :] = y
            else:  # k == 3
                for z in range(grid_size[3]):
                    A[:, :, :, z] = z

            # Derivative in direction k should be non-zero
            dA = take_finite_difference_1(A, k, grid_scale)
            assert not np.allclose(dA, 0), \
                f"Derivative in direction {k} should be non-zero"

    def test_derivative_small_grid(self):
        """Test derivatives on minimum size grid"""
        grid_size = [5, 5, 5, 5]  # Minimum for 4th order
        delta = [1.0, 1.0, 1.0, 1.0]

        A = np.random.randn(*grid_size)

        # Should compute without errors
        for k in range(4):
            dA = take_finite_difference_1(A, k, delta)
            assert dA.shape == tuple(grid_size), \
                "Output shape should match input"

    def test_derivative_too_small_grid(self):
        """Test derivatives on grid too small for 4th order"""
        grid_size = [3, 3, 3, 3]  # Too small
        delta = [1.0, 1.0, 1.0, 1.0]

        A = np.ones(grid_size)

        # Should return zeros (not enough points for stencil)
        dA = take_finite_difference_1(A, 0, delta)
        assert np.allclose(dA, 0), \
            "Should return zeros for grid too small"


class TestChristoffelSymbols:
    """Test Christoffel symbol calculations"""

    def test_christoffel_minkowski_zero(self, minkowski_metric_small, grid_scale):
        """Test that Christoffel symbols vanish for Minkowski metric"""
        # Get contravariant metric
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        # Calculate first derivatives
        diff_1_gl = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    diff_1_gl[(i, j, k)] = take_finite_difference_1(
                        gl[(i, j)], k, grid_scale
                    )

        # Christoffel symbols should be zero for flat space
        for i in range(4):
            for k in range(4):
                for l in range(4):
                    Gamma = get_christoffel_symbols(gu, diff_1_gl, i, k, l)
                    assert np.allclose(Gamma, 0, atol=1e-10), \
                        f"Christoffel Γ^{i}_{k}{l} should be zero for Minkowski"

    def test_christoffel_symmetry(self, grid_scale):
        """Test Christoffel symbol symmetry in lower indices"""
        # Create a simple non-flat metric
        grid_size = [5, 5, 5, 5]
        gl = {}

        # Diagonal metric with spatial variation
        for i in range(4):
            for j in range(4):
                if i == j:
                    if i == 0:
                        gl[(i, j)] = -np.ones(grid_size)
                    else:
                        # Add slight variation
                        gl[(i, j)] = np.ones(grid_size)
                        for x in range(grid_size[1]):
                            gl[(i, j)][:, x, :, :] *= (1 + 0.1 * x / grid_size[1])
                else:
                    gl[(i, j)] = np.zeros(grid_size)

        gu = c4_inv(gl)

        # Calculate derivatives
        diff_1_gl = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    diff_1_gl[(i, j, k)] = take_finite_difference_1(
                        gl[(i, j)], k, grid_scale
                    )

        # Check symmetry: Γ^i_kl = Γ^i_lk
        for i in range(4):
            for k in range(4):
                for l in range(k, 4):
                    Gamma_kl = get_christoffel_symbols(gu, diff_1_gl, i, k, l)
                    Gamma_lk = get_christoffel_symbols(gu, diff_1_gl, i, l, k)

                    assert np.allclose(Gamma_kl, Gamma_lk, rtol=1e-5), \
                        f"Christoffel symbols should be symmetric: Γ^{i}_{k}{l} = Γ^{i}_{l}{k}"

    def test_christoffel_shape(self, minkowski_metric_small, grid_scale):
        """Test that Christoffel symbols have correct shape"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        diff_1_gl = {}
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    diff_1_gl[(i, j, k)] = take_finite_difference_1(
                        gl[(i, j)], k, grid_scale
                    )

        Gamma = get_christoffel_symbols(gu, diff_1_gl, 0, 0, 0)
        assert Gamma.shape == minkowski_metric_small.shape, \
            "Christoffel symbol should have same shape as metric"


class TestRicciTensor:
    """Test Ricci tensor calculations"""

    def test_ricci_minkowski_zero(self, minkowski_metric_small, grid_scale):
        """Test that Ricci tensor vanishes for Minkowski metric"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)

        # Ricci tensor should be zero for flat space
        for i in range(4):
            for j in range(4):
                assert np.allclose(R_munu[(i, j)], 0, atol=1e-6), \
                    f"Ricci tensor R_{i}{j} should be zero for Minkowski"

    def test_ricci_symmetry(self, minkowski_metric_small, grid_scale):
        """Test Ricci tensor symmetry"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)

        # Ricci tensor should be symmetric
        for i in range(4):
            for j in range(4):
                assert np.allclose(R_munu[(i, j)], R_munu[(j, i)], rtol=1e-5), \
                    f"Ricci tensor should be symmetric: R_{i}{j} = R_{j}{i}"

    def test_ricci_shape(self, minkowski_metric_small, grid_scale):
        """Test Ricci tensor has correct shape"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)

        # Should have 16 components
        assert len(R_munu) == 16, "Ricci tensor should have 16 components"

        # Each component should have correct shape
        for i in range(4):
            for j in range(4):
                assert R_munu[(i, j)].shape == minkowski_metric_small.shape, \
                    f"R_{i}{j} should have same shape as metric"


class TestRicciScalar:
    """Test Ricci scalar calculations"""

    def test_ricci_scalar_minkowski_zero(self, minkowski_metric_small, grid_scale):
        """Test that Ricci scalar vanishes for Minkowski metric"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
        R = calculate_ricci_scalar(R_munu, gu)

        assert np.allclose(R, 0, atol=1e-6), \
            "Ricci scalar should be zero for Minkowski"

    def test_ricci_scalar_shape(self, minkowski_metric_small, grid_scale):
        """Test Ricci scalar has correct shape"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
        R = calculate_ricci_scalar(R_munu, gu)

        assert R.shape == minkowski_metric_small.shape, \
            "Ricci scalar should have same shape as metric"

    def test_ricci_scalar_is_scalar(self, minkowski_metric_small, grid_scale):
        """Test that Ricci scalar is a true scalar (single value per point)"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
        R = calculate_ricci_scalar(R_munu, gu)

        # Should be a single array, not a dictionary
        assert isinstance(R, np.ndarray), \
            "Ricci scalar should be a numpy array"


class TestEinsteinTensor:
    """Test Einstein tensor calculations"""

    def test_einstein_minkowski_zero(self, minkowski_metric_small, grid_scale):
        """Test that Einstein tensor vanishes for Minkowski metric"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
        R = calculate_ricci_scalar(R_munu, gu)
        G = calculate_einstein_tensor(R_munu, R, gl)

        # Einstein tensor should be zero for flat space (vacuum)
        for i in range(4):
            for j in range(4):
                assert np.allclose(G[(i, j)], 0, atol=1e-6), \
                    f"Einstein tensor G_{i}{j} should be zero for Minkowski"

    def test_einstein_symmetry(self, minkowski_metric_small, grid_scale):
        """Test Einstein tensor symmetry"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
        R = calculate_ricci_scalar(R_munu, gu)
        G = calculate_einstein_tensor(R_munu, R, gl)

        # Einstein tensor should be symmetric
        for i in range(4):
            for j in range(4):
                assert np.allclose(G[(i, j)], G[(j, i)], rtol=1e-5), \
                    f"Einstein tensor should be symmetric: G_{i}{j} = G_{j}{i}"

    def test_einstein_shape(self, minkowski_metric_small, grid_scale):
        """Test Einstein tensor has correct shape"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
        R = calculate_ricci_scalar(R_munu, gu)
        G = calculate_einstein_tensor(R_munu, R, gl)

        # Should have 16 components
        assert len(G) == 16, "Einstein tensor should have 16 components"

        # Each component should have correct shape
        for i in range(4):
            for j in range(4):
                assert G[(i, j)].shape == minkowski_metric_small.shape, \
                    f"G_{i}{j} should have same shape as metric"

    def test_einstein_formula(self):
        """Test Einstein tensor formula: G_μν = R_μν - (1/2) g_μν R"""
        grid_size = [5, 5, 5, 5]

        # Create simple tensor components
        R_munu = {(i, j): np.random.randn(*grid_size) * 0.1 for i in range(4) for j in range(4)}
        R = np.random.randn(*grid_size) * 0.1
        gl = {(i, j): np.eye(4)[i, j] * np.ones(grid_size) for i in range(4) for j in range(4)}
        gl[(0, 0)] = -np.ones(grid_size)  # Minkowski signature

        G = calculate_einstein_tensor(R_munu, R, gl)

        # Verify formula manually for one component
        i, j = 0, 0
        expected = R_munu[(i, j)] - 0.5 * gl[(i, j)] * R

        assert np.allclose(G[(i, j)], expected), \
            "Einstein tensor should match formula G_μν = R_μν - (1/2) g_μν R"


class TestSolverEdgeCases:
    """Test edge cases for solver functions"""

    def test_derivatives_with_zeros(self, grid_scale):
        """Test derivatives of zero array"""
        grid_size = [5, 5, 5, 5]
        A = np.zeros(grid_size)

        for k in range(4):
            dA = take_finite_difference_1(A, k, grid_scale)
            assert np.allclose(dA, 0), \
                "Derivative of zero should be zero"

    def test_derivatives_different_scales(self):
        """Test derivatives with different grid scales"""
        grid_size = [10, 10, 10, 10]

        # Create function f(x) = x
        A = np.zeros(grid_size)
        for i in range(grid_size[1]):
            A[:, i, :, :] = i

        # Test with different scales
        for scale in [0.5, 1.0, 2.0]:
            delta = [scale, scale, scale, scale]
            dA = take_finite_difference_1(A, 1, delta)

            # Derivative should scale with grid spacing
            # df/dx where f=x should give ~1/scale (because x is in grid units)
            assert dA.shape == tuple(grid_size), \
                "Should compute for any positive scale"

    def test_solver_chain_consistency(self, minkowski_metric_small, grid_scale):
        """Test consistency of solver chain: metric -> Ricci -> Einstein"""
        gl = minkowski_metric_small.tensor
        gu = c4_inv(gl)

        # Full chain
        R_munu = calculate_ricci_tensor(gu, gl, grid_scale)
        R = calculate_ricci_scalar(R_munu, gu)
        G = calculate_einstein_tensor(R_munu, R, gl)

        # All should be zero for Minkowski
        R_max = np.max(np.abs(R))
        G_max = max(np.max(np.abs(G[(i, j)])) for i in range(4) for j in range(4))

        assert R_max < 1e-5, "Ricci scalar should be near zero"
        assert G_max < 1e-5, "Einstein tensor should be near zero"
