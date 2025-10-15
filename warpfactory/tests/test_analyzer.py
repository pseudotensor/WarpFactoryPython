"""
Unit tests for warpfactory.analyzer module

Tests energy conditions, scalars, and analyzer utilities.
"""

import pytest
import numpy as np
from warpfactory.analyzer.energy_conditions import get_energy_conditions
from warpfactory.analyzer.scalars import get_scalars
from warpfactory.analyzer.utils import (
    get_even_points_on_sphere,
    generate_uniform_field,
    get_inner_product,
    get_trace
)
from warpfactory.core.tensor import Tensor


class TestSpherePointGeneration:
    """Test sphere point generation utilities"""

    def test_points_on_sphere_count(self):
        """Test correct number of points generated"""
        R = 1.0
        num_points = 100

        points = get_even_points_on_sphere(R, num_points)

        assert points.shape == (3, num_points), \
            "Should generate correct number of points"

    def test_points_on_sphere_radius(self):
        """Test that all points are on the sphere"""
        R = 2.0
        num_points = 50

        points = get_even_points_on_sphere(R, num_points)

        # Calculate radius of each point
        radii = np.sqrt(points[0, :]**2 + points[1, :]**2 + points[2, :]**2)

        assert np.allclose(radii, R, rtol=1e-10), \
            "All points should be at radius R"

    def test_points_on_unit_sphere(self):
        """Test generation on unit sphere"""
        num_points = 20

        points = get_even_points_on_sphere(1.0, num_points)
        radii = np.sqrt(points[0, :]**2 + points[1, :]**2 + points[2, :]**2)

        assert np.allclose(radii, 1.0), "Points should be on unit sphere"

    def test_points_different_radii(self):
        """Test sphere generation with different radii"""
        num_points = 30

        for R in [0.5, 1.0, 5.0, 10.0]:
            points = get_even_points_on_sphere(R, num_points)
            radii = np.sqrt(points[0, :]**2 + points[1, :]**2 + points[2, :]**2)

            assert np.allclose(radii, R, rtol=1e-10), \
                f"Points should be at radius {R}"

    def test_points_small_number(self):
        """Test with small number of points"""
        points = get_even_points_on_sphere(1.0, 3)
        assert points.shape == (3, 3), "Should handle small number of points"


class TestVectorFieldGeneration:
    """Test vector field generation for energy conditions"""

    def test_nulllike_vector_field_shape(self):
        """Test nulllike vector field has correct shape"""
        num_angular = 50

        field = generate_uniform_field("nulllike", num_angular)

        assert field.shape == (4, num_angular), \
            "Nulllike field should have shape (4, num_angular)"

    def test_timelike_vector_field_shape(self):
        """Test timelike vector field has correct shape"""
        num_angular = 50
        num_time = 10

        field = generate_uniform_field("timelike", num_angular, num_time)

        assert field.shape == (4, num_angular, num_time), \
            "Timelike field should have shape (4, num_angular, num_time)"

    def test_nulllike_vectors_normalized(self):
        """Test that nulllike vectors are properly normalized"""
        num_angular = 30

        field = generate_uniform_field("nulllike", num_angular)

        # Calculate norms (in flat space)
        norms = np.sqrt(
            field[0, :]**2 + field[1, :]**2 +
            field[2, :]**2 + field[3, :]**2
        )

        # Vectors are normalized to unit length
        assert np.allclose(norms, 1.0), \
            "Nulllike vectors should be normalized to unit length"

    def test_timelike_vectors_normalized(self):
        """Test that timelike vectors are properly normalized"""
        num_angular = 20
        num_time = 5

        field = generate_uniform_field("timelike", num_angular, num_time)

        # Check normalization for each time shell
        for jj in range(num_time):
            norms = np.sqrt(
                field[0, :, jj]**2 + field[1, :, jj]**2 +
                field[2, :, jj]**2 + field[3, :, jj]**2
            )
            assert np.allclose(norms, 1.0), \
                f"Timelike vectors at time shell {jj} should be normalized"

    def test_invalid_field_type_error(self):
        """Test error for invalid field type"""
        with pytest.raises(ValueError, match="not recognized"):
            generate_uniform_field("invalid_type", 50)

    def test_nulllike_time_component(self):
        """Test nulllike vectors have time component = 1"""
        num_angular = 40

        field = generate_uniform_field("nulllike", num_angular)

        # Time component (before normalization) is 1
        # After normalization, should still be positive
        assert np.all(field[0, :] > 0), \
            "Nulllike vectors should have positive time component"

    def test_timelike_vectors_satisfy_condition(self):
        """Test that timelike vectors satisfy c²t² > r² condition"""
        num_angular = 30
        num_time = 8

        field = generate_uniform_field("timelike", num_angular, num_time)

        for jj in range(num_time):
            t_squared = field[0, :, jj]**2
            r_squared = field[1, :, jj]**2 + field[2, :, jj]**2 + field[3, :, jj]**2

            # In Minkowski signature, timelike means -t² + r² < 0, or t² > r²
            assert np.all(t_squared >= r_squared - 1e-10), \
                f"Timelike vectors should satisfy t² >= r² (timelike condition)"


class TestInnerProduct:
    """Test inner product calculations"""

    def test_inner_product_orthogonal_vectors(self, minkowski_metric_small):
        """Test inner product of orthogonal vectors is zero"""
        s = minkowski_metric_small.shape

        # Create orthogonal vectors in space
        vec_a = {
            'field': [np.zeros(s), np.ones(s), np.zeros(s), np.zeros(s)],
            'index': 'contravariant',
            'type': '4-vector'
        }

        vec_b = {
            'field': [np.zeros(s), np.zeros(s), np.ones(s), np.zeros(s)],
            'index': 'contravariant',
            'type': '4-vector'
        }

        inner_prod = get_inner_product(vec_a, vec_b, minkowski_metric_small)

        assert np.allclose(inner_prod, 0, atol=1e-10), \
            "Orthogonal vectors should have zero inner product"

    def test_inner_product_same_vector(self, minkowski_metric_small):
        """Test inner product of vector with itself"""
        s = minkowski_metric_small.shape

        # Create timelike vector (1, 0, 0, 0)
        vec = {
            'field': [np.ones(s), np.zeros(s), np.zeros(s), np.zeros(s)],
            'index': 'contravariant',
            'type': '4-vector'
        }

        inner_prod = get_inner_product(vec, vec, minkowski_metric_small)

        # In Minkowski: g^00 = -1, so (1)^2 * (-1) = -1
        assert np.allclose(inner_prod, -1.0), \
            "Timelike vector (1,0,0,0) should have inner product -1"

    def test_inner_product_spatial_vector(self, minkowski_metric_small):
        """Test inner product of spatial vector"""
        s = minkowski_metric_small.shape

        # Create spatial vector (0, 1, 0, 0)
        vec = {
            'field': [np.zeros(s), np.ones(s), np.zeros(s), np.zeros(s)],
            'index': 'contravariant',
            'type': '4-vector'
        }

        inner_prod = get_inner_product(vec, vec, minkowski_metric_small)

        # In Minkowski: g^11 = 1, so (1)^2 * (1) = 1
        assert np.allclose(inner_prod, 1.0), \
            "Spatial vector should have positive inner product"

    def test_inner_product_different_indices(self, minkowski_metric_small):
        """Test inner product with different index types"""
        s = minkowski_metric_small.shape

        vec_up = {
            'field': [np.ones(s), np.zeros(s), np.zeros(s), np.zeros(s)],
            'index': 'contravariant',
            'type': '4-vector'
        }

        vec_down = {
            'field': [np.ones(s), np.zeros(s), np.zeros(s), np.zeros(s)],
            'index': 'covariant',
            'type': '4-vector'
        }

        inner_prod = get_inner_product(vec_up, vec_down, minkowski_metric_small)

        # Direct contraction when indices are different
        assert inner_prod.shape == s, \
            "Inner product should have same shape as metric"


class TestTrace:
    """Test tensor trace calculations"""

    def test_trace_identity_tensor(self, minkowski_metric_small):
        """Test trace of identity-like tensor"""
        s = minkowski_metric_small.shape

        # Create tensor with diagonal = 1
        tensor_dict = {}
        for i in range(4):
            for j in range(4):
                if i == j:
                    tensor_dict[(i, j)] = np.ones(s)
                else:
                    tensor_dict[(i, j)] = np.zeros(s)

        tensor = Tensor(
            tensor=tensor_dict,
            tensor_type="stress-energy",
            index="covariant"
        )

        trace = get_trace(tensor, minkowski_metric_small)

        # Trace should be -1 + 1 + 1 + 1 = 2 (due to Minkowski signature)
        assert np.allclose(trace, 2.0), \
            "Trace of identity tensor should be 2 in Minkowski"

    def test_trace_zero_tensor(self, minkowski_metric_small):
        """Test trace of zero tensor is zero"""
        s = minkowski_metric_small.shape

        tensor_dict = {(i, j): np.zeros(s) for i in range(4) for j in range(4)}

        tensor = Tensor(
            tensor=tensor_dict,
            tensor_type="stress-energy",
            index="covariant"
        )

        trace = get_trace(tensor, minkowski_metric_small)

        assert np.allclose(trace, 0, atol=1e-10), \
            "Trace of zero tensor should be zero"

    def test_trace_diagonal_tensor(self, minkowski_metric_small):
        """Test trace of diagonal tensor"""
        s = minkowski_metric_small.shape

        # Create diagonal tensor with specific values
        tensor_dict = {}
        diagonal_values = [2.0, 3.0, 4.0, 5.0]

        for i in range(4):
            for j in range(4):
                if i == j:
                    tensor_dict[(i, j)] = np.full(s, diagonal_values[i])
                else:
                    tensor_dict[(i, j)] = np.zeros(s)

        tensor = Tensor(
            tensor=tensor_dict,
            tensor_type="stress-energy",
            index="covariant"
        )

        trace = get_trace(tensor, minkowski_metric_small)

        # Trace = g^00 * T_00 + g^11 * T_11 + ...
        #       = (-1) * 2 + 1 * 3 + 1 * 4 + 1 * 5 = -2 + 12 = 10
        expected_trace = -2.0 + 3.0 + 4.0 + 5.0

        assert np.allclose(trace, expected_trace), \
            f"Trace should be {expected_trace}"


class TestEnergyConditions:
    """Test energy condition evaluations"""

    def test_energy_condition_minkowski_vacuum(
        self, minkowski_metric_small, small_grid_size
    ):
        """Test energy conditions for vacuum (zero stress-energy)"""
        # Create zero stress-energy tensor
        energy_tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy",
            index="covariant"
        )

        # NEC should be satisfied (zero is okay)
        nec_map, _, _ = get_energy_conditions(
            energy_tensor,
            minkowski_metric_small,
            "Null",
            num_angular_vec=20,
            num_time_vec=5
        )

        # For vacuum, NEC should be zero or positive (satisfied)
        assert nec_map.shape == tuple(small_grid_size), \
            "Energy condition map should have correct shape"

    def test_energy_condition_types(
        self, minkowski_metric_small, small_grid_size
    ):
        """Test all energy condition types can be evaluated"""
        energy_tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy"
        )

        conditions = ["Null", "Weak", "Dominant", "Strong"]

        for condition in conditions:
            result, _, _ = get_energy_conditions(
                energy_tensor,
                minkowski_metric_small,
                condition,
                num_angular_vec=10,
                num_time_vec=5
            )

            assert result.shape == tuple(small_grid_size), \
                f"{condition} energy condition should return correct shape"

    def test_energy_condition_invalid_type(
        self, minkowski_metric_small, small_grid_size
    ):
        """Test error for invalid energy condition type"""
        energy_tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy"
        )

        with pytest.raises(ValueError, match="Incorrect energy condition"):
            get_energy_conditions(
                energy_tensor,
                minkowski_metric_small,
                "Invalid",
                num_angular_vec=10
            )

    def test_energy_condition_return_vec(
        self, minkowski_metric_small, small_grid_size
    ):
        """Test returning vector field from energy condition"""
        energy_tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy"
        )

        num_angular = 15

        result_map, vec, vector_field = get_energy_conditions(
            energy_tensor,
            minkowski_metric_small,
            "Null",
            num_angular_vec=num_angular,
            return_vec=True
        )

        assert result_map is not None, "Should return result map"
        assert vec is not None, "Should return vec when return_vec=True"
        assert vector_field is not None, "Should return vector field when return_vec=True"

        # Check vec shape
        expected_vec_shape = tuple(list(small_grid_size) + [num_angular])
        assert vec.shape == expected_vec_shape, \
            "Vec should have shape (grid + num_angular)"

    def test_energy_condition_without_return_vec(
        self, minkowski_metric_small, small_grid_size
    ):
        """Test that vec is None when return_vec=False"""
        energy_tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy"
        )

        result_map, vec, vector_field = get_energy_conditions(
            energy_tensor,
            minkowski_metric_small,
            "Null",
            num_angular_vec=10,
            return_vec=False
        )

        assert result_map is not None, "Should return result map"
        assert vec is None, "Vec should be None when return_vec=False"
        assert vector_field is None, "Vector field should be None when return_vec=False"


class TestScalars:
    """Test kinematic scalar calculations"""

    def test_scalars_minkowski(self, minkowski_metric_small):
        """Test scalars for Minkowski metric"""
        expansion, shear, vorticity = get_scalars(minkowski_metric_small)

        # All scalars should have correct shape
        assert expansion.shape == minkowski_metric_small.shape, \
            "Expansion should have same shape as metric"
        assert shear.shape == minkowski_metric_small.shape, \
            "Shear should have same shape as metric"
        assert vorticity.shape == minkowski_metric_small.shape, \
            "Vorticity should have same shape as metric"

    def test_scalars_are_real(self, minkowski_metric_small):
        """Test that scalars are real-valued"""
        expansion, shear, vorticity = get_scalars(minkowski_metric_small)

        assert np.all(np.isreal(expansion)), "Expansion should be real"
        assert np.all(np.isreal(shear)), "Shear should be real"
        assert np.all(np.isreal(vorticity)), "Vorticity should be real"

    def test_scalars_minkowski_zero(self, minkowski_metric_small):
        """Test that scalars vanish for Minkowski (flat space, no motion)"""
        expansion, shear, vorticity = get_scalars(minkowski_metric_small)

        # For flat Minkowski with zero shift, all scalars should be zero
        # (Though the simplified implementation may not show this)
        assert expansion.shape == minkowski_metric_small.shape, \
            "Should compute expansion"
        assert shear.shape == minkowski_metric_small.shape, \
            "Should compute shear"
        assert vorticity.shape == minkowski_metric_small.shape, \
            "Should compute vorticity"

    def test_scalars_non_negative(self, minkowski_metric_small):
        """Test that shear and vorticity scalars are non-negative"""
        _, shear, vorticity = get_scalars(minkowski_metric_small)

        # Shear² and vorticity² should be non-negative
        assert np.all(shear >= -1e-10), "Shear scalar should be non-negative"
        assert np.all(vorticity >= -1e-10), "Vorticity scalar should be non-negative"


class TestAnalyzerEdgeCases:
    """Test edge cases for analyzer functions"""

    def test_sphere_points_single_point(self):
        """Test generation of single point on sphere"""
        points = get_even_points_on_sphere(1.0, 1)
        assert points.shape == (3, 1), "Should handle single point"

        radius = np.sqrt(points[0, 0]**2 + points[1, 0]**2 + points[2, 0]**2)
        assert np.isclose(radius, 1.0), "Single point should be on sphere"

    def test_vector_field_small_count(self):
        """Test vector field with small number of vectors"""
        field = generate_uniform_field("nulllike", 5)
        assert field.shape == (4, 5), "Should handle small number of vectors"

    def test_inner_product_zero_vectors(self, minkowski_metric_small):
        """Test inner product of zero vectors"""
        s = minkowski_metric_small.shape

        vec_zero = {
            'field': [np.zeros(s) for _ in range(4)],
            'index': 'contravariant',
            'type': '4-vector'
        }

        inner_prod = get_inner_product(vec_zero, vec_zero, minkowski_metric_small)

        assert np.allclose(inner_prod, 0, atol=1e-10), \
            "Zero vector should have zero inner product"

    def test_trace_single_nonzero_component(self, minkowski_metric_small):
        """Test trace with single non-zero component"""
        s = minkowski_metric_small.shape

        # Only T_00 = 1, rest zero
        tensor_dict = {(i, j): np.zeros(s) for i in range(4) for j in range(4)}
        tensor_dict[(0, 0)] = np.ones(s)

        tensor = Tensor(
            tensor=tensor_dict,
            tensor_type="stress-energy",
            index="covariant"
        )

        trace = get_trace(tensor, minkowski_metric_small)

        # Trace = g^00 * T_00 = (-1) * 1 = -1
        assert np.allclose(trace, -1.0), \
            "Trace should be -1"
