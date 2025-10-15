"""
Unit tests for warpfactory.core module

Tests the Tensor class and tensor operations.
"""

import pytest
import numpy as np
from warpfactory.core.tensor import Tensor
from warpfactory.core.tensor_ops import (
    c3_inv, c4_inv, c_det, verify_tensor, change_tensor_index,
    get_array_module
)


class TestTensorClass:
    """Test the Tensor class"""

    def test_tensor_initialization(self, small_grid_size):
        """Test basic tensor initialization"""
        tensor_dict = {}
        for i in range(4):
            for j in range(4):
                tensor_dict[(i, j)] = np.zeros(small_grid_size)

        tensor = Tensor(
            tensor=tensor_dict,
            tensor_type="metric",
            name="Test",
            index="covariant"
        )

        assert tensor.type == "metric", "Tensor type should be 'metric'"
        assert tensor.name == "Test", "Tensor name should be 'Test'"
        assert tensor.index == "covariant", "Tensor index should be 'covariant'"
        assert tensor.coords == "cartesian", "Default coords should be 'cartesian'"

    def test_tensor_shape(self, minkowski_metric_small, small_grid_size):
        """Test tensor shape property"""
        shape = minkowski_metric_small.shape
        assert shape == tuple(small_grid_size), "Tensor shape should match grid size"

    def test_tensor_getitem(self, minkowski_metric_small):
        """Test tensor component access"""
        g00 = minkowski_metric_small[(0, 0)]
        assert isinstance(g00, np.ndarray), "Tensor component should be numpy array"
        assert np.all(g00 == -1), "Minkowski g_00 should be -1"

    def test_tensor_setitem(self, zero_tensor_small):
        """Test tensor component assignment"""
        test_value = np.ones(zero_tensor_small.shape)
        zero_tensor_small[(1, 1)] = test_value
        assert np.all(zero_tensor_small[(1, 1)] == 1), "Set component should equal test value"

    def test_tensor_copy(self, minkowski_metric_small):
        """Test tensor deep copy"""
        tensor_copy = minkowski_metric_small.copy()
        assert tensor_copy.name == minkowski_metric_small.name, "Copy should have same name"

        # Modify copy and ensure original unchanged
        tensor_copy[(0, 0)] = np.ones(tensor_copy.shape)
        assert not np.array_equal(
            tensor_copy[(0, 0)],
            minkowski_metric_small[(0, 0)]
        ), "Modifying copy should not affect original"

    def test_tensor_zeros(self, small_grid_size):
        """Test Tensor.zeros class method"""
        tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="metric",
            name="Zero",
            index="covariant"
        )

        assert tensor.shape == tuple(small_grid_size), "Zero tensor should have correct shape"
        for i in range(4):
            for j in range(4):
                assert np.all(tensor[(i, j)] == 0), f"Component ({i},{j}) should be zero"

    def test_tensor_repr(self, minkowski_metric_small):
        """Test tensor string representation"""
        repr_str = repr(minkowski_metric_small)
        assert "Tensor" in repr_str, "Repr should contain 'Tensor'"
        assert "metric" in repr_str, "Repr should contain tensor type"
        assert "Minkowski" in repr_str, "Repr should contain tensor name"

    def test_tensor_scaling(self, grid_scale):
        """Test tensor with custom scaling"""
        custom_scale = [2.0, 3.0, 4.0, 5.0]
        tensor = Tensor.zeros(
            [5, 5, 5, 5],
            tensor_type="metric",
            scaling=custom_scale
        )
        assert tensor.scaling == custom_scale, "Tensor should store custom scaling"

    def test_tensor_params(self):
        """Test tensor with custom parameters"""
        params = {"velocity": 1.0, "radius": 2.0}
        tensor = Tensor.zeros(
            [5, 5, 5, 5],
            tensor_type="metric",
            params=params
        )
        assert tensor.params["velocity"] == 1.0, "Tensor should store velocity param"
        assert tensor.params["radius"] == 2.0, "Tensor should store radius param"

    def test_tensor_date(self, minkowski_metric_small):
        """Test tensor date attribute"""
        assert hasattr(minkowski_metric_small, 'date'), "Tensor should have date attribute"
        assert isinstance(minkowski_metric_small.date, str), "Date should be a string"


class TestTensorTypes:
    """Test different tensor types"""

    def test_metric_tensor_type(self, minkowski_metric_small):
        """Test metric tensor type"""
        assert minkowski_metric_small.type == "metric", "Should be metric tensor"

    def test_stress_energy_tensor_type(self, small_grid_size):
        """Test stress-energy tensor type"""
        tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy",
            name="Energy"
        )
        assert tensor.type == "stress-energy", "Should be stress-energy tensor"

    def test_tensor_index_types(self, small_grid_size):
        """Test different index types"""
        for index_type in ["covariant", "contravariant", "mixedupdown", "mixeddownup"]:
            tensor = Tensor.zeros(
                small_grid_size,
                tensor_type="metric",
                index=index_type
            )
            assert tensor.index == index_type, f"Should have {index_type} index"


class TestTensor3x3Inversion:
    """Test 3x3 tensor inversion"""

    def test_identity_3x3_inversion(self, identity_3x3):
        """Test inversion of 3x3 identity"""
        inv = c3_inv(identity_3x3)

        # Check all components
        for i in range(3):
            for j in range(3):
                if i == j:
                    assert np.allclose(inv[(i, j)], 1.0), f"Diagonal ({i},{j}) should be 1"
                else:
                    assert np.allclose(inv[(i, j)], 0.0), f"Off-diagonal ({i},{j}) should be 0"

    def test_3x3_inversion_invertibility(self):
        """Test that (A^-1)^-1 = A for 3x3"""
        # Create a random 3x3 cell array
        original = {}
        for i in range(3):
            for j in range(3):
                original[(i, j)] = np.random.randn(2, 2, 2, 2) * 0.1
                if i == j:
                    original[(i, j)] += 1.0  # Make it diagonally dominant

        inv = c3_inv(original)
        inv_inv = c3_inv(inv)

        # Check that double inversion returns original
        for i in range(3):
            for j in range(3):
                assert np.allclose(
                    inv_inv[(i, j)],
                    original[(i, j)],
                    rtol=1e-5
                ), f"Double inversion failed for ({i},{j})"

    def test_3x3_wrong_size_error(self):
        """Test that c3_inv raises error for non-3x3"""
        wrong_size = {(0, 0): np.array([1.0])}
        with pytest.raises(ValueError, match="not 3x3"):
            c3_inv(wrong_size)


class TestTensor4x4Operations:
    """Test 4x4 tensor operations"""

    def test_4x4_determinant_identity(self, identity_4x4):
        """Test determinant of 4x4 identity"""
        det = c_det(identity_4x4)
        assert np.allclose(det, 1.0), "Determinant of identity should be 1"

    def test_4x4_determinant_minkowski(self, minkowski_metric_small):
        """Test determinant of Minkowski metric"""
        det = c_det(minkowski_metric_small.tensor)
        # Minkowski has signature (-,+,+,+) so det = -1
        assert np.allclose(det, -1.0), "Determinant of Minkowski should be -1"

    def test_4x4_inversion_identity(self, identity_4x4):
        """Test inversion of 4x4 identity"""
        inv = c4_inv(identity_4x4)

        for i in range(4):
            for j in range(4):
                if i == j:
                    assert np.allclose(inv[(i, j)], 1.0), f"Diagonal ({i},{j}) should be 1"
                else:
                    assert np.allclose(inv[(i, j)], 0.0), f"Off-diagonal ({i},{j}) should be 0"

    def test_4x4_inversion_minkowski(self, minkowski_metric_small):
        """Test inversion of Minkowski metric"""
        inv = c4_inv(minkowski_metric_small.tensor)

        # Inverse of Minkowski covariant is contravariant
        # g^00 = -1, g^ii = 1, off-diagonal = 0
        assert np.allclose(inv[(0, 0)], -1.0), "g^00 should be -1"
        assert np.allclose(inv[(1, 1)], 1.0), "g^11 should be 1"
        assert np.allclose(inv[(2, 2)], 1.0), "g^22 should be 1"
        assert np.allclose(inv[(3, 3)], 1.0), "g^33 should be 1"

        for i in range(4):
            for j in range(4):
                if i != j:
                    assert np.allclose(inv[(i, j)], 0.0), f"Off-diagonal ({i},{j}) should be 0"

    def test_4x4_inversion_invertibility(self, minkowski_metric_small):
        """Test that (A^-1)^-1 = A for 4x4"""
        original = minkowski_metric_small.tensor
        inv = c4_inv(original)
        inv_inv = c4_inv(inv)

        for i in range(4):
            for j in range(4):
                assert np.allclose(
                    inv_inv[(i, j)],
                    original[(i, j)],
                    rtol=1e-5
                ), f"Double inversion failed for ({i},{j})"

    def test_4x4_wrong_size_error(self):
        """Test that c4_inv raises error for non-4x4"""
        wrong_size = {(0, 0): np.array([1.0])}
        with pytest.raises(ValueError, match="not 4x4"):
            c4_inv(wrong_size)


class TestTensorVerification:
    """Test tensor verification"""

    def test_verify_valid_metric(self, minkowski_metric_small):
        """Test verification of valid metric tensor"""
        assert verify_tensor(minkowski_metric_small, suppress_msgs=True), \
            "Valid metric should verify"

    def test_verify_valid_stress_energy(self, small_grid_size):
        """Test verification of valid stress-energy tensor"""
        tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy"
        )
        assert verify_tensor(tensor, suppress_msgs=True), \
            "Valid stress-energy should verify"

    def test_verify_missing_type(self, small_grid_size):
        """Test verification fails for missing type"""
        tensor_dict = {(i, j): np.zeros(small_grid_size) for i in range(4) for j in range(4)}
        tensor = Tensor(tensor_dict, tensor_type="metric")
        delattr(tensor, 'type')

        assert not verify_tensor(tensor, suppress_msgs=True), \
            "Should fail without type attribute"

    def test_verify_wrong_type(self, small_grid_size):
        """Test verification fails for invalid type"""
        tensor_dict = {(i, j): np.zeros(small_grid_size) for i in range(4) for j in range(4)}
        tensor = Tensor(tensor_dict, tensor_type="invalid_type")

        assert not verify_tensor(tensor, suppress_msgs=True), \
            "Should fail with invalid type"

    def test_verify_missing_components(self, small_grid_size):
        """Test verification fails for missing components"""
        # Create incomplete tensor (missing some components)
        tensor_dict = {(0, 0): np.zeros(small_grid_size)}
        tensor = Tensor(tensor_dict, tensor_type="metric")

        assert not verify_tensor(tensor, suppress_msgs=True), \
            "Should fail with incomplete components"

    def test_verify_inconsistent_shapes(self, small_grid_size):
        """Test verification fails for inconsistent shapes"""
        tensor_dict = {}
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    tensor_dict[(i, j)] = np.zeros([3, 3, 3, 3])  # Wrong shape
                else:
                    tensor_dict[(i, j)] = np.zeros(small_grid_size)

        tensor = Tensor(tensor_dict, tensor_type="metric")

        assert not verify_tensor(tensor, suppress_msgs=True), \
            "Should fail with inconsistent shapes"

    def test_verify_non_4d_arrays(self, small_grid_size):
        """Test verification fails for non-4D arrays"""
        tensor_dict = {}
        for i in range(4):
            for j in range(4):
                tensor_dict[(i, j)] = np.zeros([5, 5, 5])  # 3D instead of 4D

        tensor = Tensor(tensor_dict, tensor_type="metric")

        assert not verify_tensor(tensor, suppress_msgs=True), \
            "Should fail with non-4D arrays"


class TestIndexTransformation:
    """Test tensor index transformations"""

    def test_metric_covariant_to_contravariant(self, minkowski_metric_small):
        """Test metric index raising"""
        contravariant = change_tensor_index(
            minkowski_metric_small,
            "contravariant"
        )

        assert contravariant.index == "contravariant", \
            "Output should have contravariant index"

        # For Minkowski, raising and lowering should give same (diagonal) result
        assert np.allclose(contravariant[(0, 0)], -1.0), "g^00 should be -1"
        assert np.allclose(contravariant[(1, 1)], 1.0), "g^11 should be 1"

    def test_metric_contravariant_to_covariant(self, minkowski_metric_small):
        """Test metric index lowering"""
        # First raise index
        contravariant = change_tensor_index(minkowski_metric_small, "contravariant")

        # Then lower it back
        covariant = change_tensor_index(contravariant, "covariant")

        assert covariant.index == "covariant", "Output should have covariant index"

        # Should get back original
        for i in range(4):
            for j in range(4):
                assert np.allclose(
                    covariant[(i, j)],
                    minkowski_metric_small[(i, j)]
                ), f"Round-trip failed for ({i},{j})"

    def test_stress_energy_index_change_requires_metric(self, small_grid_size):
        """Test that non-metric tensors require metric for index change"""
        tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy"
        )

        with pytest.raises(ValueError, match="metric_tensor is needed"):
            change_tensor_index(tensor, "contravariant")

    def test_stress_energy_covariant_to_contravariant(
        self, small_grid_size, minkowski_metric_small
    ):
        """Test stress-energy tensor index raising"""
        # Create a stress-energy tensor
        tensor = Tensor.zeros(
            small_grid_size,
            tensor_type="stress-energy",
            index="covariant"
        )
        tensor[(0, 0)] = np.ones(small_grid_size)  # Set some component

        contravariant = change_tensor_index(
            tensor,
            "contravariant",
            minkowski_metric_small
        )

        assert contravariant.index == "contravariant", \
            "Output should have contravariant index"

    def test_invalid_index_type_error(self, minkowski_metric_small):
        """Test error for invalid index type"""
        with pytest.raises(ValueError, match="not allowed"):
            change_tensor_index(minkowski_metric_small, "invalid_index")

    def test_metric_cannot_be_mixed(self, minkowski_metric_small):
        """Test that metric tensors cannot have mixed indices"""
        with pytest.raises(ValueError, match="mixed index"):
            change_tensor_index(minkowski_metric_small, "mixedupdown")


class TestArrayModule:
    """Test array module detection"""

    def test_get_array_module_numpy(self):
        """Test detection of numpy arrays"""
        arr = np.array([1, 2, 3])
        module = get_array_module(arr)
        assert module == np, "Should detect numpy module"

    def test_get_array_module_with_operations(self):
        """Test that detected module works for operations"""
        arr = np.array([1, 2, 3])
        xp = get_array_module(arr)

        result = xp.zeros(5)
        assert isinstance(result, np.ndarray), "Should create numpy array"
        assert len(result) == 5, "Should have correct length"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_empty_tensor_dict(self):
        """Test behavior with empty tensor dictionary"""
        # This should work but tensor won't verify
        tensor = Tensor(
            tensor={},
            tensor_type="metric"
        )
        assert tensor.type == "metric", "Should still have type"

    def test_single_point_grid(self):
        """Test tensor on 1x1x1x1 grid"""
        grid_size = [1, 1, 1, 1]
        tensor = Tensor.zeros(grid_size, tensor_type="metric")
        assert tensor.shape == (1, 1, 1, 1), "Should handle single point grid"

    def test_large_grid_creation(self):
        """Test that large grids can be created (without filling)"""
        large_grid = [20, 20, 20, 20]
        tensor = Tensor.zeros(large_grid, tensor_type="metric")
        assert tensor.shape == tuple(large_grid), "Should handle larger grids"

    def test_tensor_with_nan(self, small_grid_size):
        """Test tensor containing NaN values"""
        tensor = Tensor.zeros(small_grid_size, tensor_type="metric")
        tensor[(0, 0)] = np.full(small_grid_size, np.nan)

        assert np.all(np.isnan(tensor[(0, 0)])), "Should preserve NaN values"

    def test_tensor_with_inf(self, small_grid_size):
        """Test tensor containing infinity"""
        tensor = Tensor.zeros(small_grid_size, tensor_type="metric")
        tensor[(1, 1)] = np.full(small_grid_size, np.inf)

        assert np.all(np.isinf(tensor[(1, 1)])), "Should preserve infinity"
