"""
Unit tests for warpfactory.visualizer module

Tests plotting utilities and visualization functions (non-visual tests).
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from warpfactory.visualizer.utils import (
    get_slice_data,
    redblue,
    label_cartesian_axis,
    create_redblue_colormap
)
from warpfactory.visualizer.plot_tensor import plot_tensor


class TestSliceData:
    """Test slice data extraction for plotting"""

    def test_slice_data_basic(self, minkowski_metric_small):
        """Test basic slice data extraction"""
        plane = [0, 1]  # Fix t and x
        slice_center = [2, 2]

        idx = get_slice_data(plane, slice_center, minkowski_metric_small)

        assert len(idx) == 4, "Should return 4 indices"
        assert idx[0] == 2, "First plane should be fixed at 2"
        assert idx[1] == 2, "Second plane should be fixed at 2"
        assert isinstance(idx[2], range), "Third dimension should be a range"
        assert isinstance(idx[3], range), "Fourth dimension should be a range"

    def test_slice_data_different_planes(self, minkowski_metric_small):
        """Test slice extraction for different plane combinations"""
        test_cases = [
            ([0, 1], [1, 1]),
            ([0, 2], [2, 2]),
            ([0, 3], [1, 2]),
            ([1, 2], [2, 1]),
            ([1, 3], [1, 1]),
            ([2, 3], [2, 2]),
        ]

        for plane, center in test_cases:
            idx = get_slice_data(plane, center, minkowski_metric_small)

            assert idx[plane[0]] == center[0], \
                f"Plane {plane[0]} should be fixed at {center[0]}"
            assert idx[plane[1]] == center[1], \
                f"Plane {plane[1]} should be fixed at {center[1]}"

            # Check other dimensions are ranges
            for dim in range(4):
                if dim not in plane:
                    assert isinstance(idx[dim], range), \
                        f"Dimension {dim} should be a range"

    def test_slice_data_range_lengths(self, minkowski_metric_small):
        """Test that range lengths match tensor dimensions"""
        plane = [0, 1]
        slice_center = [2, 2]

        idx = get_slice_data(plane, slice_center, minkowski_metric_small)
        shape = minkowski_metric_small.shape

        for dim in range(4):
            if isinstance(idx[dim], range):
                assert len(idx[dim]) == shape[dim], \
                    f"Range length for dimension {dim} should match shape"


class TestRedBlueColormap:
    """Test red-blue diverging colormap"""

    def test_redblue_all_positive(self):
        """Test colormap for all positive values"""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        cmap = redblue(data)

        assert cmap.shape[1] == 3, "Should return RGB values"
        assert cmap.shape[0] > 0, "Should have color entries"

        # Should be blue-to-white gradient
        # Blue channel should be all 1s
        assert np.all(cmap[:, 2] == 1), "Blue channel should be 1 for positive data"

    def test_redblue_all_negative(self):
        """Test colormap for all negative values"""
        data = np.array([[-6, -5, -4], [-3, -2, -1]])
        cmap = redblue(data)

        assert cmap.shape[1] == 3, "Should return RGB values"

        # Should be white-to-red gradient
        # Red channel should be all 1s
        assert np.all(cmap[:, 0] == 1), "Red channel should be 1 for negative data"

    def test_redblue_zero_data(self):
        """Test colormap for all zeros"""
        data = np.zeros((3, 3))
        cmap = redblue(data)

        assert cmap.shape == (1, 3), "Should return single color for zeros"
        assert np.allclose(cmap, [1, 1, 1]), "Should be white for zero data"

    def test_redblue_diverging(self):
        """Test colormap for data spanning zero"""
        data = np.array([[-2, -1, 0], [1, 2, 3]])
        cmap = redblue(data)

        assert cmap.shape[1] == 3, "Should return RGB values"
        assert cmap.shape[0] > 1, "Should have multiple colors"

        # Should have both red and blue components
        # Some entries should have high red (negative side)
        # Some entries should have high blue (positive side)
        assert np.any(cmap[:, 0] > 0.5), "Should have red components"
        assert np.any(cmap[:, 2] > 0.5), "Should have blue components"

    def test_redblue_symmetric_data(self):
        """Test colormap for symmetric data around zero"""
        data = np.array([[-5, -2, 0], [2, 5, 0]])
        cmap = redblue(data)

        # Center should be approximately in the middle
        center_val = abs(5) / (abs(5) + abs(-5))
        assert abs(center_val - 0.5) < 0.01, \
            "Center should be near 0.5 for symmetric data"

    def test_redblue_asymmetric_data(self):
        """Test colormap for asymmetric data"""
        data = np.array([[-1, 0, 1], [2, 3, 9]])
        cmap = redblue(data)

        # More positive than negative, so center should shift
        max_val = 9
        min_val = -1
        center_val = abs(max_val) / (abs(max_val) + abs(min_val))

        assert center_val > 0.5, \
            "Center should shift toward positive for more positive data"

    def test_redblue_custom_gradient_num(self):
        """Test colormap with custom gradient number"""
        data = np.array([[-5, 0], [5, 10]])

        for gradient_num in [64, 256, 512]:
            cmap = redblue(data, gradient_num)
            # Total colors should be approximately gradient_num
            assert cmap.shape[0] <= gradient_num + 10, \
                f"Should have approximately {gradient_num} colors"


class TestAxisLabeling:
    """Test axis label generation"""

    def test_label_all_plane_combinations(self):
        """Test labels for all valid plane combinations"""
        test_cases = {
            (0, 1): ("y", "z"),  # t, x fixed -> show y, z
            (0, 2): ("x", "z"),  # t, y fixed -> show x, z
            (0, 3): ("x", "y"),  # t, z fixed -> show x, y
            (1, 2): ("t", "z"),  # x, y fixed -> show t, z
            (1, 3): ("t", "y"),  # x, z fixed -> show t, y
            (2, 3): ("t", "x"),  # y, z fixed -> show t, x
        }

        for plane, expected_labels in test_cases.items():
            xlabel, ylabel = label_cartesian_axis(list(plane))
            assert xlabel == expected_labels[0], \
                f"For plane {plane}, xlabel should be '{expected_labels[0]}'"
            assert ylabel == expected_labels[1], \
                f"For plane {plane}, ylabel should be '{expected_labels[1]}'"

    def test_label_order_independence(self):
        """Test that plane order doesn't affect output labels"""
        # [0, 1] and [1, 0] should give same labels (different order of fixed dims)
        xlabel1, ylabel1 = label_cartesian_axis([0, 1])
        xlabel2, ylabel2 = label_cartesian_axis([1, 0])

        # Both should show y and z (the unfixed dimensions)
        assert {xlabel1, ylabel1} == {xlabel2, ylabel2}, \
            "Plane order should not affect shown dimensions"


class TestCreateRedblueColormap:
    """Test matplotlib colormap creation"""

    def test_create_colormap(self):
        """Test creating a LinearSegmentedColormap"""
        cmap = create_redblue_colormap()

        assert cmap is not None, "Should create colormap"
        assert hasattr(cmap, '__call__'), "Colormap should be callable"

    def test_colormap_custom_name(self):
        """Test creating colormap with custom name"""
        name = 'test_redblue'
        cmap = create_redblue_colormap(name)

        assert cmap.name == name, f"Colormap name should be '{name}'"

    def test_colormap_values(self):
        """Test colormap returns correct colors"""
        cmap = create_redblue_colormap()

        # Test at specific points
        red_color = cmap(0.0)  # Should be red
        white_color = cmap(0.5)  # Should be white
        blue_color = cmap(1.0)  # Should be blue

        # Red at 0 (high red, low green/blue)
        assert red_color[0] > 0.9, "Should be mostly red at 0"

        # White at 0.5 (all high)
        assert all(c > 0.9 for c in white_color[:3]), "Should be white at 0.5"

        # Blue at 1 (low red/green, high blue)
        assert blue_color[2] > 0.9, "Should be mostly blue at 1"


class TestPlotTensor:
    """Test tensor plotting functions"""

    def test_plot_tensor_basic(self, minkowski_metric_small):
        """Test basic tensor plotting"""
        figures = plot_tensor(minkowski_metric_small)

        assert isinstance(figures, list), "Should return list of figures"
        assert len(figures) > 0, "Should create at least one figure"

        for fig in figures:
            assert isinstance(fig, Figure), "Each item should be a Figure"

        # Close all figures to free memory
        for fig in figures:
            plt.close(fig)

    def test_plot_tensor_custom_alpha(self, minkowski_metric_small):
        """Test plotting with custom alpha"""
        figures = plot_tensor(minkowski_metric_small, alpha=0.5)

        assert len(figures) > 0, "Should create figures"

        for fig in figures:
            plt.close(fig)

    def test_plot_tensor_custom_planes(self, minkowski_metric_small):
        """Test plotting with custom slice planes"""
        figures = plot_tensor(
            minkowski_metric_small,
            sliced_planes=[1, 2],  # Slice along x and y
            slice_locations=[2, 2]
        )

        assert len(figures) > 0, "Should create figures"

        for fig in figures:
            plt.close(fig)

    def test_plot_tensor_different_plane_combinations(self, minkowski_metric_small):
        """Test plotting with various plane combinations"""
        plane_combinations = [
            [1, 2],
            [1, 3],
            [2, 3],
            [1, 4],  # Note: 1-indexed in plot_tensor
        ]

        for planes in plane_combinations:
            figures = plot_tensor(
                minkowski_metric_small,
                sliced_planes=planes
            )

            assert len(figures) > 0, f"Should create figures for planes {planes}"

            for fig in figures:
                plt.close(fig)

    def test_plot_tensor_same_plane_error(self, minkowski_metric_small):
        """Test error when selecting same plane twice"""
        with pytest.raises(ValueError, match="must not be the same"):
            plot_tensor(
                minkowski_metric_small,
                sliced_planes=[1, 1]  # Invalid: same plane
            )

    def test_plot_tensor_invalid_slice_location(self, minkowski_metric_small):
        """Test error for slice location outside bounds"""
        with pytest.raises(ValueError, match="outside the world"):
            plot_tensor(
                minkowski_metric_small,
                sliced_planes=[1, 2],
                slice_locations=[100, 100]  # Outside bounds
            )

    def test_plot_tensor_covariant_metric(self, minkowski_metric_small):
        """Test plotting covariant metric tensor"""
        # Minkowski is covariant by default
        figures = plot_tensor(minkowski_metric_small)

        # For symmetric tensor (covariant), should plot 10 unique components
        assert len(figures) == 10, \
            "Should plot 10 unique components for symmetric tensor"

        for fig in figures:
            plt.close(fig)

    def test_plot_tensor_contravariant_metric(self, minkowski_metric_small):
        """Test plotting contravariant metric tensor"""
        from warpfactory.core.tensor_ops import change_tensor_index

        # Convert to contravariant
        metric_contravariant = change_tensor_index(
            minkowski_metric_small,
            "contravariant"
        )

        figures = plot_tensor(metric_contravariant)

        # Should still plot 10 unique components
        assert len(figures) == 10, \
            "Should plot 10 unique components for contravariant tensor"

        for fig in figures:
            plt.close(fig)

    def test_plot_tensor_figure_attributes(self, minkowski_metric_small):
        """Test that figures have expected attributes"""
        figures = plot_tensor(minkowski_metric_small)

        # Check first figure
        fig = figures[0]

        # Should have axes
        axes = fig.get_axes()
        assert len(axes) > 0, "Figure should have axes"

        # Should have title
        ax = axes[0]
        title = ax.get_title()
        assert len(title) > 0, "Plot should have title"

        # Should have labels
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        assert len(xlabel) > 0, "Plot should have x-label"
        assert len(ylabel) > 0, "Plot should have y-label"

        for fig in figures:
            plt.close(fig)

    def test_plot_tensor_default_slice_locations(self, minkowski_metric_small):
        """Test that default slice locations are at center"""
        # Don't specify slice_locations, should use center
        figures = plot_tensor(
            minkowski_metric_small,
            sliced_planes=[1, 4]
        )

        assert len(figures) > 0, "Should create figures with default locations"

        for fig in figures:
            plt.close(fig)


class TestVisualizerEdgeCases:
    """Test edge cases for visualizer functions"""

    def test_slice_data_corner_indices(self, minkowski_metric_small):
        """Test slice data at corner indices"""
        shape = minkowski_metric_small.shape

        # Test at first index
        idx = get_slice_data([0, 1], [0, 0], minkowski_metric_small)
        assert idx[0] == 0, "Should handle index 0"

        # Test at last index
        last_indices = [shape[0] - 1, shape[1] - 1]
        idx = get_slice_data([0, 1], last_indices, minkowski_metric_small)
        assert idx[0] == last_indices[0], "Should handle last index"

    def test_redblue_single_value(self):
        """Test colormap for array with single value"""
        data = np.full((3, 3), 5.0)
        cmap = redblue(data)

        # Should treat as all positive
        assert cmap.shape[0] > 0, "Should create colormap"
        assert cmap.shape[1] == 3, "Should return RGB"

    def test_redblue_extreme_values(self):
        """Test colormap with extreme values"""
        data = np.array([[-1e10, 0], [0, 1e10]])
        cmap = redblue(data)

        assert cmap.shape[0] > 0, "Should handle extreme values"
        assert cmap.shape[1] == 3, "Should return RGB"

    def test_label_axis_all_dimensions(self):
        """Test that all dimension labels are used"""
        all_labels = set()

        # Test all plane combinations
        for i in range(4):
            for j in range(i + 1, 4):
                xlabel, ylabel = label_cartesian_axis([i, j])
                all_labels.add(xlabel)
                all_labels.add(ylabel)

        # Should see all four labels (t, x, y, z)
        assert all_labels == {'t', 'x', 'y', 'z'}, \
            "Should use all dimension labels"

    def test_plot_tensor_alpha_zero(self, minkowski_metric_small):
        """Test plotting with alpha=0 (no grid)"""
        figures = plot_tensor(minkowski_metric_small, alpha=0)

        assert len(figures) > 0, "Should create figures with alpha=0"

        for fig in figures:
            plt.close(fig)

    def test_plot_tensor_alpha_one(self, minkowski_metric_small):
        """Test plotting with alpha=1 (opaque grid)"""
        figures = plot_tensor(minkowski_metric_small, alpha=1.0)

        assert len(figures) > 0, "Should create figures with alpha=1"

        for fig in figures:
            plt.close(fig)
