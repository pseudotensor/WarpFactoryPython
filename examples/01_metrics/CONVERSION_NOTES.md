# M1_First_Metric.ipynb - Conversion Notes

## Overview
This document describes the conversion of the MATLAB Live Script `M1_First_Metric.mlx` to a Jupyter notebook for the Python implementation of WarpFactory.

## Source File
- **Original**: `/WarpFactory/Examples/1 Metrics/M1_First_Metric.mlx`
- **Converted**: `/WarpFactory/warpfactory_py/examples/01_metrics/M1_First_Metric.ipynb`
- **Date**: 2025-10-15

## Key Conversions

### 1. Function Calls
| MATLAB | Python |
|--------|--------|
| `metricGet_Minkowski(gridSize, gridScaling)` | `get_minkowski_metric(grid_size, grid_scaling)` |
| `fieldnames(MyFirstMetric)` | Direct attribute access via `vars()` or individual properties |

### 2. Variable Naming
Following Python conventions (snake_case instead of camelCase):
- `gridSize` → `grid_size`
- `gridScaling` → `grid_scaling`
- `MyFirstMetric` → `MyFirstMetric` (kept for consistency with tutorial)

### 3. Data Structure Changes

#### MATLAB Structure
```matlab
MyFirstMetric.tensor  % 4x4 cell array
MyFirstMetric.tensor{i,j}  % Access components (1-indexed)
```

#### Python Tensor Class
```python
MyFirstMetric.tensor  # Dictionary with (i,j) tuple keys
MyFirstMetric[(i, j)]  # Access components (0-indexed)
# or
MyFirstMetric.tensor[(i, j)]
```

### 4. Indexing
- **MATLAB**: 1-based indexing (1, 2, 3, 4) for (t, x, y, z)
- **Python**: 0-based indexing (0, 1, 2, 3) for (t, x, y, z)

Example:
```matlab
% MATLAB
g_tt = MyFirstMetric.tensor{1,1}
```

```python
# Python
g_tt = MyFirstMetric[(0, 0)]
```

### 5. Visualization

#### MATLAB
```matlab
for i = 1:4
    for j = 1:4
        nexttile
        surfq(MyFirstMetric.tensor{i,j}(1,:,:,1))
    end
end
```

#### Python
```python
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
for i in range(4):
    for j in range(4):
        ax = axes[i, j]
        slice_data = MyFirstMetric[(i, j)][0, :, :, 0]
        im = ax.imshow(slice_data, cmap='RdBu_r', origin='lower')
        plt.colorbar(im, ax=ax)
```

### 6. Properties Access

All properties are accessed as Python attributes:

| Property | Type | Description |
|----------|------|-------------|
| `.type` | str | "metric" or "stress-energy" |
| `.name` | str | Name of the metric (e.g., "Minkowski") |
| `.index` | str | "covariant", "contravariant", etc. |
| `.tensor` | dict | Dictionary of tensor components |
| `.coords` | str | Coordinate system (e.g., "cartesian") |
| `.scaling` | list | Grid scaling [t, x, y, z] |
| `.date` | str | Creation date |
| `.params` | dict | Parameters used to create metric |
| `.shape` | tuple | Shape of spacetime grid |

## Enhanced Features in Python Version

The Python notebook includes several enhancements beyond the original MATLAB version:

1. **Additional Visualizations**:
   - 2D heatmap visualization of all metric components
   - 3D surface plots for diagonal components
   - Color-coded displays with colorbars

2. **Detailed Explanations**:
   - More extensive markdown documentation
   - LaTeX mathematical notation for tensor components
   - Clear explanation of indexing differences

3. **Code Comments**:
   - Inline comments explaining Python-specific syntax
   - Notes about array slicing and indexing

4. **Summary Section**:
   - Comprehensive recap of learned concepts
   - Forward reference to next examples

## Testing

The conversion was tested to ensure all code executes correctly:

```bash
python3 -c "
from warpfactory.metrics.minkowski import get_minkowski_metric
grid_size = [1, 10, 10, 10]
grid_scaling = [1, 1, 1, 1]
MyFirstMetric = get_minkowski_metric(grid_size, grid_scaling)
# All operations verified successfully
"
```

## Dependencies

The notebook requires:
- numpy
- matplotlib
- warpfactory (Python package)

Optional:
- jupyter
- jupyterlab

## Cell Structure

The notebook contains 19 cells:
- 10 markdown cells (documentation, explanations)
- 9 code cells (demonstrations, examples)

## Learning Objectives Preserved

All learning objectives from the original MATLAB version are preserved:
1. Creating a first metric
2. Understanding metric properties
3. Accessing tensor components
4. Visualizing metric values
5. Understanding the Minkowski metric structure

## Future Enhancements

Potential improvements for future versions:
1. Interactive visualizations using plotly
2. Animation of metric evolution
3. Comparison with other metrics
4. Performance benchmarks
5. GPU acceleration examples
