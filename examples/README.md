# WarpFactory Python Examples

This directory contains Jupyter notebook examples demonstrating how to use the WarpFactory Python package. These examples are conversions of the original MATLAB Live Scripts from the WarpFactory MATLAB package.

## Directory Structure

- `01_metrics/` - Introduction to spacetime metrics and the Tensor class
- More example directories coming soon...

## Getting Started

To run these examples, you'll need to install the required dependencies:

```bash
pip install numpy matplotlib jupyter
```

Then navigate to an example directory and launch Jupyter:

```bash
cd examples/01_metrics
jupyter notebook
```

## Examples Overview

### 01_metrics - Introduction to Metrics

- **M1_First_Metric.ipynb**: Learn the basics of creating and working with spacetime metrics
  - Creating a Minkowski (flat spacetime) metric
  - Understanding the Tensor class structure
  - Accessing tensor components
  - Visualizing metric components

## Differences from MATLAB Version

The Python implementation has some key differences from the MATLAB version:

1. **Indexing**: Python uses 0-based indexing (0, 1, 2, 3) instead of MATLAB's 1-based indexing
2. **Data Structure**: Instead of MATLAB structs, Python uses:
   - `Tensor` class for metric objects
   - Dictionary with tuple keys `(i, j)` for tensor components
   - NumPy arrays for tensor values
3. **Function Names**:
   - `metricGet_Minkowski` → `get_minkowski_metric`
   - `fieldnames()` → direct attribute access
4. **Visualization**: Uses matplotlib instead of MATLAB's plotting functions

## Contributing

When adding new examples:
1. Follow the existing naming convention (M#_Description.ipynb)
2. Include markdown cells with clear explanations
3. Add comments to code cells
4. Include visualizations where appropriate
5. Test all code cells to ensure they run without errors
