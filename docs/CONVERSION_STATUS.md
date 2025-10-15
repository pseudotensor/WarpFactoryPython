# WarpFactory MATLAB to Python Conversion Status

## Overview
This document tracks the conversion of WarpFactory from MATLAB to Python using NumPy, SciPy, and optionally CuPy for GPU acceleration.

## Completed Components ✅

### 1. Package Infrastructure
- ✅ `setup.py` - Package configuration with dependencies
- ✅ `requirements.txt` - Core and optional dependencies
- ✅ `README.md` - Package documentation
- ✅ Main `__init__.py` files for all modules

### 2. Units Module (`warpfactory/units/`)
- ✅ `constants.py` - Physical constants (c, G)
- ✅ `length.py` - Length unit conversions (mm, cm, meter, km)
- ✅ `mass.py` - Mass unit conversions (gram, kg, tonne)
- ✅ `time.py` - Time unit conversions (ms, second)

### 3. Core Tensor System (`warpfactory/core/`)
- ✅ `tensor.py` - Tensor class with GPU support
  - Dictionary-based 4x4 tensor storage
  - Automatic GPU/CPU conversion
  - Metadata (type, name, index, coords, params)
- ✅ `tensor_ops.py` - Tensor operations
  - `c3_inv()` - 3x3 matrix inversion
  - `c4_inv()` - 4x4 matrix inversion
  - `c_det()` - 4x4 determinant calculation
  - `verify_tensor()` - Tensor validation
  - `change_tensor_index()` - Index transformations (covariant ↔ contravariant ↔ mixed)

### 4. Metrics Module (`warpfactory/metrics/`)
- ✅ `three_plus_one.py` - 3+1 spacetime decomposition
  - `set_minkowski_three_plus_one()` - Flat space decomposition
  - `three_plus_one_builder()` - Build metric from α, β, γ components
  - `three_plus_one_decomposer()` - Decompose metric into components
- ✅ `minkowski/` - Minkowski (flat) spacetime
  - `get_minkowski_metric()` - Create flat spacetime metric
- ✅ `alcubierre/` - Alcubierre warp drive
  - `get_alcubierre_metric()` - Create Alcubierre metric
  - `shape_function_alcubierre()` - Warp bubble shape function

## In Progress 🔄

### 5. Solver Module (`warpfactory/solver/`)
The solver module contains the Einstein field equation solvers and numerical methods.

**High Priority:**
- 🔄 `finite_differences.py` - 4th and 2nd order finite difference schemes
- 🔄 `met2den.py` - Metric to stress-energy tensor conversion
- 🔄 `ricci.py` - Ricci tensor calculation
- 🔄 `einstein.py` - Einstein tensor calculation
- 🔄 `christoffel.py` - Christoffel symbols

## Pending Components 📋

### 6. Additional Metrics
- ⏳ Lentz metric
- ⏳ Van Den Broeck metric
- ⏳ Schwarzschild metric
- ⏳ Modified Time metric
- ⏳ Warp Shell metric

### 7. Analyzer Module (`warpfactory/analyzer/`)
- ⏳ `energy_conditions.py` - Null, Weak, Dominant, Strong conditions
- ⏳ `scalars.py` - Shear, expansion, vorticity
- ⏳ `momentum_flow.py` - Momentum flow calculations
- ⏳ `frame_transfer.py` - Eulerian frame transformations

### 8. Visualizer Module (`warpfactory/visualizer/`)
- ⏳ `plot_tensor.py` - 2D/3D tensor visualization
- ⏳ `plot_three_plus_one.py` - 3+1 decomposition plots
- ⏳ `utils.py` - Plotting utilities (colormaps, slicing)

### 9. Examples (Jupyter Notebooks)
- ⏳ M1_First_Metric.ipynb
- ⏳ M2_Default_Metrics.ipynb
- ⏳ M3_Building_a_Metric.ipynb
- ⏳ T1_First_Energy_Tensor.ipynb
- ⏳ T2_Cartoon_Methods.ipynb
- ⏳ T3_GPU_Computation.ipynb
- ⏳ T4_Second_vs_Fourth_Order.ipynb
- ⏳ T5_Errors.ipynb
- ⏳ A1_Energy_Conditions.ipynb
- ⏳ A2_Metric_Scalars.ipynb
- ⏳ A3_Eval_Metric.ipynb
- ⏳ A4_Momentum_Flow.ipynb
- ⏳ W1_Warp_Shell.ipynb

### 10. Testing & Validation
- ⏳ Unit tests for all modules
- ⏳ Integration tests
- ⏳ Comparison tests (Python vs MATLAB outputs)
- ⏳ Performance benchmarks

## Key Design Decisions

### Tensor Representation
**MATLAB**: Cell arrays of 4D matrices
```matlab
metric.tensor{i,j}  % 4x4 cell array
```

**Python**: Dictionary of numpy arrays
```python
metric.tensor[(i,j)]  # dict with (i,j) tuples as keys
```

### Index Convention
- Both use 0-based indexing internally for Python
- MATLAB uses 1-based, converted to 0-based in Python
- Index types: covariant, contravariant, mixedupdown, mixeddownup

### GPU Support
**MATLAB**: `gpuArray()` from Parallel Computing Toolbox
**Python**: CuPy (drop-in NumPy replacement)
```python
metric.to_gpu()  # Convert to GPU
metric.to_cpu()  # Convert back to CPU
```

## Installation

```bash
# Basic installation
cd /WarpFactory/warpfactory_py
pip install -e .

# With GPU support
pip install -e ".[gpu]"

# With visualization
pip install -e ".[viz]"

# Full installation (dev + notebooks)
pip install -e ".[gpu,viz,dev,notebooks]"
```

## Usage Example

```python
import warpfactory as wf
import numpy as np

# Create Alcubierre metric
grid_size = [10, 20, 20, 20]  # [t, x, y, z]
world_center = [5, 10, 10, 10]
velocity = 1.0  # Speed in units of c
radius = 2.0
sigma = 0.5

metric = wf.metrics.alcubierre.get_alcubierre_metric(
    grid_size, world_center, velocity, radius, sigma
)

# Verify tensor
from wf.core import verify_tensor
verify_tensor(metric)

# Convert to GPU (if CuPy installed)
metric_gpu = metric.to_gpu()

# Calculate stress-energy tensor (when solver is complete)
# energy = wf.solver.get_energy_tensor(metric)
```

## Performance Considerations

1. **Finite Differences**: 4th order scheme requires 5 grid points in each direction
2. **Memory**: Full 4x4 tensor with 4D arrays → significant memory usage
3. **GPU Acceleration**: Major speedup for large grids (100+ per dimension)
4. **Vectorization**: All operations fully vectorized using NumPy

## Next Steps

1. Complete Solver module (highest priority - needed for energy tensor calculations)
2. Implement Analyzer module (energy conditions, scalars)
3. Create remaining metric implementations
4. Build Visualizer module
5. Convert MATLAB live scripts to Jupyter notebooks
6. Write comprehensive tests
7. Performance optimization and benchmarking

## Contributing

See `CONTRIBUTING.md` for guidelines on contributing to the Python port.

## References

- Original MATLAB repository: https://github.com/NerdsWithAttitudes/WarpFactory
- Documentation: https://applied-physics.gitbook.io/warp-factory
- Paper: https://iopscience.iop.org/article/10.1088/1361-6382/ad2e42
