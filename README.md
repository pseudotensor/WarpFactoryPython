# WarpFactory Python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**WarpFactory** is a powerful numerical toolkit written in Python for analyzing warp drive spacetimes using Einstein's theory of General Relativity. This is a complete Python port of the original MATLAB implementation.

## Key Features

- 3D finite difference solver for the stress-energy tensor
- Energy condition evaluations for the point-wise Null, Weak, Dominant, and Strong Energy Conditions
- Metric scalar evaluation for the shear, expansion, and vorticity
- Momentum flow visualizations
- GPU utilization for accelerated computations (via CuPy)
- Pure Python implementation using NumPy - no MATLAB required!

## Installation

### Basic Installation

```bash
pip install -e .
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

### With Visualization Tools

```bash
pip install -e ".[viz]"
```

### Full Installation (All Features)

```bash
pip install -e ".[gpu,viz,dev,notebooks]"
```

## Quick Start

```python
import warpfactory as wf
import numpy as np

# Create an Alcubierre warp drive metric
grid_size = [10, 20, 20, 20]  # [t, x, y, z]
world_center = [5, 10, 10, 10]
velocity = 1.0  # Speed in units of c
radius = 2.0
sigma = 0.5

metric = wf.metrics.alcubierre.get_alcubierre_metric(
    grid_size, world_center, velocity, radius, sigma
)

# Calculate the stress-energy tensor
energy_tensor = wf.solver.get_energy_tensor(metric)

# Evaluate energy conditions
null_condition = wf.analyzer.get_energy_conditions(
    energy_tensor, metric, "Null"
)

# Visualize
wf.visualizer.plot_tensor(energy_tensor)
```

## Package Structure

```
warpfactory/
├── units/          # Physical constants and unit conversions
├── metrics/        # Spacetime metric definitions
│   ├── alcubierre/
│   ├── lentz/
│   ├── minkowski/
│   ├── schwarzschild/
│   └── ...
├── solver/         # Einstein field equation solvers
├── analyzer/       # Energy conditions and metric analysis
├── visualizer/     # Plotting and visualization tools
└── examples/       # Jupyter notebook examples
```

## Documentation

### Quick Links
- [Getting Started Guide](docs/00_START_HERE.md) - Start here if you're new to WarpFactory
- [Installation Guide](docs/INSTALLATION_GUIDE.md) - Detailed installation instructions
- [Feature Matrix](docs/FEATURE_MATRIX.md) - Complete feature comparison
- [Project Documentation](docs/) - Full documentation directory

### Online Resources
For comprehensive API documentation and interactive tutorials, visit the [WarpFactory Documentation](https://applied-physics.gitbook.io/warp-factory).

### Validation
The Python implementation has been thoroughly validated against published results. See the [validation/](validation/) directory for validation scripts and detailed reports.

### Testing
Unit and integration tests are available in the [tests/](tests/) directory.

### References
- [CQG Paper](https://iopscience.iop.org/article/10.1088/1361-6382/ad2e42)
- [arXiv](https://arxiv.org/abs/2404.03095)

## Development Team

- Christopher Helmerich
- Jared Fuchs
- Python Port Contributors

We would like to extend our gratitude to the following individuals for their contributions and code reviews:
- Alexey Bobrick
- Luke Sellers
- Brandon Melcher
- Justin Feng
- Gianni Martire

## License

WarpFactory is released under the [MIT License](https://opensource.org/licenses/MIT).

## Citation

If you use WarpFactory in your research, please cite:

```bibtex
@article{warpfactory2024,
  title={WarpFactory: Numerical Toolkit for Analyzing Warp Drive Spacetimes},
  author={Helmerich, Christopher and Fuchs, Jared},
  journal={Classical and Quantum Gravity},
  year={2024}
}
```
