# WarpFactory Python - Installation & Quick Start Guide

## System Requirements

### Minimum Requirements
- **Python:** 3.8 or higher
- **RAM:** 4 GB minimum, 8 GB recommended
- **Storage:** 50 MB for package + dependencies
- **OS:** Linux, macOS, or Windows

### Recommended for GPU Acceleration
- **GPU:** NVIDIA GPU with CUDA support
- **CUDA:** Version 11.x or 12.x
- **RAM:** 16 GB system RAM + 4 GB GPU RAM
- **Storage:** Additional 2 GB for CuPy

---

## Installation Options

### Option 1: Basic Installation (CPU Only)

Install just the core functionality:

```bash
cd /WarpFactory/warpfactory_py
pip install -e .
```

**Includes:**
- NumPy (arrays and linear algebra)
- SciPy (scientific computing)
- Matplotlib (plotting)

**Use Cases:**
- Learning and tutorials
- Small grid computations (< 50³)
- Metric creation and basic analysis

### Option 2: GPU-Accelerated Installation

For faster computations on large grids:

```bash
cd /WarpFactory/warpfactory_py
pip install -e ".[gpu]"
```

**Additional Dependencies:**
- CuPy (GPU arrays, requires CUDA)

**Use Cases:**
- Large grid computations (100³ and above)
- Parameter sweeps
- Production research

**CUDA Installation:**
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### Option 3: Full Installation with Visualization

For advanced 3D visualization:

```bash
cd /WarpFactory/warpfactory_py
pip install -e ".[gpu,viz]"
```

**Additional Dependencies:**
- PyVista (advanced 3D rendering)
- VTK (visualization toolkit)

**Use Cases:**
- Publication-quality figures
- Interactive 3D exploration
- Presentations and demonstrations

### Option 4: Development Installation

For contributing to WarpFactory:

```bash
cd /WarpFactory/warpfactory_py
pip install -e ".[gpu,viz,dev,notebooks]"
```

**Additional Dependencies:**
- Jupyter (notebooks)
- Pytest (testing)
- Black (code formatting)
- Flake8 (linting)
- MyPy (type checking)

**Use Cases:**
- Package development
- Running example notebooks
- Contributing new features
- Creating tests

---

## Quick Start

### 1. Verify Installation

```bash
python -c "import warpfactory as wf; print(f'WarpFactory {wf.__version__} installed successfully!')"
```

Expected output: `WarpFactory 1.0.0 installed successfully!`

### 2. Run Basic Tests

```bash
cd /WarpFactory/warpfactory_py
python test_basic.py
```

Expected: 5-6 tests passing

### 3. Run Comprehensive Tests

```bash
pytest warpfactory/tests/ -v
```

Expected: All 190 tests passing in ~8 seconds

### 4. First Warp Drive Simulation

```python
import warpfactory as wf

# Create Alcubierre warp drive
metric = wf.metrics.alcubierre.get_alcubierre_metric(
    grid_size=[10, 20, 20, 20],
    world_center=[5, 10, 10, 10],
    velocity=1.0,  # Speed of light!
    radius=2.0,
    sigma=0.5
)

print(f"Created: {metric}")
print(f"Shape: {metric.shape}")
print(f"Time-time component: {metric[(0,0)].min():.3f} to {metric[(0,0)].max():.3f}")
```

### 5. Compute Energy Requirements

```python
from warpfactory.solver.energy import get_energy_tensor

# Calculate exotic matter needed
energy = get_energy_tensor(metric)

# Energy density at each point
rho = energy[(0, 0)]  # T^00 component
print(f"Energy density range: {rho.min():.2e} to {rho.max():.2e} J/m³")
```

### 6. Check Energy Conditions

```python
from warpfactory.analyzer.energy_conditions import get_energy_conditions

# Null energy condition
nec_map, _, _ = get_energy_conditions(energy, metric, "Null")

violations = (nec_map < 0).sum()
total = nec_map.size
print(f"NEC violations: {violations}/{total} ({100*violations/total:.1f}%)")
```

---

## Running Example Notebooks

### Start Jupyter

```bash
cd /WarpFactory/warpfactory_py
jupyter notebook examples/
```

Or with JupyterLab:

```bash
jupyter lab examples/
```

### Recommended Notebook Order

**For Beginners:**
1. `01_metrics/M1_First_Metric.ipynb` - Start here!
2. `01_metrics/M2_Default_Metrics.ipynb` - Explore all metrics
3. `02_energy_tensor/T1_First_Energy_Tensor.ipynb` - Compute energy
4. `03_analysis/A1_Energy_Conditions.ipynb` - Analyze physics

**For Advanced Users:**
- `01_metrics/M3_Building_a_Metric.ipynb` - Custom metrics
- `02_energy_tensor/T5_Errors.ipynb` - Numerical analysis
- `03_analysis/A3_Eval_Metric.ipynb` - Complete workflow
- `04_warp_shell/W1_Warp_Shell.ipynb` - Advanced physics

---

## GPU Setup (Optional)

### Check GPU Availability

```python
import warpfactory as wf

try:
    import cupy as cp
    print(f"✓ CuPy installed: {cp.__version__}")
    print(f"✓ CUDA available: {cp.cuda.is_available()}")

    # Get GPU info
    device = cp.cuda.Device()
    print(f"✓ GPU: {device.name}")
    print(f"✓ Memory: {device.mem_info[1] / 1e9:.1f} GB")
except ImportError:
    print("✗ CuPy not installed (CPU mode only)")
```

### Enable GPU Acceleration

```python
# Method 1: Use try_gpu parameter
energy = get_energy_tensor(metric, try_gpu=True)

# Method 2: Manual transfer
metric_gpu = metric.to_gpu()
energy_gpu = get_energy_tensor(metric_gpu)
result_cpu = energy_gpu.to_cpu()
```

---

## Troubleshooting

### Common Issues

#### 1. Import Error: "No module named 'warpfactory'"

**Solution:**
```bash
# Install in editable mode
cd /WarpFactory/warpfactory_py
pip install -e .
```

#### 2. "CuPy not installed" warning

**Solution:**
```bash
# Install CuPy for your CUDA version
pip install cupy-cuda11x  # or cupy-cuda12x
```

#### 3. Matplotlib plots not showing

**Solution:**
```python
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode

# Or explicitly show
figures = plot_tensor(metric)
plt.show()
```

#### 4. "Metric is not verified" error

**Solution:**
```python
from warpfactory.core import verify_tensor

# Check what's wrong
is_valid = verify_tensor(metric, suppress_msgs=False)
```

#### 5. Tests failing

**Solution:**
```bash
# Update dependencies
pip install --upgrade numpy scipy matplotlib

# Re-run tests
pytest warpfactory/tests/ -v
```

---

## Performance Optimization

### For Large Computations

1. **Use GPU:** 8-15x speedup for grids > 50³
2. **Reduce Grid Size:** Use cartoon methods (see T2 notebook)
3. **Lower FD Order:** Use 2nd order for speed (see T4 notebook)
4. **Optimize Grid:** Only compute regions of interest

### Memory Management

```python
# For large grids, use GPU to avoid CPU memory limits
metric = get_alcubierre_metric([10, 200, 200, 200])  # Large!
metric_gpu = metric.to_gpu()  # Move to GPU
energy_gpu = get_energy_tensor(metric_gpu, try_gpu=True)

# Only transfer results back if needed
energy_density = energy_gpu[(0, 0)].get()  # Get just one component
```

---

## Getting Help

### Documentation Resources

1. **README.md** - Package overview
2. **Example Notebooks** - Step-by-step tutorials
3. **FEATURE_MATRIX.md** - Complete feature list
4. **FINAL_CONVERSION_REPORT.md** - Detailed conversion info
5. **Function Docstrings** - `help(function_name)` in Python

### Example: Getting Function Help

```python
from warpfactory.metrics.alcubierre import get_alcubierre_metric

help(get_alcubierre_metric)
# Shows: parameters, return values, description
```

### Online Resources

- **Original Documentation:** https://applied-physics.gitbook.io/warp-factory
- **Paper:** https://iopscience.iop.org/article/10.1088/1361-6382/ad2e42
- **arXiv:** https://arxiv.org/abs/2404.03095
- **GitHub:** https://github.com/NerdsWithAttitudes/WarpFactory

### Community Support

For questions or issues:
1. Check example notebooks for similar use cases
2. Review test files for usage patterns
3. Open an issue on GitHub
4. Consult the original MATLAB documentation

---

## Next Steps

### After Installation

1. ✅ Run `test_basic.py` to verify installation
2. ✅ Open `examples/01_metrics/M1_First_Metric.ipynb`
3. ✅ Create your first warp drive!
4. ✅ Explore other example notebooks
5. ✅ Try GPU acceleration if available

### Learning Path

**Week 1:** Understand metrics (M1-M3 notebooks)
**Week 2:** Compute energy tensors (T1-T5 notebooks)
**Week 3:** Analyze energy conditions (A1-A4 notebooks)
**Week 4:** Advanced topics (Warp Shell, custom metrics)

---

## Citation

If you use WarpFactory Python in your research, please cite:

```bibtex
@article{warpfactory2024,
  title={WarpFactory: Numerical Toolkit for Analyzing Warp Drive Spacetimes},
  author={Helmerich, Christopher and Fuchs, Jared},
  journal={Classical and Quantum Gravity},
  year={2024},
  note={Python implementation available at https://github.com/NerdsWithAttitudes/WarpFactory}
}
```

---

**Ready to explore exotic spacetimes with Python!** 🚀
