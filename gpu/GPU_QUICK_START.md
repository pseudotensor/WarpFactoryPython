# GPU Quick Start Guide

Get GPU acceleration working in 5 minutes.

## The Problem

**GPU was never used** - even with `try_gpu=True`, computations ran on CPU only.

## The Root Cause

Two issues:
1. CuPy installed for CUDA 11.x, but system has CUDA 12.3
2. Docker container missing CUDA runtime libraries

## The Solution

### Option 1: Use Docker (Recommended)

```bash
# Build GPU-enabled container
cd /WarpFactory/warpfactory_py
docker-compose build warpfactory-gpu

# Run with GPU support
docker-compose run --rm warpfactory-gpu

# Verify GPU works
python test_gpu_usage.py
```

**Done!** The Docker container has everything pre-configured.

### Option 2: Fix Your Environment

```bash
# 1. Check your CUDA version
nvidia-smi  # Look for "CUDA Version: X.Y"

# 2. Uninstall wrong CuPy
pip uninstall -y cupy cupy-cuda11x cupy-cuda12x

# 3. Install correct CuPy
pip install cupy-cuda12x  # For CUDA 12.x
# OR
pip install cupy-cuda11x  # For CUDA 11.x

# 4. Test it works
python test_gpu_usage.py
```

## Quick Test

```python
# Test CuPy works
import cupy as cp
a = cp.array([1, 2, 3])
print(cp.asnumpy(a + a))  # Should print [2 4 6]

# Test WarpFactory GPU
from warpfactory.metrics.alcubierre import get_alcubierre_metric
from warpfactory.solver.energy import get_energy_tensor

metric = get_alcubierre_metric(
    grid_size=[1, 40, 40, 40],
    world_center=[1, 21, 21, 21],
    velocity=0.9
)

energy = get_energy_tensor(metric, try_gpu=True)
print("GPU working!")
```

## Monitor GPU Usage

```bash
# In a separate terminal
watch -n 0.5 nvidia-smi

# Look for:
# - GPU memory increase
# - GPU utilization > 0%
# - Your Python process listed
```

## Performance Expectations

| Grid Size | Speedup |
|-----------|---------|
| 40³       | 2-4x    |
| 60³       | 4-6x    |
| 80³       | 6-8x    |
| 100³      | 8-10x   |

Grids smaller than 40³ may not benefit from GPU due to transfer overhead.

## Troubleshooting

### "CuPy failed to load libnvrtc"

**Fix:** Wrong CuPy version installed
```bash
pip uninstall -y cupy-cuda11x cupy-cuda12x
pip install cupy-cuda12x  # Match your CUDA version
```

### "Multiple CuPy packages installed"

**Fix:**
```bash
pip uninstall -y cupy cupy-cuda11x cupy-cuda12x
pip install cupy-cuda12x
```

### GPU still not working

**Fix:** Use Docker container (has everything pre-installed)
```bash
docker-compose build warpfactory-gpu
docker-compose run --rm warpfactory-gpu
```

## Files Created

- `test_gpu_usage.py` - Comprehensive GPU test script
- `Dockerfile` - GPU-enabled container definition
- `docker-compose.yml` - Easy container orchestration
- `GPU_USAGE.md` - Complete GPU documentation
- `DOCKER_GPU_SETUP.md` - Docker-specific instructions
- `GPU_QUICK_START.md` - This file

## Next Steps

1. **Read** `GPU_USAGE.md` for complete documentation
2. **Run** `python test_gpu_usage.py` to verify setup
3. **Check** `examples/02_energy_tensor/T3_GPU_Computation.ipynb` for examples
4. **Use** `try_gpu=True` in your code

## Support

If GPU still doesn't work:
1. Run `python test_gpu_usage.py` and share output
2. Check `nvidia-smi` output
3. Try the Docker container
4. See `GPU_USAGE.md` troubleshooting section
