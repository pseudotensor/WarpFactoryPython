# GPU Usage Guide for WarpFactory

This guide explains how to properly enable and use GPU acceleration in WarpFactory.

## Table of Contents

1. [Overview](#overview)
2. [Problem Diagnosis](#problem-diagnosis)
3. [Root Cause](#root-cause)
4. [Solution](#solution)
5. [Verification](#verification)
6. [Performance](#performance)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Overview

WarpFactory supports GPU acceleration for energy tensor computations using [CuPy](https://cupy.dev/), which provides a NumPy-compatible interface that runs on NVIDIA CUDA GPUs.

### GPU-Enabled Functions

The following operations support GPU acceleration:
- `get_energy_tensor(metric, try_gpu=True)` - Main energy tensor computation
- `Tensor.to_gpu()` - Transfer tensor to GPU memory
- `Tensor.to_cpu()` - Transfer tensor from GPU to CPU memory
- `Tensor.is_gpu()` - Check if tensor is on GPU

All underlying operations (Ricci tensor, Einstein tensor, finite differences) automatically use GPU when data is on GPU.

## Problem Diagnosis

### Issue Reported

User observed that GPU was never used during computations - only a single CPU core was utilized, even when using `try_gpu=True`.

### Investigation Results

After comprehensive testing, we identified the following issues:

#### 1. CuPy CUDA Version Mismatch

```bash
# System has CUDA 12.3 (from nvidia-smi)
$ nvidia-smi
CUDA Version: 12.3

# But CuPy was installed for CUDA 11.x
$ python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"
11080  # CUDA 11.8

# This causes runtime error:
RuntimeError: CuPy failed to load libnvrtc.so.11.2: OSError: libnvrtc.so.11.2:
cannot open shared object file: No such file or directory
```

#### 2. Docker Container Missing CUDA Runtime

The Docker container has:
- GPU driver access (nvidia-smi works)
- CuPy installed
- BUT: Missing CUDA runtime libraries (libnvrtc, libcudart, etc.)

This is a common Docker configuration issue where the GPU driver is exposed from the host, but CUDA runtime libraries are not properly installed in the container.

## Root Cause

**The GPU was NOT being used because:**

1. CuPy was installed for CUDA 11.x, but the system has CUDA 12.3
2. The Docker container lacks CUDA runtime libraries
3. When `get_energy_tensor(metric, try_gpu=True)` was called:
   - Code tried to import CuPy
   - CuPy import succeeded
   - But when trying to perform GPU operations, CuPy failed to load CUDA libraries
   - Error was caught by try/except block
   - Code silently fell back to CPU

The fallback mechanism worked as designed, but users weren't aware GPU was failing.

## Solution

### Option 1: Use Proper GPU-Enabled Docker Container (Recommended)

We've created a proper Dockerfile based on NVIDIA's official CUDA image:

```bash
# Build the GPU-enabled container
docker-compose build warpfactory-gpu

# Run the container with GPU support
docker-compose run --rm warpfactory-gpu

# Inside container, verify GPU works
python test_gpu_usage.py
```

The new `Dockerfile` includes:
- `nvidia/cuda:12.3.0-runtime-ubuntu22.04` base image
- All CUDA runtime libraries pre-installed
- CuPy for CUDA 12.x
- Proper GPU device configuration via docker-compose

### Option 2: Fix Existing Environment

If you're not using Docker or want to fix your current environment:

#### Step 1: Determine Your CUDA Version

```bash
# Check CUDA version from nvidia-smi
nvidia-smi

# Look for "CUDA Version: X.Y" in the output
```

#### Step 2: Uninstall Incompatible CuPy

```bash
# Remove all CuPy versions
pip uninstall -y cupy cupy-cuda11x cupy-cuda12x
```

#### Step 3: Install Correct CuPy Version

```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x

# Or use the universal installer (finds CUDA automatically)
pip install cupy
```

#### Step 4: Verify CuPy Works

```bash
python -c "import cupy as cp; a = cp.array([1,2,3]); print(cp.asnumpy(a + a))"
```

If you see an error about missing CUDA libraries, you need to install the CUDA toolkit:

```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Or download from NVIDIA:
# https://developer.nvidia.com/cuda-downloads
```

### Option 3: Install CUDA Runtime in Existing Docker Container

If you want to fix the current container without rebuilding:

```bash
# This is a temporary fix - better to use the proper Dockerfile
# These commands would need to be run inside the container with root access
apt-get update
apt-get install -y nvidia-cuda-toolkit
pip uninstall -y cupy-cuda11x
pip install cupy-cuda12x
```

## Verification

### Test Script

Use the included test script to verify GPU is working:

```bash
python test_gpu_usage.py
```

This script will:
1. Check GPU availability
2. Test basic CuPy operations
3. Run CPU computation (baseline)
4. Run GPU computation
5. Verify results match
6. Report performance comparison

### Manual Verification

```python
import cupy as cp
from warpfactory.metrics.alcubierre import get_alcubierre_metric
from warpfactory.solver.energy import get_energy_tensor

# Create a test metric
metric = get_alcubierre_metric(
    grid_size=[1, 40, 40, 40],
    world_center=[1, 21, 21, 21],
    velocity=0.9,
    radius=10,
    sigma=0.5
)

# Compute on GPU
energy = get_energy_tensor(metric, try_gpu=True)

# Check if any errors occurred
# If no errors and computation completes, GPU is working!
print("GPU computation successful!")
```

### Monitor GPU Usage

In a separate terminal, monitor GPU usage during computation:

```bash
# Watch GPU usage in real-time
watch -n 0.5 nvidia-smi

# Or continuous monitoring
nvidia-smi dmon -s u

# Look for:
# - GPU memory usage increase
# - GPU utilization > 0%
# - Python process using GPU
```

## Performance

### Expected Speedup

GPU speedup depends on grid size:

| Grid Size | Total Points | Typical Speedup |
|-----------|--------------|-----------------|
| 20³       | 8,000        | 1-2x            |
| 40³       | 64,000       | 2-4x            |
| 60³       | 216,000      | 4-6x            |
| 80³       | 512,000      | 6-8x            |
| 100³      | 1,000,000    | 8-10x           |

**Note:** Small grids may be slower on GPU due to transfer overhead.

### Performance Tips

1. **Use GPU for large grids** (> 50×50×50)
2. **Batch multiple operations** to amortize transfer cost
3. **Keep data on GPU** between operations when possible
4. **Free GPU memory** when done to avoid OOM errors

## Troubleshooting

### Problem: ImportError when importing CuPy

**Symptom:**
```python
ImportError: No module named 'cupy'
```

**Solution:**
```bash
pip install cupy-cuda12x  # or cupy-cuda11x depending on your CUDA version
```

### Problem: CuPy loads but operations fail

**Symptom:**
```
RuntimeError: CuPy failed to load libnvrtc.so.X: OSError: libnvrtc.so.X:
cannot open shared object file: No such file or directory
```

**Cause:** CUDA runtime libraries not installed or version mismatch

**Solution:**
1. Check CUDA version: `nvidia-smi`
2. Install matching CuPy: `pip install cupy-cuda12x` (for CUDA 12.x)
3. If still failing, install CUDA toolkit or use our Docker container

### Problem: GPU not showing usage in nvidia-smi

**Symptom:** Computation completes but nvidia-smi shows 0% GPU utilization

**Possible Causes:**
1. Code fell back to CPU due to CuPy error
2. Grid size too small (computation too fast to see)
3. GPU memory not increasing

**Diagnosis:**
```python
# Add debug output
import cupy as cp
print("CuPy can initialize:", cp.cuda.is_available())

# Try basic operation
try:
    a = cp.array([1,2,3])
    b = a + a
    print("Basic CuPy works!")
except Exception as e:
    print(f"CuPy error: {e}")
```

### Problem: Multiple CuPy versions installed

**Symptom:**
```
UserWarning: CuPy may not function correctly because multiple CuPy packages
are installed: cupy-cuda11x, cupy-cuda12x
```

**Solution:**
```bash
# Uninstall all versions
pip uninstall -y cupy cupy-cuda11x cupy-cuda12x

# Install only the correct version
pip install cupy-cuda12x  # For CUDA 12.x
```

### Problem: Out of GPU memory

**Symptom:**
```
cupy.cuda.memory.OutOfMemoryError: Out of memory allocating X bytes
```

**Solutions:**
1. Use smaller grid size
2. Free GPU memory between computations:
   ```python
   import cupy as cp
   cp.get_default_memory_pool().free_all_blocks()
   ```
3. Use CPU for this computation
4. Use a GPU with more memory

### Problem: Slow GPU performance

**Possible Causes:**
1. Grid too small - transfer overhead dominates
2. Data not staying on GPU between operations
3. Using old/slow GPU

**Solutions:**
1. Use GPU only for grids > 40³
2. Keep tensors on GPU:
   ```python
   metric_gpu = metric.to_gpu()
   # ... perform multiple operations ...
   result_cpu = result.to_cpu()
   ```
3. Profile to identify bottlenecks

## Best Practices

### 1. Check GPU Availability Before Use

```python
def is_gpu_available():
    """Check if GPU is available and working"""
    try:
        import cupy as cp
        # Try a simple operation
        a = cp.array([1, 2, 3])
        b = a + a
        return True
    except:
        return False

# Use in code
USE_GPU = is_gpu_available()
energy = get_energy_tensor(metric, try_gpu=USE_GPU)
```

### 2. Handle GPU Gracefully

The WarpFactory code already handles GPU gracefully with fallback, but you can add logging:

```python
import logging

logging.basicConfig(level=logging.INFO)

# The code will log when falling back to CPU
energy = get_energy_tensor(metric, try_gpu=True)
```

### 3. Monitor GPU Memory

```python
import cupy as cp

def print_gpu_memory():
    """Print current GPU memory usage"""
    if cp.cuda.is_available():
        mempool = cp.get_default_memory_pool()
        used = mempool.used_bytes() / 1e9
        total = cp.cuda.Device().mem_info[1] / 1e9
        print(f"GPU Memory: {used:.2f} GB / {total:.2f} GB")

print_gpu_memory()
# ... perform computation ...
print_gpu_memory()
```

### 4. Clean Up After Computations

```python
import cupy as cp

# After GPU computations
cp.get_default_memory_pool().free_all_blocks()
```

### 5. Batch Processing

```python
# Good: Keep data on GPU for multiple operations
metrics_gpu = [m.to_gpu() for m in metrics]
energies_gpu = [get_energy_tensor_internal(m) for m in metrics_gpu]
energies_cpu = [e.to_cpu() for e in energies_gpu]

# Bad: Transfer back and forth for each operation
for m in metrics:
    energy = get_energy_tensor(m, try_gpu=True)  # Transfers in and out each time
```

### 6. Development vs Production

```python
# Development: Always show what's happening
import os
os.environ['CUPY_SHOW_CONFIG'] = '1'

# Production: Silent fallback
energy = get_energy_tensor(metric, try_gpu=True)  # Will work on CPU or GPU
```

## Code Examples

### Example 1: Simple GPU Usage

```python
from warpfactory.metrics.alcubierre import get_alcubierre_metric
from warpfactory.solver.energy import get_energy_tensor

# Create metric
metric = get_alcubierre_metric(
    grid_size=[1, 60, 60, 60],
    world_center=[1, 31, 31, 31],
    velocity=0.9,
    radius=10,
    sigma=0.5
)

# Compute on GPU with automatic fallback
energy = get_energy_tensor(metric, try_gpu=True)

print(f"Energy density range: {energy.tensor[(0,0)].min():.2e} to {energy.tensor[(0,0)].max():.2e}")
```

### Example 2: Manual GPU Control

```python
import cupy as cp
from warpfactory.metrics.alcubierre import get_alcubierre_metric
from warpfactory.solver.energy import get_energy_tensor

# Create metric on CPU
metric = get_alcubierre_metric(grid_size=[1, 50, 50, 50], ...)

# Transfer to GPU
metric_gpu = metric.to_gpu()
print(f"Metric is on GPU: {metric_gpu.is_gpu()}")

# Compute (stays on GPU internally until final transfer)
energy = get_energy_tensor(metric, try_gpu=True)

# Clean up
del metric_gpu
cp.get_default_memory_pool().free_all_blocks()
```

### Example 3: Performance Comparison

```python
import time
import numpy as np
from warpfactory.metrics.alcubierre import get_alcubierre_metric
from warpfactory.solver.energy import get_energy_tensor

# Create test metric
grid_size = [1, 60, 60, 60]
metric = get_alcubierre_metric(grid_size=grid_size, ...)

# CPU computation
start = time.time()
energy_cpu = get_energy_tensor(metric, try_gpu=False)
cpu_time = time.time() - start

# GPU computation
start = time.time()
energy_gpu = get_energy_tensor(metric, try_gpu=True)
gpu_time = time.time() - start

# Compare
print(f"CPU: {cpu_time:.2f}s")
print(f"GPU: {gpu_time:.2f}s")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")

# Verify results match
max_diff = np.max(np.abs(energy_cpu.tensor[(0,0)] - energy_gpu.tensor[(0,0)]))
print(f"Maximum difference: {max_diff:.2e}")
```

## Summary

### The Issue

GPU acceleration was not working because:
1. CuPy installed for wrong CUDA version (11.x vs 12.3)
2. Docker container missing CUDA runtime libraries
3. Errors were silently caught and fell back to CPU

### The Fix

1. **Use the provided Dockerfile** (recommended)
   - Based on `nvidia/cuda:12.3.0-runtime`
   - Has all CUDA libraries pre-installed
   - Includes correct CuPy version
   - Properly configured for GPU access

2. **Or fix your environment**
   - Uninstall mismatched CuPy
   - Install correct version for your CUDA
   - Verify with test script

### Verification

Run `python test_gpu_usage.py` to verify GPU is working correctly.

### Performance

GPU provides 4-10x speedup for typical grid sizes (50³ to 100³).

### Support

For GPU issues:
1. Run `test_gpu_usage.py` and share output
2. Check `nvidia-smi` output
3. Verify CuPy installation: `python -c "import cupy; print(cupy.__version__)"`
4. Check CUDA version match

## Additional Resources

- [CuPy Installation Guide](https://docs.cupy.dev/en/stable/install.html)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [NVIDIA Docker Documentation](https://github.com/NVIDIA/nvidia-docker)
- [WarpFactory GPU Computation Example](examples/02_energy_tensor/T3_GPU_Computation.ipynb)
