# Docker GPU Setup for WarpFactory

This guide explains how to build and run WarpFactory with GPU support using Docker.

## Prerequisites

1. **NVIDIA GPU** with compute capability 3.5 or higher
2. **NVIDIA Driver** installed on host (version 450.80.02 or higher)
3. **Docker** installed (version 19.03 or higher)
4. **NVIDIA Container Toolkit** installed

### Installing NVIDIA Container Toolkit

If you haven't installed the NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Test that Docker can access GPU
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

You should see your GPU(s) listed.

## Quick Start

### Build and Run

```bash
# Build the GPU-enabled container
docker-compose build warpfactory-gpu

# Run the container
docker-compose run --rm warpfactory-gpu

# Inside the container, test GPU
python test_gpu_usage.py
```

### Run Jupyter Notebook with GPU

```bash
# Start container with Jupyter
docker-compose run --rm -p 8888:8888 warpfactory-gpu \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Open the URL shown in your browser
```

## Building the Container

### Option 1: Using Docker Compose (Recommended)

```bash
# Build GPU-enabled container
docker-compose build warpfactory-gpu

# Or build CPU-only container for comparison
docker-compose build warpfactory-cpu
```

### Option 2: Using Docker Directly

```bash
# Build GPU-enabled image
docker build -t warpfactory:gpu-latest .

# Run with GPU support
docker run --rm --gpus all -it \
    -v $(pwd):/WarpFactory/warpfactory_py \
    warpfactory:gpu-latest
```

## Usage Examples

### Run Interactive Shell

```bash
# GPU-enabled container
docker-compose run --rm warpfactory-gpu bash

# Inside container
python
>>> from warpfactory.metrics.alcubierre import get_alcubierre_metric
>>> from warpfactory.solver.energy import get_energy_tensor
>>> # ... your code ...
```

### Run a Script

```bash
# Run script from host
docker-compose run --rm warpfactory-gpu python your_script.py

# Or copy script into container and run
docker-compose run --rm -v $(pwd)/scripts:/scripts warpfactory-gpu \
    python /scripts/your_script.py
```

### Run Tests

```bash
# Run GPU verification test
docker-compose run --rm warpfactory-gpu python test_gpu_usage.py

# Run pytest
docker-compose run --rm warpfactory-gpu pytest tests/
```

### Run Jupyter Notebook

```bash
# Start Jupyter Lab with GPU access
docker-compose run --rm -p 8888:8888 warpfactory-gpu \
    jupyter lab --ip=0.0.0.0 --allow-root --no-browser

# Access at http://localhost:8888
# Token will be shown in terminal output
```

## Configuration

### GPU Selection

To use specific GPUs:

```bash
# Use GPU 0 only
docker run --rm --gpus '"device=0"' -it warpfactory:gpu-latest

# Use GPUs 0 and 1
docker run --rm --gpus '"device=0,1"' -it warpfactory:gpu-latest

# Use all GPUs (default)
docker run --rm --gpus all -it warpfactory:gpu-latest
```

### Memory Limits

To limit container memory:

```yaml
# In docker-compose.yml, add under warpfactory-gpu:
deploy:
  resources:
    limits:
      memory: 32G
    reservations:
      memory: 16G
```

### Persistent Data

Mount volumes for persistent data:

```bash
docker-compose run --rm \
    -v $(pwd)/data:/data \
    -v $(pwd)/output:/output \
    warpfactory-gpu \
    python your_script.py
```

## Troubleshooting

### Problem: docker-compose can't find GPU

**Error:**
```
could not select device driver "" with capabilities: [[gpu]]
```

**Solution:**
1. Verify NVIDIA Container Toolkit is installed:
   ```bash
   nvidia-container-cli --version
   ```
2. Update Docker Compose to version 1.28+ (for `deploy.resources.reservations.devices`)
3. Or use older syntax in docker-compose.yml:
   ```yaml
   runtime: nvidia
   environment:
     - NVIDIA_VISIBLE_DEVICES=all
   ```

### Problem: Permission denied inside container

**Solution:**
The container runs as `warpuser`. To run as root:

```bash
docker-compose run --rm --user root warpfactory-gpu bash
```

### Problem: Can't access files

**Solution:**
Ensure volumes are mounted:

```bash
docker-compose run --rm \
    -v $(pwd):/WarpFactory/warpfactory_py \
    warpfactory-gpu bash
```

### Problem: GPU out of memory

**Solution:**
1. Clear GPU memory between runs:
   ```python
   import cupy as cp
   cp.get_default_memory_pool().free_all_blocks()
   ```
2. Use smaller grid sizes
3. Stop other containers using GPU
4. Restart Docker daemon

### Problem: CuPy not finding CUDA libraries

**Solution:**
This shouldn't happen with our Dockerfile, but if it does:

```bash
# Inside container
echo $LD_LIBRARY_PATH
# Should include /usr/local/cuda/lib64

# If not, set it
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Performance Tips

### 1. Use tmpfs for Temporary Files

```yaml
# In docker-compose.yml
volumes:
  - type: tmpfs
    target: /tmp
    tmpfs:
      size: 10G
```

### 2. Increase Shared Memory

```yaml
# In docker-compose.yml
shm_size: '2gb'
```

### 3. Use Host Network for Jupyter

```yaml
# In docker-compose.yml
network_mode: host
```

Then access Jupyter at `http://localhost:8888`

## Development Workflow

### Live Code Editing

Mount source code as volume for live editing:

```bash
# Changes to code on host are immediately reflected in container
docker-compose run --rm warpfactory-gpu bash
```

### Installing Additional Packages

```bash
# Temporary (lost when container stops)
docker-compose run --rm warpfactory-gpu bash
pip install some-package

# Permanent (add to Dockerfile)
# Edit Dockerfile, add: RUN pip install some-package
docker-compose build warpfactory-gpu
```

### Debugging

```bash
# Run with Python debugger
docker-compose run --rm warpfactory-gpu python -m pdb your_script.py

# Or use IPython for interactive debugging
docker-compose run --rm warpfactory-gpu ipython
```

## Comparison: CPU vs GPU Containers

Run both for comparison:

```bash
# Terminal 1: GPU container
docker-compose run --rm warpfactory-gpu python benchmark.py

# Terminal 2: CPU container
docker-compose run --rm warpfactory-cpu python benchmark.py
```

## Multi-GPU Usage

To use multiple GPUs in Python:

```python
import cupy as cp

# List available GPUs
print(f"Available GPUs: {cp.cuda.runtime.getDeviceCount()}")

# Use specific GPU
with cp.cuda.Device(0):
    # Operations here use GPU 0
    metric_gpu0 = metric.to_gpu()

with cp.cuda.Device(1):
    # Operations here use GPU 1
    metric_gpu1 = metric.to_gpu()
```

## Cleaning Up

```bash
# Stop all containers
docker-compose down

# Remove images
docker rmi warpfactory:gpu-latest
docker rmi warpfactory:cpu-latest

# Clean up Docker system
docker system prune -a
```

## Production Deployment

For production:

1. **Use specific version tags**
   ```dockerfile
   FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04
   ```

2. **Pin Python package versions**
   ```
   cupy-cuda12x==13.6.0
   numpy==1.26.4
   ```

3. **Run as non-root user** (already configured)

4. **Use Docker secrets** for sensitive data

5. **Set resource limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '8'
         memory: 32G
   ```

## Summary

### Key Points

1. **Use `docker-compose`** for easiest setup
2. **NVIDIA Container Toolkit** required on host
3. **GPU-enabled base image** includes all CUDA libraries
4. **Automatic fallback** to CPU if GPU unavailable
5. **Test with** `test_gpu_usage.py` inside container

### Common Commands

```bash
# Build
docker-compose build warpfactory-gpu

# Run interactive shell
docker-compose run --rm warpfactory-gpu

# Run script
docker-compose run --rm warpfactory-gpu python script.py

# Run Jupyter
docker-compose run --rm -p 8888:8888 warpfactory-gpu jupyter lab --ip=0.0.0.0 --allow-root

# Test GPU
docker-compose run --rm warpfactory-gpu python test_gpu_usage.py

# Clean up
docker-compose down
```

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [CuPy Installation in Docker](https://docs.cupy.dev/en/stable/install.html#using-cupy-on-docker)
- [WarpFactory GPU Usage Guide](GPU_USAGE.md)
