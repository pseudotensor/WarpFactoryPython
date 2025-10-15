#!/usr/bin/env python
"""
GPU Usage Test Script for WarpFactory

This script tests GPU acceleration in WarpFactory by:
1. Running computations on CPU (baseline)
2. Running same computations on GPU
3. Monitoring GPU usage during computation
4. Timing both runs
5. Verifying results match

Run this script and monitor GPU usage with:
    watch -n 0.5 nvidia-smi
"""

import numpy as np
import time
import sys
import os

# Add WarpFactory to path if needed
sys.path.insert(0, '/WarpFactory/warpfactory_py')

def check_gpu_availability():
    """Check if GPU and CuPy are available"""
    print("=" * 80)
    print("GPU AVAILABILITY CHECK")
    print("=" * 80)

    try:
        import cupy as cp
        print(f"✓ CuPy version: {cp.__version__}")
        print(f"✓ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")

        if cp.cuda.is_available():
            print(f"✓ CUDA is available")
            device = cp.cuda.Device(0)
            mem_info = device.mem_info
            total_mem = mem_info[1] / 1e9
            free_mem = mem_info[0] / 1e9
            used_mem = total_mem - free_mem
            print(f"✓ GPU memory: {used_mem:.2f} GB used / {total_mem:.2f} GB total")
            return True
        else:
            print("✗ CUDA is not available")
            return False
    except ImportError as e:
        print(f"✗ CuPy is not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def test_basic_cupy():
    """Test basic CuPy operations"""
    print("\n" + "=" * 80)
    print("BASIC CUPY TEST")
    print("=" * 80)

    try:
        import cupy as cp

        # Create arrays on GPU
        print("Creating arrays on GPU...")
        a_gpu = cp.array([1, 2, 3, 4, 5])
        b_gpu = cp.array([5, 4, 3, 2, 1])

        # Perform computation
        print("Performing computation on GPU...")
        c_gpu = a_gpu + b_gpu

        # Transfer back to CPU
        c_cpu = cp.asnumpy(c_gpu)

        print(f"✓ Result: {c_cpu}")
        print("✓ Basic CuPy operations working correctly")

        # Check memory usage
        mempool = cp.get_default_memory_pool()
        print(f"✓ GPU memory pool used: {mempool.used_bytes() / 1e6:.2f} MB")

        # Clean up
        mempool.free_all_blocks()
        print("✓ GPU memory freed")

        return True
    except Exception as e:
        print(f"✗ Basic CuPy test failed: {e}")
        return False


def create_test_metric(grid_size):
    """Create an Alcubierre metric for testing"""
    print(f"\nCreating Alcubierre metric with grid size {grid_size}...")

    from warpfactory.metrics.alcubierre import get_alcubierre_metric

    world_center = [(grid_size[i] + 1) / 2 for i in range(4)]

    metric = get_alcubierre_metric(
        grid_size=grid_size,
        world_center=world_center,
        velocity=0.9,
        radius=10,
        sigma=0.5
    )

    total_points = np.prod(grid_size)
    print(f"✓ Metric created: {metric.name}")
    print(f"✓ Grid shape: {metric.shape}")
    print(f"✓ Total grid points: {total_points:,}")

    # Calculate memory size
    bytes_per_component = metric.tensor[(0, 0)].nbytes
    total_memory = bytes_per_component * 16 / 1e6  # 16 components in MB
    print(f"✓ Memory per component: {bytes_per_component / 1e6:.2f} MB")
    print(f"✓ Total metric memory: {total_memory:.2f} MB")

    return metric


def run_cpu_computation(metric):
    """Run energy tensor computation on CPU"""
    print("\n" + "=" * 80)
    print("CPU COMPUTATION")
    print("=" * 80)

    from warpfactory.solver.energy import get_energy_tensor

    print("Computing energy tensor on CPU...")
    print("(This should NOT show GPU activity in nvidia-smi)")

    start_time = time.time()
    energy_cpu = get_energy_tensor(metric, try_gpu=False)
    end_time = time.time()

    elapsed = end_time - start_time

    print(f"✓ CPU computation completed")
    print(f"✓ Time elapsed: {elapsed:.3f} seconds")

    # Check result statistics
    energy_density = energy_cpu.tensor[(0, 0)]
    print(f"✓ Energy density shape: {energy_density.shape}")
    print(f"✓ Energy density range: [{np.min(energy_density):.6e}, {np.max(energy_density):.6e}]")
    print(f"✓ Energy density mean: {np.mean(energy_density):.6e}")

    return energy_cpu, elapsed


def run_gpu_computation(metric):
    """Run energy tensor computation on GPU"""
    print("\n" + "=" * 80)
    print("GPU COMPUTATION")
    print("=" * 80)

    from warpfactory.solver.energy import get_energy_tensor
    import cupy as cp

    # Clear GPU memory before starting
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()

    initial_gpu_mem = mempool.used_bytes() / 1e6
    print(f"Initial GPU memory: {initial_gpu_mem:.2f} MB")

    print("\nComputing energy tensor on GPU...")
    print("*** WATCH nvidia-smi NOW - YOU SHOULD SEE GPU UTILIZATION ***")
    print("*** Run: watch -n 0.5 nvidia-smi ***")
    print("\nStarting in 3 seconds...")
    time.sleep(3)

    start_time = time.time()
    energy_gpu = get_energy_tensor(metric, try_gpu=True)
    end_time = time.time()

    elapsed = end_time - start_time

    peak_gpu_mem = mempool.used_bytes() / 1e6
    print(f"\n✓ GPU computation completed")
    print(f"✓ Time elapsed: {elapsed:.3f} seconds")
    print(f"✓ Peak GPU memory: {peak_gpu_mem:.2f} MB")
    print(f"✓ GPU memory increase: {peak_gpu_mem - initial_gpu_mem:.2f} MB")

    # Check result statistics
    energy_density = energy_gpu.tensor[(0, 0)]
    print(f"✓ Energy density shape: {energy_density.shape}")
    print(f"✓ Energy density range: [{np.min(energy_density):.6e}, {np.max(energy_density):.6e}]")
    print(f"✓ Energy density mean: {np.mean(energy_density):.6e}")

    # Check if result is actually on CPU (should be, as per energy.py line 123)
    print(f"✓ Result is on GPU: {energy_gpu.is_gpu()}")

    return energy_gpu, elapsed


def verify_results(energy_cpu, energy_gpu):
    """Verify that CPU and GPU results match"""
    print("\n" + "=" * 80)
    print("RESULT VERIFICATION")
    print("=" * 80)

    print("Comparing CPU and GPU results...")

    max_diffs = {}
    for i in range(4):
        for j in range(4):
            diff = np.abs(energy_cpu.tensor[(i, j)] - energy_gpu.tensor[(i, j)])
            max_diff = np.max(diff)
            max_diffs[(i, j)] = max_diff

    overall_max_diff = max(max_diffs.values())

    print(f"\nMaximum differences per component:")
    for i in range(4):
        for j in range(4):
            print(f"  T[{i},{j}]: {max_diffs[(i, j)]:.6e}")

    print(f"\n✓ Overall maximum difference: {overall_max_diff:.6e}")

    if overall_max_diff < 1e-10:
        print("✓ Results match perfectly!")
        return True
    elif overall_max_diff < 1e-6:
        print("✓ Results match within acceptable tolerance")
        return True
    else:
        print("⚠ Results differ more than expected - may indicate numerical issues")
        return False


def print_performance_summary(cpu_time, gpu_time):
    """Print performance comparison summary"""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    speedup = cpu_time / gpu_time

    print(f"CPU Time:  {cpu_time:.3f} seconds")
    print(f"GPU Time:  {gpu_time:.3f} seconds")
    print(f"Speedup:   {speedup:.2f}x")

    if speedup > 1.5:
        print("✓ GPU provides significant speedup!")
    elif speedup > 1.0:
        print("✓ GPU is faster (modest speedup)")
    elif speedup > 0.8:
        print("⚠ GPU and CPU are roughly equal (overhead may dominate)")
    else:
        print("⚠ CPU is faster (grid may be too small for GPU)")


def run_sustained_gpu_test():
    """Run a longer GPU test to monitor usage"""
    print("\n" + "=" * 80)
    print("SUSTAINED GPU TEST (for monitoring)")
    print("=" * 80)

    try:
        import cupy as cp
        from warpfactory.metrics.alcubierre import get_alcubierre_metric
        from warpfactory.solver.energy import get_energy_tensor

        # Create a larger metric for sustained computation
        grid_size = [1, 60, 60, 60]
        print(f"\nCreating larger metric (grid size {grid_size}) for sustained test...")
        world_center = [(grid_size[i] + 1) / 2 for i in range(4)]

        metric = get_alcubierre_metric(
            grid_size=grid_size,
            world_center=world_center,
            velocity=0.9,
            radius=10,
            sigma=0.5
        )

        print("\n*** STARTING SUSTAINED GPU COMPUTATION ***")
        print("*** Monitor with: nvidia-smi (in another terminal) ***")
        print("*** This will run for ~30-60 seconds ***")
        print("\nStarting in 5 seconds...")
        time.sleep(5)

        start_time = time.time()
        energy = get_energy_tensor(metric, try_gpu=True)
        elapsed = time.time() - start_time

        print(f"\n✓ Sustained test completed in {elapsed:.3f} seconds")

        return True
    except Exception as e:
        print(f"✗ Sustained test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test routine"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "WARPFACTORY GPU USAGE TEST" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Check GPU availability
    gpu_available = check_gpu_availability()

    if not gpu_available:
        print("\n" + "!" * 80)
        print("GPU NOT AVAILABLE - Cannot run GPU tests")
        print("Please install CuPy: pip install cupy-cuda11x or cupy-cuda12x")
        print("!" * 80)
        return 1

    # Test basic CuPy
    if not test_basic_cupy():
        print("\n" + "!" * 80)
        print("Basic CuPy test failed - GPU may not be working correctly")
        print("!" * 80)
        return 1

    # Create test metric (moderate size for reasonable timing)
    grid_size = [1, 50, 50, 50]  # 125,000 points
    metric = create_test_metric(grid_size)

    # Run CPU computation
    energy_cpu, cpu_time = run_cpu_computation(metric)

    # Run GPU computation
    energy_gpu, gpu_time = run_gpu_computation(metric)

    # Verify results match
    results_match = verify_results(energy_cpu, energy_gpu)

    # Print summary
    print_performance_summary(cpu_time, gpu_time)

    # Ask if user wants sustained test
    print("\n" + "=" * 80)
    print("OPTIONAL: Sustained GPU Test")
    print("=" * 80)
    print("\nWould you like to run a sustained GPU test for monitoring?")
    print("This will run a larger computation (~30-60 seconds) so you can")
    print("watch GPU usage in nvidia-smi.")
    print("\nRun: watch -n 0.5 nvidia-smi")
    print("\nThen press Enter to continue, or Ctrl+C to skip...")

    try:
        input()
        run_sustained_gpu_test()
    except KeyboardInterrupt:
        print("\n\nSustained test skipped.")

    # Final summary
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 30 + "TEST COMPLETE" + " " * 35 + "║")
    print("╚" + "=" * 78 + "╝")

    if results_match:
        print("\n✓ ALL TESTS PASSED")
        return 0
    else:
        print("\n⚠ SOME ISSUES DETECTED")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
