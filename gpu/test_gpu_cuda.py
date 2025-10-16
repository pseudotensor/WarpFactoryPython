#!/usr/bin/env python3
"""
GPU and CUDA Test Script for WarpFactory
Tests CuPy, CUDA availability, and basic GPU computations
"""

import sys
import time

def test_cupy_import():
    """Test if CuPy can be imported"""
    print("=" * 70)
    print("TEST 1: CuPy Import")
    print("=" * 70)
    try:
        import cupy as cp
        print("✓ CuPy imported successfully")
        print(f"  CuPy version: {cp.__version__}")
        return True, cp
    except ImportError as e:
        print(f"✗ Failed to import CuPy: {e}")
        return False, None
    except Exception as e:
        print(f"✗ Unexpected error importing CuPy: {e}")
        return False, None


def test_cuda_available(cp):
    """Test if CUDA is available"""
    print("\n" + "=" * 70)
    print("TEST 2: CUDA Availability")
    print("=" * 70)
    try:
        available = cp.cuda.is_available()
        if available:
            print("✓ CUDA is available")
            return True
        else:
            print("✗ CUDA is not available")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA availability: {e}")
        return False


def test_gpu_count(cp):
    """Test GPU device count"""
    print("\n" + "=" * 70)
    print("TEST 3: GPU Device Count")
    print("=" * 70)
    try:
        count = cp.cuda.runtime.getDeviceCount()
        print(f"✓ Found {count} GPU device(s)")
        return True, count
    except Exception as e:
        print(f"✗ Error getting GPU count: {e}")
        return False, 0


def test_gpu_properties(cp, gpu_count):
    """Test GPU device properties"""
    print("\n" + "=" * 70)
    print("TEST 4: GPU Device Properties")
    print("=" * 70)
    all_success = True
    for i in range(gpu_count):
        try:
            device = cp.cuda.Device(i)
            props = device.attributes
            compute_cap = device.compute_capability
            print(f"\n  GPU {i}:")
            print(f"    Name: {props.get('Name', 'Unknown')}")
            print(f"    Compute Capability: {compute_cap}")
            print(f"    Total Memory: {device.mem_info[1] / (1024**3):.2f} GB")
            print(f"    Free Memory: {device.mem_info[0] / (1024**3):.2f} GB")
            print(f"    Multiprocessors: {props.get('MultiProcessorCount', 'Unknown')}")
        except Exception as e:
            print(f"  ✗ Error getting properties for GPU {i}: {e}")
            all_success = False

    if all_success:
        print("\n✓ Successfully retrieved properties for all GPUs")
    return all_success


def test_simple_gpu_computation(cp):
    """Test simple GPU computation"""
    print("\n" + "=" * 70)
    print("TEST 5: Simple GPU Array Operations")
    print("=" * 70)
    try:
        # Create arrays on GPU
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([10, 20, 30, 40, 50])

        # Perform operations
        c = a + b
        d = a * b

        print("✓ GPU array creation successful")
        print(f"  a + b = {c.get()}")
        print(f"  a * b = {d.get()}")
        return True
    except Exception as e:
        print(f"✗ GPU computation failed: {e}")
        return False


def test_gpu_matrix_multiply(cp):
    """Test GPU matrix multiplication with timing"""
    print("\n" + "=" * 70)
    print("TEST 6: GPU Matrix Multiplication Performance")
    print("=" * 70)

    sizes = [100, 500, 1000]
    all_success = True

    for size in sizes:
        try:
            # Create random matrices on GPU
            a = cp.random.rand(size, size, dtype=cp.float32)
            b = cp.random.rand(size, size, dtype=cp.float32)

            # Warm up
            _ = cp.dot(a, b)
            cp.cuda.Stream.null.synchronize()

            # Time the computation
            start = time.time()
            c = cp.dot(a, b)
            cp.cuda.Stream.null.synchronize()
            elapsed = time.time() - start

            print(f"  ✓ Matrix {size}x{size}: {elapsed*1000:.2f} ms")

        except Exception as e:
            print(f"  ✗ Matrix {size}x{size} failed: {e}")
            all_success = False

    if all_success:
        print("\n✓ All matrix multiplication tests passed")
    return all_success


def test_gpu_memory_operations(cp):
    """Test GPU memory allocation and transfer"""
    print("\n" + "=" * 70)
    print("TEST 7: GPU Memory Operations")
    print("=" * 70)
    try:
        # Test memory allocation
        size_mb = 100
        elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per float32

        # Allocate on GPU
        gpu_array = cp.zeros(elements, dtype=cp.float32)
        print(f"✓ Allocated {size_mb} MB on GPU")

        # Transfer CPU -> GPU
        import numpy as np
        cpu_array = np.random.rand(1000, 1000).astype(np.float32)
        gpu_from_cpu = cp.asarray(cpu_array)
        print(f"✓ Transferred array from CPU to GPU (shape: {gpu_from_cpu.shape})")

        # Transfer GPU -> CPU
        back_to_cpu = cp.asnumpy(gpu_from_cpu)
        print(f"✓ Transferred array from GPU to CPU (shape: {back_to_cpu.shape})")

        # Verify data integrity
        if np.allclose(cpu_array, back_to_cpu):
            print("✓ Data integrity verified (CPU->GPU->CPU)")
        else:
            print("✗ Data integrity check failed")
            return False

        return True
    except Exception as e:
        print(f"✗ Memory operations failed: {e}")
        return False


def test_cuda_kernel_compilation(cp):
    """Test CUDA kernel compilation"""
    print("\n" + "=" * 70)
    print("TEST 8: CUDA Kernel Compilation")
    print("=" * 70)
    try:
        # Simple element-wise kernel
        kernel_code = '''
        extern "C" __global__
        void add_kernel(const float* a, const float* b, float* c, int n) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] + b[idx];
            }
        }
        '''

        # Compile kernel
        add_kernel = cp.RawKernel(kernel_code, 'add_kernel')
        print("✓ CUDA kernel compiled successfully")

        # Test kernel execution
        n = 1000
        a = cp.random.rand(n, dtype=cp.float32)
        b = cp.random.rand(n, dtype=cp.float32)
        c = cp.zeros(n, dtype=cp.float32)

        threads_per_block = 256
        blocks = (n + threads_per_block - 1) // threads_per_block

        add_kernel((blocks,), (threads_per_block,), (a, b, c, n))
        cp.cuda.Stream.null.synchronize()

        # Verify result
        expected = a + b
        if cp.allclose(c, expected):
            print("✓ CUDA kernel execution successful and verified")
            return True
        else:
            print("✗ CUDA kernel result verification failed")
            return False

    except Exception as e:
        print(f"✗ CUDA kernel compilation/execution failed: {e}")
        return False


def test_multi_gpu(cp, gpu_count):
    """Test multi-GPU capability"""
    print("\n" + "=" * 70)
    print("TEST 9: Multi-GPU Operations")
    print("=" * 70)

    if gpu_count < 2:
        print(f"⊘ Skipping multi-GPU test (only {gpu_count} GPU available)")
        return True

    try:
        size = 1000
        results = []

        for i in range(min(gpu_count, 4)):  # Test up to 4 GPUs
            with cp.cuda.Device(i):
                a = cp.random.rand(size, size, dtype=cp.float32)
                b = cp.random.rand(size, size, dtype=cp.float32)
                c = cp.dot(a, b)
                cp.cuda.Stream.null.synchronize()
                results.append(c.shape)
                print(f"  ✓ GPU {i}: Computation successful (result shape: {c.shape})")

        print(f"\n✓ Successfully ran computations on {len(results)} GPUs")
        return True

    except Exception as e:
        print(f"✗ Multi-GPU test failed: {e}")
        return False


def main():
    """Run all GPU/CUDA tests"""
    print("\n" + "=" * 70)
    print("WarpFactory GPU/CUDA Test Suite")
    print("=" * 70)

    results = {}

    # Test 1: Import CuPy
    success, cp = test_cupy_import()
    results['cupy_import'] = success
    if not success:
        print("\n❌ FATAL: Cannot proceed without CuPy")
        sys.exit(1)

    # Test 2: CUDA availability
    results['cuda_available'] = test_cuda_available(cp)
    if not results['cuda_available']:
        print("\n❌ FATAL: CUDA not available")
        sys.exit(1)

    # Test 3: GPU count
    success, gpu_count = test_gpu_count(cp)
    results['gpu_count'] = success
    if not success or gpu_count == 0:
        print("\n❌ FATAL: No GPUs detected")
        sys.exit(1)

    # Test 4: GPU properties
    results['gpu_properties'] = test_gpu_properties(cp, gpu_count)

    # Test 5: Simple computation
    results['simple_computation'] = test_simple_gpu_computation(cp)

    # Test 6: Matrix multiplication
    results['matrix_multiply'] = test_gpu_matrix_multiply(cp)

    # Test 7: Memory operations
    results['memory_operations'] = test_gpu_memory_operations(cp)

    # Test 8: Kernel compilation
    results['kernel_compilation'] = test_cuda_kernel_compilation(cp)

    # Test 9: Multi-GPU
    results['multi_gpu'] = test_multi_gpu(cp, gpu_count)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print("\n" + "=" * 70)
    print(f"Result: {passed_tests}/{total_tests} tests passed")
    print("=" * 70)

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! GPU/CUDA is working correctly.")
        sys.exit(0)
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
