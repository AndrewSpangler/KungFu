import numpy as np
import time
import kungfu as kf
from kungfu import GPUMath, gpu_kernel, NP_GLTypes, IOTypes
from direct.showbase.ShowBase import ShowBase

app = ShowBase()
engine = kf.GPUMath(app, headless=True)

# Kernel versions of the operations
@gpu_kernel({
    "x": (NP_GLTypes.float, IOTypes.buffer),
    "res": (NP_GLTypes.float, IOTypes.buffer)
})
def complex_math_kernel(x):
    sin_x = sin(x)
    cos_x = cos(x)
    sqrt_x = sqrt(abs(x))
    x_squared = x * x
    return sin_x * cos_x + sqrt_x - x_squared

@gpu_kernel({
    "x": (NP_GLTypes.float, IOTypes.buffer),
    "y": (NP_GLTypes.float, IOTypes.buffer),
    "res": (NP_GLTypes.float, IOTypes.buffer)
})
def monte_carlo_kernel(x, y):
    return x * x + y * y

# Test different problem sizes
sizes = [10000, 100000, 1000000, 10000000]

print("=" * 80)
print("GPU Buffer vs Kernel vs NumPy Performance Comparison")
print("Computing: result = sin(x) * cos(x) + sqrt(abs(x)) - x^2")
print("=" * 80)

# Compile and warm up kernels
print("\nWarming up kernels...")
complex_math_compiled = engine.compile_fused(complex_math_kernel)
warmup_data = np.linspace(0, 10, 1000, dtype=np.float32)
_ = engine.fetch(complex_math_compiled(warmup_data))
print("Kernel warmup complete.\n")

for n in sizes:
    print(f"\nProblem size: {n:,} elements")
    print("-" * 40)
    
    # Generate test data
    x = np.linspace(0, 10, n, dtype=np.float32)
    
    # NumPy computation
    t0 = time.perf_counter()
    np_result = np.sin(x) * np.cos(x) + np.sqrt(np.abs(x)) - x**2
    np_time = time.perf_counter() - t0
    
    # GPU Buffer API computation
    t0 = time.perf_counter()
    handle = engine.sin(x)
    handle2 = engine.cos(x)
    handle = engine.mult(handle, handle2)
    
    handle3 = engine.abs(x)
    handle3 = engine.sqrt(handle3)
    handle = engine.add(handle, handle3)
    
    handle4 = engine.mult(x, x)
    handle = engine.sub(handle, handle4)
    gpu_buffer_time = time.perf_counter() - t0
    gpu_buffer_result = engine.fetch(handle)
    
    # GPU Kernel computation
    t0 = time.perf_counter()
    kernel_handle = complex_math_compiled(x)
    gpu_kernel_time = time.perf_counter() - t0
    gpu_kernel_result = engine.fetch(kernel_handle)
    
    # Verify correctness
    max_diff_buffer = np.max(np.abs(gpu_buffer_result - np_result))
    max_diff_kernel = np.max(np.abs(gpu_kernel_result - np_result))
    
    print(f"NumPy time:        {np_time*1000:.3f} ms")
    print(f"GPU Buffer time:   {gpu_buffer_time*1000:.3f} ms (speedup: {np_time/gpu_buffer_time:.2f}x)")
    print(f"GPU Kernel time:   {gpu_kernel_time*1000:.3f} ms (speedup: {np_time/gpu_kernel_time:.2f}x)")
    print(f"Kernel vs Buffer:  {gpu_buffer_time/gpu_kernel_time:.2f}x faster")
    print(f"Max diff (buffer): {max_diff_buffer:.2e}")
    print(f"Max diff (kernel): {max_diff_kernel:.2e}")

print("\n" + "=" * 80)
print("Monte Carlo Pi Estimation")
print("=" * 80)

n_samples = 10_000_000
print(f"\nEstimating π using {n_samples:,} random samples")
print("-" * 40)

# Compile and warm up Monte Carlo kernel
print("Warming up Monte Carlo kernel...")
monte_carlo_compiled = engine.compile_fused(monte_carlo_kernel)
warmup_x = np.random.uniform(-1, 1, 1000).astype(np.float32)
warmup_y = np.random.uniform(-1, 1, 1000).astype(np.float32)
_ = engine.fetch(monte_carlo_compiled(warmup_x, warmup_y))
print("Warmup complete.\n")

# Generate random data once
x_data = np.random.uniform(-1, 1, n_samples).astype(np.float32)
y_data = np.random.uniform(-1, 1, n_samples).astype(np.float32)

# NumPy version
t0 = time.perf_counter()
distances_np = x_data**2 + y_data**2
inside_np = np.sum(distances_np <= 1.0)
pi_estimate_np = 4.0 * inside_np / n_samples
np_time = time.perf_counter() - t0

# GPU Buffer version
t0 = time.perf_counter()
x_squared = engine.mult(x_data, x_data)
y_squared = engine.mult(y_data, y_data)
distances_buffer = engine.add(x_squared, y_squared)
dist_buffer_result = engine.fetch(distances_buffer)
gpu_buffer_time = time.perf_counter() - t0
inside_buffer = np.sum(dist_buffer_result <= 1.0)
pi_estimate_buffer = 4.0 * inside_buffer / n_samples

# GPU Kernel version
t0 = time.perf_counter()
distances_kernel = monte_carlo_compiled(x_data, y_data)
dist_kernel_result = engine.fetch(distances_kernel)
gpu_kernel_time = time.perf_counter() - t0
inside_kernel = np.sum(dist_kernel_result <= 1.0)
pi_estimate_kernel = 4.0 * inside_kernel / n_samples

print(f"NumPy π:       {pi_estimate_np:.6f} (time: {np_time*1000:.1f} ms)")
print(f"Buffer π:      {pi_estimate_buffer:.6f} (time: {gpu_buffer_time*1000:.1f} ms, speedup: {np_time/gpu_buffer_time:.2f}x)")
print(f"Kernel π:      {pi_estimate_kernel:.6f} (time: {gpu_kernel_time*1000:.1f} ms, speedup: {np_time/gpu_kernel_time:.2f}x)")
print(f"Actual π:      {np.pi:.6f}")
print(f"Kernel vs Buffer: {gpu_buffer_time/gpu_kernel_time:.2f}x faster")