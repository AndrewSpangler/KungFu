import time
import numpy as np
from kungfu import GPUMath, gpu_kernel, NP_GLTypes, IOTypes

@gpu_kernel({
    "a": (NP_GLTypes.float, IOTypes.buffer),
    "b": (NP_GLTypes.float, IOTypes.buffer),
    "res": (NP_GLTypes.float, IOTypes.buffer)
}, vectorized=True)
def squared_sum(a, b):
    return a * a + b * b

def test(engine) -> bool:
    compile_start = time.time()
    compiled = engine.compile_fused(squared_sum, debug=True)
    compile_end = time.time()
    
    print(f"Compilation time: {(compile_end - compile_start) * 1000:.2f} ms")

    # Create test data
    x = np.random.rand(10000).astype(np.float32)
    y = np.random.rand(10000).astype(np.float32)
    
    # GPU execution time
    gpu_start = time.time()
    result = compiled(x, y)
    gpu_output = engine.fetch(result)
    gpu_end = time.time()
    
    print(f"GPU execution + fetch time: {(gpu_end - gpu_start) * 1000:.2f} ms")

    # CPU execution time
    cpu_start = time.time()
    cpu_output = x*x + y*y
    cpu_end = time.time()
    
    print(f"CPU execution time: {(cpu_end - cpu_start) * 1000:.2f} ms")

    # Calculate percentage difference for basic kernel too
    abs_diff = np.abs(gpu_output - cpu_output)
    near_zero_mask = np.abs(cpu_output) < 1e-10
    percent_diff = np.zeros_like(gpu_output)
    
    not_near_zero = ~near_zero_mask
    if np.any(not_near_zero):
        percent_diff[not_near_zero] = 100.0 * abs_diff[not_near_zero] / np.abs(cpu_output[not_near_zero])
    
    if np.any(near_zero_mask):
        percent_diff[near_zero_mask] = abs_diff[near_zero_mask]
    
    max_percent_diff = np.max(percent_diff)
    max_abs_diff = np.max(abs_diff)
    
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Maximum percentage difference: {max_percent_diff:.6f}%")
    print(f"Sample GPU output: {gpu_output[:5]}")
    print(f"Sample CPU output: {cpu_output[:5]}")
    
    # Speedup calculation
    if cpu_end - cpu_start > 0:
        speedup = (cpu_end - cpu_start) / (gpu_end - gpu_start)
        print(f"Speedup (CPU/GPU): {speedup:.2f}x")

    return max_percent_diff <= 0.01