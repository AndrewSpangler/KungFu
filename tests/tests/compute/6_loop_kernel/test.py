import time
import numpy as np
from kungfu import GPUMath, gpu_kernel, NP_GLTypes, IOTypes

@gpu_kernel({
    "data":         (NP_GLTypes.float, "buffer"),
    "scale":        (NP_GLTypes.float, "uniform"),
    "iterations":   (NP_GLTypes.int,   "uniform"),
    "res":          (NP_GLTypes.float, "buffer")
})
def uniform_loop_kernel(data, scale, iterations):
    # Example with loop and uniform
    result = data
    
    # GLSL for loop
    for i in range(iterations):
        result = result * scale + float(i)
    
    return result

def loop_py(data, scale, iterations):
    result = data.copy()
    for i in range(iterations):
        result = result * scale + i
    return result

def test(engine) -> bool:
    # Compile
    compiled = engine.compile_fused(uniform_loop_kernel, debug=True)
    
    # Test data
    data = np.random.rand(1000).astype(np.float32)
    scale = 1.5  # Single value
    iterations = 10  # Single value
    
    print("\n=== Testing Vector + Loop ===")
    # Run with uniform values
    result = compiled(data, scale, iterations)
    gpu_output = engine.fetch(result)
    
    print(f"Output shape: {gpu_output.shape}")
    print(f"Sample output: {gpu_output[:5]}")
    
    cpu_output = loop_py(data, scale, iterations)

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

    return max_percent_diff <= 0.001