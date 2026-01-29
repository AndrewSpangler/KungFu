import time
import numpy as np
from kungfu import GPUMath, gpu_kernel, NP_GLTypes, IOTypes

@gpu_kernel({
    "input_data": (NP_GLTypes.float, IOTypes.buffer),
    "window_size": (NP_GLTypes.int, IOTypes.uniform),
    "res": (NP_GLTypes.float, IOTypes.buffer)
})
def moving_average_kernel(input_data, window_size):
    # Create a temporary buffer for the window
    window : float[32]  # Max window size of 32
    
    # Fill window with data from input_data with offset
    for i in range(window_size):
        window[i] = input_data * float(i + 1)
    
    # Compute weighted average using dynamic loop
    sum_val : float = 0.0
    for i in range(window_size):
        weight = 1.0 / float(i + 1)
        sum_val = sum_val + (window[i] * weight)
    
    avg = sum_val / float(window_size)
    return avg

def test(engine) -> bool:
    print("\n\nMoving Average Test:")
    compile_start = time.time()
    compiled_avg = engine.compile_fused(moving_average_kernel, debug=False)
    compile_end = time.time()
    print(f"Compilation time: {(compile_end - compile_start) * 1000:.2f} ms")
    
    data = np.random.rand(10000).astype(np.float32) * 10.0
    window_size = 5
    
    gpu_start = time.time()
    result = compiled_avg(data, window_size)
    gpu_output = engine.fetch(result)
    gpu_end = time.time()
    
    print(f"GPU execution: {(gpu_end - gpu_start) * 1000:.2f} ms")
    print(f"Sample output: {gpu_output[:5]}")
    
    # CPU reference
    def moving_average_cpu(input_data, window_size):
        output = np.zeros_like(input_data)
        for idx in range(len(input_data)):
            weighted_sum = 0.0
            for i in range(window_size):
                weight = 1.0 / float(i + 1)
                weighted_sum += input_data[idx] * float(i + 1) * weight
            output[idx] = weighted_sum / float(window_size)
        return output
    
    cpu_output = moving_average_cpu(data, window_size)
    print(f"CPU reference: {cpu_output[:5]}")
    
    abs_diff = np.abs(gpu_output - cpu_output)
    max_abs_diff = np.max(abs_diff)
    print(f"Maximum absolute difference: {max_abs_diff:.6f}")
    passed = max_abs_diff <= 0.001
    return passed