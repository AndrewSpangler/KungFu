import time
import numpy as np
from kungfu import GPUMath, gpu_kernel, NP_GLTypes, IOTypes

@gpu_kernel({
    "a"     :    (NP_GLTypes.float, IOTypes.buffer),
    "b"     :    (NP_GLTypes.float, IOTypes.buffer),
    "c"     :    (NP_GLTypes.float, IOTypes.buffer),
    "res"   :    (NP_GLTypes.float, IOTypes.buffer)
})
def complex_equation(a, b, c):
    @inline_always
    def add_two_inline(f, g) -> float:
        return f + g

    def add_two(f, g) -> float:
        return f + g

    def half_val(f) -> float:
        return f / 2

    a2   : int   = int(a)
    a3   : float = a2 * b
    c2   : float = cos(a + c)
    a3   : float = a3 ** add_two(b, 1)
    temp : float = 0.0
    for i in range(3):
        for j in range(4):
            temp = half_val(temp + (i + j))

    res = add_two_inline(add_two(a3, temp), b * c2)
    return res

def test(engine) -> bool:
    def complex_equation_py(a, b, c):
        def add_two(f, g):
            return f + g

        a2 = a.astype(np.int32)
        a2 = a2 * b
        c2 = np.cos(a + c)
        a2 = a2 ** add_two(b, 1)

        temp = np.zeros(a.shape, dtype=np.float32)
        for i in range(3):
            for j in range(4):
                temp = (temp + (i + j)) / 2

        res = add_two(add_two(a2, temp), b * c2)
        return res

    compile_start = time.time()
    compiled = engine.compile_fused(complex_equation, debug=True)
    compile_end = time.time()
    
    print(f"Compilation time: {(compile_end - compile_start) * 1000:.2f} ms")

    count = 1000000
    x = np.linspace(0, 10, count).astype(np.float32)
    y = np.linspace(0, 10, count).astype(np.float32)
    z = np.linspace(0, 10, count).astype(np.float32)

    # Warmup jit
    result = compiled(x, y, z)
    gpu_output = engine.fetch(result)

    # GPU execution time
    gpu_start = time.time()
    result = compiled(x, y, z)
    
    gpu_end = time.time()

    gpu_output = engine.fetch(result)
    
    print(f"GPU execution {(gpu_end - gpu_start) * 1000:.2f} ms")

    # CPU execution time
    cpu_start = time.time()
    cpu_output = complex_equation_py(x, y, z)
    cpu_end = time.time()
    
    print(f"CPU execution time: {(cpu_end - cpu_start) * 1000:.2f} ms")

    print(f"Sample GPU output: {gpu_output[300:310]}")
    print(f"Sample CPU output: {cpu_output[300:310]}")
    
    # Verify correctness with percentage difference
    # Avoid division by zero by handling near-zero values
    abs_diff = np.abs(gpu_output - cpu_output)
    
    # Use absolute difference when reference is near zero
    near_zero_mask = np.abs(cpu_output) < 1e-10
    percent_diff = np.zeros_like(gpu_output)
    
    # For non-zero values, calculate percentage difference
    not_near_zero = ~near_zero_mask
    if np.any(not_near_zero):
        percent_diff[not_near_zero] = 100.0 * abs_diff[not_near_zero] / np.abs(cpu_output[not_near_zero])
    
    # For near-zero values, use absolute difference as percentage would be misleading
    if np.any(near_zero_mask):
        percent_diff[near_zero_mask] = abs_diff[near_zero_mask]

    max_percent_diff = np.max(percent_diff)
    
    # Also calculate max absolute difference for context
    max_abs_diff = np.max(abs_diff)
    
    print(f"Maximum absolute difference: {max_abs_diff:.2e}")
    print(f"Maximum percentage difference: {max_percent_diff:.6f}%")
    
    # Speedup calculation
    if cpu_end - cpu_start > 0:
        speedup = (cpu_end - cpu_start) / (gpu_end - gpu_start)
        print(f"Speedup (CPU/GPU): {speedup:.2f}x")
    passed = (max_percent_diff <= 0.001)
    print(passed)
    return passed