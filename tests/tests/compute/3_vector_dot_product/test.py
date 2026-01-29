import time
import numpy as np
from kungfu import GPUMath, gpu_kernel, NP_GLTypes, IOTypes

@gpu_kernel({
    "vec_a": (NP_GLTypes.float, IOTypes.buffer),
    "vec_b": (NP_GLTypes.float, IOTypes.buffer),
    "n": (NP_GLTypes.int, IOTypes.uniform),
    "res": (NP_GLTypes.float, IOTypes.buffer)
})
def vector_dot_product_kernel(vec_a, vec_b, n):
    # Create temporary arrays
    temp_a : float[16]
    temp_b : float[16]
    
    # Fill temporary arrays with transformed values
    for i in range(n):
        temp_a[i] = vec_a * float(i + 1)
        temp_b[i] = vec_b * float(i + 1)
    
    # Compute dot product of temporary arrays
    dot_product : float = 0.0
    for i in range(n):
        dot_product = dot_product + (temp_a[i] * temp_b[i])
    
    # Apply normalization and return
    return dot_product / float(n)

def test(engine) -> bool:
    compiled_dot = engine.compile_fused(vector_dot_product_kernel, debug=False)
    
    vec_a = np.random.rand(5000).astype(np.float32) * 3.0
    vec_b = np.random.rand(5000).astype(np.float32) * 3.0
    n = 8
    
    result = compiled_dot(vec_a, vec_b, n)
    dot_output = engine.fetch(result)
    
    print(f"Sample output: {dot_output[:5]}")
    
    # CPU reference
    def vector_dot_product_cpu(a, b, n):
        output = np.zeros_like(a)
        for idx in range(len(a)):
            temp_a = np.zeros(n)
            temp_b = np.zeros(n)
            
            for i in range(n):
                temp_a[i] = a[idx] * float(i + 1)
                temp_b[i] = b[idx] * float(i + 1)
            
            dot_product = 0.0
            for i in range(n):
                dot_product += temp_a[i] * temp_b[i]
            
            output[idx] = dot_product / float(n)
        
        return output
    
    cpu_dot_output = vector_dot_product_cpu(vec_a, vec_b, n)
    print(f"CPU reference: {cpu_dot_output[:5]}")
    
    abs_diff_dot = np.abs(dot_output - cpu_dot_output)
    max_abs_diff_dot = np.max(abs_diff_dot)
    print(f"Maximum absolute difference: {max_abs_diff_dot:.6f}")
    
    return max_abs_diff_dot <= 0.001