import time
import numpy as np
from kungfu import GPUMath, gpu_kernel, NP_GLTypes, IOTypes

@gpu_kernel({
    "matrix_a": (NP_GLTypes.float, IOTypes.buffer),
    "matrix_b": (NP_GLTypes.float, IOTypes.buffer),
    "size": (NP_GLTypes.int, IOTypes.uniform),
    "res": (NP_GLTypes.float, IOTypes.buffer)
})
def matrix_operations_kernel(matrix_a, matrix_b, size):
    # Create 2D array
    temp_result : float[8, 8]
    
    # Fill 2D array with computed values
    for i in range(size):
        for j in range(size):
            value = matrix_a * float(i + 1) + matrix_b * float(j + 1)
            temp_result[i][j] = value
    
    # Compute row-wise sums
    row_sums : float[8]
    for i in range(size):
        row_sum : float = 0.0
        for j in range(size):
            row_sum = row_sum + temp_result[i][j]
        row_sums[i] = row_sum
    
    # Compute average of row sums
    total_sum : float = 0.0
    for i in range(size):
        total_sum = total_sum + row_sums[i]
    
    return total_sum / float(size)

def matrix_operations_cpu(a, b, size):
    output = np.zeros_like(a)
    for idx in range(len(a)):
        temp_result = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                temp_result[i, j] = a[idx] * float(i + 1) + b[idx] * float(j + 1)
        
        row_sums = np.zeros(size)
        for i in range(size):
            row_sum = 0.0
            for j in range(size):
                row_sum += temp_result[i, j]
            row_sums[i] = row_sum
        
        total_sum = 0.0
        for i in range(size):
            total_sum += row_sums[i]
        
        output[idx] = total_sum / float(size)
    
    return output

def test(engine) -> bool:
    print("Matrix Operations Test:")
    compiled_mat = engine.compile_fused(matrix_operations_kernel, debug=True)
    
    mat_a = np.random.rand(2000).astype(np.float32) * 3.0
    mat_b = np.random.rand(2000).astype(np.float32) * 3.0
    size = 4
    
    result = compiled_mat(mat_a, mat_b, size)
    mat_output = engine.fetch(result)
    
    print(f"Sample output: {mat_output[:5]}")
    cpu_mat_output = matrix_operations_cpu(mat_a, mat_b, size)
    print(f"CPU reference: {cpu_mat_output[:5]}")
    
    abs_diff_mat = np.abs(mat_output - cpu_mat_output)
    max_abs_diff_mat = np.max(abs_diff_mat)
    print(f"Maximum absolute difference: {max_abs_diff_mat:.6f}")
    
    return max_abs_diff_mat <= 0.00001