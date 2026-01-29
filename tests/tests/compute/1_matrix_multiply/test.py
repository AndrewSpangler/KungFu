import numpy as np
import time
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer, 
    GeomEnums, ComputeNode, load_prc_file_data,
    GraphicsPipeSelection, FrameBufferProperties,
    WindowProperties, GraphicsPipe, Vec2
)
from kungfu import (
    GPUMath,
    gpu_kernel,
    inline_always,
    CastBuffer,
    static_constant,
    NP_GLTypes,
    Vec_GLTypes,
    GLTypes,
    IOTypes
)

# ==================== MATRIX MULTIPLICATION ====================



# # ==================== BITONIC SORT ====================

# @gpu_kernel({
#     "data":      (NP_GLTypes.float, IOTypes.array),
#     "stage":     (NP_GLTypes.uint,  IOTypes.uniform),
#     "step":      (NP_GLTypes.uint,  IOTypes.uniform),
#     "nItems":    (NP_GLTypes.uint,  IOTypes.uniform)
# }, vectorized=True)
# def bitonic_sort_kernel(
#     data:  IOTypes.array,
#     stage: IOTypes.uniform,
#     step:  IOTypes.uniform
# ) -> GLTypes.void:
#     gidx : uint = gl_GlobalInvocationID.x
    
#     pair_dist : uint = uint(1) << (step - uint(1))
#     block_size : uint = uint(2) * pair_dist
    
#     gidx_mod : uint = gidx - (gidx / pair_dist) * pair_dist
#     gidx_quot : uint = gidx // pair_dist
#     left_idx : uint = gidx_mod + gidx_quot * block_size
#     right_idx : uint = left_idx + pair_dist
    
#     left_elem : float = data[left_idx]
#     right_elem : float = data[right_idx]
    
#     stage_size : uint = uint(1) << stage
#     stage_quot : uint = left_idx // stage_size
#     stage_mod : uint = stage_quot - (stage_quot // uint(2)) * uint(2)

#     if bool((left_elem > right_elem) == (stage_mod == uint(0))):
#         data[left_idx] = right_elem
#         data[right_idx] = left_elem

# class BitonicSort:
#     def __init__(self, math_engine: GPUMath):
#         self.engine = math_engine
#         self.sort_kernel = self.engine.compile_fused(bitonic_sort_kernel, debug=True)
    
#     def sort(self, data):
#         if isinstance(data, CastBuffer):
#             buf = data
#             data_copy = self.engine.fetch(buf)
#         else:
#             data_copy = data.copy().astype(np.float32)
#             buf = self.engine.push(data_copy)
        
#         n = len(buf)
#         if n & (n - 1) != 0:
#             raise ValueError("Array size must be power of 2")
        
#         num_stages = int(np.log2(n))
        
#         for stage in range(num_stages):
#             for step in range(stage + 1, 0, -1):
#                 num_threads = n // 2
#                 self.sort_kernel(buf, int(stage), int(step), n_items=num_threads)
        
#         return self.engine.fetch(buf)

# ==================== DEMO & BENCHMARKING ====================

class AlgorithmsDemo(ShowBase):
    def __init__(self, headless=True):
        ShowBase.__init__(self)
        self.engine = GPUMath(self, headless=True)
        self.matmul = MatrixMultiply(self.engine)
        # self.bitonic = BitonicSort(self.engine)
    
    def test_prefix_sum(self):
        print("\n" + "="*60)
        print("TESTING PREFIX SUM (SCAN)")
        print("="*60)
        
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        
        t0 = time.perf_counter()
        result_gpu = self.engine.fetch(self.prefix_sum.scan(data))
        t_gpu = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        result_cpu = np.cumsum(data)
        t_cpu = time.perf_counter() - t1
        
        print(f"Input:  {data}")
        print(f"GPU:    {result_gpu}")
        print(f"CPU:    {result_cpu}")
        print(f"Match:  {np.allclose(result_gpu, result_cpu)}")
        print(f"GPU Time: {t_gpu:.6f}s | CPU Time: {t_cpu:.6f}s")
    
    def test_matmul(self):
        print("\n" + "="*60)
        print("TESTING MATRIX MULTIPLICATION")
        print("="*60)
        
        A = np.random.randn(64, 64).astype(np.float32)
        B = np.random.randn(64, 64).astype(np.float32)
        
        t0 = time.perf_counter()
        result_gpu = self.matmul.multiply(A, B)
        t_gpu = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        result_cpu = A @ B
        t_cpu = time.perf_counter() - t1
        
        max_diff = np.max(np.abs(result_gpu - result_cpu))
        print(f"Matrix Size: {A.shape} x {B.shape}")
        print(f"Max Difference: {max_diff:.3e}")
        print(f"Match: {np.allclose(result_gpu, result_cpu, rtol=1e-4)}")
        print(f"GPU Time: {t_gpu:.6f}s | CPU Time: {t_cpu:.6f}s | Speedup: {t_cpu/t_gpu:.2f}x")
    
    def test_bitonic_sort(self):
        print("\n" + "="*60)
        print("TESTING BITONIC SORT")
        print("="*60)
        
        data = np.random.randn(256).astype(np.float32)
        
        t0 = time.perf_counter()
        result_gpu = self.bitonic.sort(data)
        t_gpu = time.perf_counter() - t0
        
        t1 = time.perf_counter()
        result_cpu = np.sort(data)
        t_cpu = time.perf_counter() - t1
        
        print(f"Data Size: {len(data)}")
        print(f"First 10 GPU: {result_gpu[:10]}")
        print(f"First 10 CPU: {result_cpu[:10]}")
        print(f"Match: {np.allclose(result_gpu, result_cpu)}")
        print(f"GPU Time: {t_gpu:.6f}s | CPU Time: {t_cpu:.6f}s")
    
    def run_all_tests(self):
        self.test_matmul()
        # self.test_bitonic_sort()







from kungfu import gpu_kernel, GPUMath, IOTypes, NP_GLTypes





@gpu_kernel({
    "A":      (NP_GLTypes.float, IOTypes.array),
    "B":      (NP_GLTypes.float, IOTypes.array),
    "C":      (NP_GLTypes.float, IOTypes.array),
    "M":      (NP_GLTypes.uint,  IOTypes.uniform),
    "N":      (NP_GLTypes.uint,  IOTypes.uniform),
    "K":      (NP_GLTypes.uint,  IOTypes.uniform),
    "nItems": (NP_GLTypes.uint,  IOTypes.uniform),
}, vectorized=False)
def matmul_kernel(
    A: IOTypes.array,
    B: IOTypes.array,
    C: IOTypes.array,
    M: IOTypes.uniform,
    N: IOTypes.uniform,
    K: IOTypes.uniform,
    nItems: IOTypes.uniform
) -> GLTypes.void:
    gidx : uint = gl_GlobalInvocationID.x
    
    if gidx >= nItems:
        return
    
    row : uint = gidx // N
    col : uint = gidx - row * N
    
    sum_val : float = 0.0
    for i in range(K):
        a_idx : uint = row * K + i
        b_idx : uint = i * N + col
        sum_val = sum_val + A[a_idx] * B[b_idx]
    
    C[gidx] = sum_val

class MatrixMultiply:
    def __init__(self, math_engine: GPUMath):
        self.engine = math_engine
        self.matmul = self.engine.compile_fused(matmul_kernel, debug=True)
    
    def multiply(self, A, B):
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise ValueError("Inputs must be 2D matrices")
        
        M, K1 = A.shape
        K2, N = B.shape
        
        if K1 != K2:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} x {B.shape}")
        
        K = K1
        
        A_flat = self.engine.push(A.flatten().astype(np.float32))
        B_flat = self.engine.push(B.flatten().astype(np.float32))
        C_flat = self.engine.push(np.zeros(M * N, dtype=np.float32))
        
        self.matmul(A_flat, B_flat, C_flat, int(M), int(N), int(K), int(M * N), n_items=M * N)
        
        result = self.engine.fetch(C_flat)
        return result.reshape(M, N)

def test(engine) -> bool:
    print("TESTING MATRIX MULTIPLY")
    matmul = MatrixMultiply(engine)

    A = np.random.randn(64, 64).astype(np.float32)
    B = np.random.randn(64, 64).astype(np.float32)
    
    t0 = time.perf_counter()
    result_gpu = matmul.multiply(A, B)
    t_gpu = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    result_cpu = A @ B
    t_cpu = time.perf_counter() - t1
    
    max_diff = np.max(np.abs(result_gpu - result_cpu))

    match = np.allclose(result_gpu, result_cpu, rtol=1e-3)
    print(f"Matrix Size: {A.shape} x {B.shape}")
    print(f"Max Difference: {max_diff:.3e}")
    print(f"Match: {match}")
    print(f"GPU Time: {t_gpu:.6f}s | CPU Time: {t_cpu:.6f}s | Speedup: {t_cpu/t_gpu:.2f}x")
    
    passed = max_diff <= 0.001

    return passed