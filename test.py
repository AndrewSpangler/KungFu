import numpy as np
import time
from direct.showbase.ShowBase import ShowBase
from gpu_math import GPUMath, gpu_kernel, inline_always, NP_GLTypes, IOTypes

class GraphCompilationDemo(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.math = GPUMath(self, headless=True)
    
    def example_complex_equation(self):
        print("\n=== Complex Kernel ===")
        
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

        # Compilation time
        compile_start = time.time()
        compiled = self.math.compile_fused(complex_equation, debug=True)
        compile_end = time.time()
        
        print(f"Compilation time: {(compile_end - compile_start) * 1000:.2f} ms")

        count = 1000000
        x = np.linspace(0, 10, count).astype(np.float32)
        y = np.linspace(0, 10, count).astype(np.float32)
        z = np.linspace(0, 10, count).astype(np.float32)

        # Warmup jit
        result = compiled(x, y, z)
        gpu_output = self.math.fetch(result)

        # GPU execution time
        gpu_start = time.time()
        result = compiled(x, y, z)
        
        gpu_end = time.time()

        gpu_output = self.math.fetch(result)
        
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

    def example_dynamic_loops_and_arrays(self):
        print("\n=== Dynamic Loops and Temporary Arrays (Simplified) ===")
        
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
        
        # Test moving average
        print("\n1. Moving Average Test:")
        compile_start = time.time()
        compiled_avg = self.math.compile_fused(moving_average_kernel, debug=False)
        compile_end = time.time()
        print(f"   Compilation time: {(compile_end - compile_start) * 1000:.2f} ms")
        
        data = np.random.rand(10000).astype(np.float32) * 10.0
        window_size = 5
        
        gpu_start = time.time()
        result = compiled_avg(data, window_size)
        gpu_output = self.math.fetch(result)
        gpu_end = time.time()
        
        print(f"   GPU execution: {(gpu_end - gpu_start) * 1000:.2f} ms")
        print(f"   Sample output: {gpu_output[:5]}")
        
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
        print(f"   CPU reference: {cpu_output[:5]}")
        
        abs_diff = np.abs(gpu_output - cpu_output)
        max_abs_diff = np.max(abs_diff)
        print(f"   Maximum absolute difference: {max_abs_diff:.6f}")
        
        # Test vector dot product
        print("\n2. Vector Dot Product Test:")
        compiled_dot = self.math.compile_fused(vector_dot_product_kernel, debug=False)
        
        vec_a = np.random.rand(5000).astype(np.float32) * 3.0
        vec_b = np.random.rand(5000).astype(np.float32) * 3.0
        n = 8
        
        result = compiled_dot(vec_a, vec_b, n)
        dot_output = self.math.fetch(result)
        
        print(f"   Sample output: {dot_output[:5]}")
        
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
        print(f"   CPU reference: {cpu_dot_output[:5]}")
        
        abs_diff_dot = np.abs(dot_output - cpu_dot_output)
        max_abs_diff_dot = np.max(abs_diff_dot)
        print(f"   Maximum absolute difference: {max_abs_diff_dot:.6f}")
        
        # Test matrix operations
        print("\n3. Matrix Operations Test:")
        compiled_mat = self.math.compile_fused(matrix_operations_kernel, debug=True)
        
        mat_a = np.random.rand(2000).astype(np.float32) * 3.0
        mat_b = np.random.rand(2000).astype(np.float32) * 3.0
        size = 4
        
        result = compiled_mat(mat_a, mat_b, size)
        mat_output = self.math.fetch(result)
        
        print(f"   Sample output: {mat_output[:5]}")
        
        # CPU reference
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
        
        cpu_mat_output = matrix_operations_cpu(mat_a, mat_b, size)
        print(f"   CPU reference: {cpu_mat_output[:5]}")
        
        abs_diff_mat = np.abs(mat_output - cpu_mat_output)
        max_abs_diff_mat = np.max(abs_diff_mat)
        print(f"   Maximum absolute difference: {max_abs_diff_mat:.6f}")
        
        print("\n=== Summary ===")
        print("Moving average:   ", "✓" if max_abs_diff < 0.001 else "✗")
        print("Vector dot product:", "✓" if max_abs_diff_dot < 0.001 else "✗")
        print("Matrix operations: ", "✓" if max_abs_diff_mat < 0.001 else "✗")

    def example_basic_kernel(self):
        print("\n=== Basic Kernel ===")
        
        @gpu_kernel
        def squared_sum(a, b):
            return a * a + b * b
        
        # Compilation time
        compile_start = time.time()
        compiled = self.math.compile_fused(squared_sum, debug=True)
        compile_end = time.time()
        
        print(f"Compilation time: {(compile_end - compile_start) * 1000:.2f} ms")

        # Create test data
        x = np.random.rand(10000).astype(np.float32)
        y = np.random.rand(10000).astype(np.float32)
        
        # GPU execution time
        gpu_start = time.time()
        result = compiled(x, y)
        gpu_output = self.math.fetch(result)
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
    
    def example_uniforms_and_loops(self):
        print("\n=== Uniforms and Loops Example ===")
        
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
        
        # Compile
        compiled = self.math.compile_fused(uniform_loop_kernel, debug=True)
        
        # Test data
        data = np.random.rand(1000).astype(np.float32)
        scale = 1.5  # Single value
        iterations = 10  # Single value
        
        # Run with uniform values
        result = compiled(data, scale, iterations)
        gpu_output = self.math.fetch(result)
        
        print(f"Output shape: {gpu_output.shape}")
        print(f"Sample output: {gpu_output[:5]}")
        
        # Test with pre-allocated output buffer
        print("\n=== Pre-allocated Output Buffer ===")
        output_buffer = np.empty_like(data)
        result = compiled(data, scale, iterations, out=output_buffer)
        print(f"Output buffer filled: {output_buffer[:5]}")
    
    def example_void_function(self):
        print("\n=== Void Function Example ===")
        
        @gpu_kernel({
            "input_data":   (NP_GLTypes.float, IOTypes.buffer),
            "output_data":  (NP_GLTypes.float, IOTypes.buffer),
            "multiplier":   (NP_GLTypes.float, IOTypes.uniform),
        })
        def inplace_multiply(input_data, output_data, multiplier):
            # Void function that writes to output buffer
            # Using a loop
            for i in range(10):  # Fixed iteration loop
                output_data = input_data * multiplier * float(i + 1)
        
        # Compile
        compiled = self.math.compile_fused(inplace_multiply, debug=True)
        
        # Test data
        input_data = np.random.rand(1000).astype(np.float32)
        output_data = np.empty_like(input_data)
        multiplier = 2.5
        
        # Run void function
        result = compiled(input_data, output_data, multiplier, out=output_data)
        print(f"Result (should be None for void function): {result}")
        print(f"Output data modified: {output_data[:5]}")
    
    def run_all_examples(self):
        """Run all examples"""
        total_start = time.time()
        
        self.example_basic_kernel()
        self.example_complex_equation()
        self.example_uniforms_and_loops()
        self.example_void_function()
        self.example_dynamic_loops_and_arrays()
                
        total_end = time.time()
        
        print(f"\n=== All Examples Complete ===")
        print(f"Total execution time: {(total_end - total_start) * 1000:.2f} ms")

def main():
    """Run demo"""
    demo = GraphCompilationDemo()
    demo.run_all_examples()
    demo.destroy()

if __name__ == "__main__":
    main()