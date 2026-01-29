import numpy as np
import time
import math
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

"""
Generated GLSL code:
#version 430
layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uniform int inverse;
uniform uint nItems;
uniform uint stage;
layout(std430, binding = 0) buffer D0 { vec2 data_in[]; };
layout(std430, binding = 1) buffer D1 { vec2 data_out[]; };

void main() {

        if((gl_GlobalInvocationID.x >= nItems)) {
                return;
        }
        int gid_x_int = int(gl_GlobalInvocationID.x);
        int _t3 = (int(stage) + 1);
        int block_size = (1 << _t3);
        int half_block = (1 << int(stage));
        float block_idx = (gid_x_int / int(half_block));
        int in_block_idx = (gid_x_int % int(half_block));
        uint _t12 = (int(block_idx) * int(block_size));
        uint idx1 = (int(_t12) + int(in_block_idx));
        uint idx2 = (int(idx1) + int(half_block));
        vec2 x1 = data_in[int(idx1)];
        vec2 x2 = data_in[int(idx2)];
        float PI = 3.141592653589793;
        float _t22 = (-2.0);
        float _t23 = (_t22 * PI);
        float _t24 = float(in_block_idx);
        float _t25 = (_t23 * _t24);
        float _t26 = float(block_size);
        float angle = (_t25 / _t26);
        bool _t29 = (inverse == 1);
        if(_t29) {
                angle = (-angle);
        }
        float tw_real = cos(angle);
        float tw_imag = sin(angle);
        float _t36 = x2.x;
        float _t37 = (_t36 * tw_real);
        float _t38 = x2.y;
        float _t39 = (_t38 * tw_imag);
        float temp_real = (_t37 - _t39);
        float _t42 = x2.x;
        float _t43 = (_t42 * tw_imag);
        float _t44 = x2.y;
        float _t45 = (_t44 * tw_real);
        float temp_imag = (_t43 + _t45);
        float _t48 = x1.x;
        float _t49 = (_t48 + temp_real);
        float _t50 = x1.y;
        float _t51 = (_t50 + temp_imag);
        vec2 _t52 = vec2(_t49, _t51);
        data_out[idx1] = _t52;
        float _t53 = x1.x;
        float _t54 = (_t53 - temp_real);
        float _t55 = x1.y;
        float _t56 = (_t55 - temp_imag);
        vec2 _t57 = vec2(_t54, _t56);
        data_out[idx2] = _t57;
}
"""
@gpu_kernel({
    "data_in": (Vec_GLTypes.vec2, IOTypes.array),
    "data_out": (Vec_GLTypes.vec2, IOTypes.array),
    "nItems": (NP_GLTypes.uint, IOTypes.uniform),
    "stage": (NP_GLTypes.uint, IOTypes.uniform),
    "inverse": (NP_GLTypes.int, IOTypes.uniform),
}, vectorized=False)
def radix2_butterfly(
    data_in: IOTypes.array,
    data_out: IOTypes.array,
    nItems: IOTypes.uniform,
    stage: IOTypes.uniform,
    inverse: IOTypes.uniform
) -> GLTypes.void: 
    if gl_GlobalInvocationID.x >= nItems:
        return
    
    gid_x_int: int = int(gl_GlobalInvocationID.x)
    
    # Calculate butterfly parameters
    block_size: uint = 1 << (stage + 1)  # 2^(stage+1)
    half_block: uint = 1 << stage        # 2^stage
    
    # Which block and position within block
    block_idx: uint = gid_x_int / half_block
    in_block_idx: uint = gid_x_int % half_block
    
    # Input indices
    idx1: uint = block_idx * block_size + in_block_idx
    idx2: uint = idx1 + half_block
    
    # Load input values
    x1: vec2 = data_in[idx1]
    x2: vec2 = data_in[idx2]
    
    # Calculate twiddle factor
    PI: float = 3.14159265358979323846
    angle: float = -2.0 * PI * float(in_block_idx) / float(block_size)
    
    # Handle inverse FFT
    if inverse == 1:
        angle = -angle  # Inverse FFT uses positive exponent
    
    # Twiddle factor W = exp(i*angle) = cos(angle) + i*sin(angle)
    tw_real: float = cos(angle)
    tw_imag: float = sin(angle)
    
    # Complex multiplication: x2 * W
    temp_real: float = x2.x * tw_real - x2.y * tw_imag
    temp_imag: float = x2.x * tw_imag + x2.y * tw_real
    
    # Butterfly operation
    # y1 = x1 + x2*W
    # y2 = x1 - x2*W
    data_out[idx1] = vec2(x1.x + temp_real, x1.y + temp_imag)
    data_out[idx2] = vec2(x1.x - temp_real, x1.y - temp_imag)



"""
Generated GLSL code:
#version 430
layout (local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

uniform uint log2n;
uniform uint nItems;
layout(std430, binding = 0) buffer D0 { vec2 data[]; };

void main() {

        if((gl_GlobalInvocationID.x >= nItems)) {
                return;
        }
        int gid_x_int = int(gl_GlobalInvocationID.x);
        uint rev_idx = uint(0);
        uint temp_gid = uint(gid_x_int);
        for(int i = 0; i < 16; i += 1) {
                uint _t7 = uint(1);
                uint bit = (int(temp_gid) & int(_t7));
                uint _t10 = (int(rev_idx) << 1);
                rev_idx = (int(_t10) | int(bit));
                temp_gid = (int(temp_gid) >> 1);
        }
        int _t15 = (16 - int(log2n));
        rev_idx = (int(rev_idx) >> _t15);
        int _t18 = int(rev_idx);
        bool _t19 = (gid_x_int < _t18);
        if(_t19) {
                vec2 temp_val = data[int(gid_x_int)];
                vec2 _t22 = data[int(rev_idx)];
                data[gid_x_int] = _t22;
                data[rev_idx] = temp_val;
        }
}
"""
@gpu_kernel({
    "data": (Vec_GLTypes.vec2, IOTypes.array),
    "nItems": (NP_GLTypes.uint, IOTypes.uniform),
    "log2n": (NP_GLTypes.uint, IOTypes.uniform),
}, vectorized=False)
def bit_reverse_permute(
    data: IOTypes.array,
    nItems: IOTypes.uniform,
    log2n: IOTypes.uniform
) -> GLTypes.void:
    if gl_GlobalInvocationID.x >= nItems:
        return
    
    gid_x_int: int = int(gl_GlobalInvocationID.x)
    
    # Bit reverse using fixed unrolled approach
    rev_idx: uint = uint(0)
    temp_gid: uint = uint(gid_x_int)
    
    # Reverse 16 bits (up to 2^16 = 65536)
    for i in range(16):
        bit: uint = temp_gid & uint(1)
        rev_idx = (rev_idx << 1) | bit
        temp_gid = temp_gid >> 1
    
    # Shift result right to match bit width
    rev_idx = rev_idx >> (16 - log2n)
    
    if gid_x_int < int(rev_idx):
        temp_val: vec2 = data[gid_x_int]
        data[gid_x_int] = data[rev_idx]
        data[rev_idx] = temp_val


class Radix2FFT:
    def __init__(self, math_engine: GPUMath):
        self.engine = math_engine
        self.butterfly_function = self.engine.compile_fused(radix2_butterfly, debug=True)
        self.bit_reverse_function = self.engine.compile_fused(bit_reverse_permute, debug=True)
        
    def fft(self, data, inverse=False):
        if isinstance(data, CastBuffer):
            buf = data
        else:
            buf = self.engine.push(data)
        
        n = len(buf)
        
        # Check if n is a power of 2
        if n & (n - 1) != 0:
            raise ValueError(f"FFT size must be power of 2, got {n}")
        
        # Calculate number of stages
        num_stages = int(math.log2(n))
        
        # Create ping-pong buffers
        buf_a = buf
        buf_b = self.engine.push(np.zeros_like(self.engine.fetch(buf)))
        
        # Perform bit-reversal permutation (in-place on buf_a)
        self.bit_reverse_function(
            buf_a,
            int(n),
            int(num_stages),
            n_items=n
        )
        self.engine.fetch(buf_a)
        
        current_in = buf_a
        current_out = buf_b
        
        inv_flag = 1 if inverse else 0
        
        for stage in range(num_stages):
            num_butterflies = n // 2
            
            self.butterfly_function(
                current_in,
                current_out,
                int(n),
                int(stage),
                inv_flag,
                n_items=num_butterflies
            )
            
            self.engine.fetch(current_out)
            
            current_in, current_out = current_out, current_in
        
        result = current_in
        
        if inverse:
            # If inverse FFT, scale by 1/n
            result = self.engine.div(result, n)
            result = self.engine.fetch(result)
            result = self.engine.push(result)

        return result


class Radix2Demo(ShowBase):
    """Demo application for Radix-2 FFT"""
    
    def __init__(self, headless=True):
        ShowBase.__init__(self)
        self.engine = GPUMath(self, headless=headless)
        self.fft_class = Radix2FFT(self.engine)
    
    def run_test(self, N=1024, output=True, test_inverse=True):
        def out(*args, **kw):
            if output:
                print(*args, **kw)
        
        out(f"Testing Radix-2 {N}-point FFT...")
        
        # Generate test signal: sum of sinusoids with noise
        t = np.linspace(0, 1, N, endpoint=False)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        x = x + 0.2 * np.random.randn(N)
        x = x.astype(np.complex64)
        
        # GPU FFT
        t0 = time.perf_counter()
        g_res_handle = self.fft_class.fft(x)
        final_gpu = self.engine.fetch(g_res_handle)
        t_gpu = time.perf_counter() - t0
        
        # CPU (reference)
        t1 = time.perf_counter()
        final_cpu = np.fft.fft(x)
        t_cpu = time.perf_counter() - t1
        
        out("\n" + "="*90)
        out(f"{'Index':<8} | {'GPU Value':<30} | {'CPU Value':<30} | {'Abs Diff':<12}")
        out("-" * 90)
        for i in range(min(20, N)):
            g = final_gpu[i]
            c = final_cpu[i]
            diff = abs(g - c)
            out(f"{i:<8} | {str(g):<30} | {str(c):<30} | {diff:.3e}")
        out("="*90 + "\n")
        
        max_diff = np.max(np.abs(final_gpu - final_cpu))
        mean_diff = np.mean(np.abs(final_gpu - final_cpu))
        relative_error = max_diff / (np.max(np.abs(final_cpu)) + 1e-10)
        
        out(f"GPU Time:        {t_gpu:.5f}s")
        out(f"CPU Time:        {t_cpu:.5f}s")
        out(f"Speedup:         {t_cpu/t_gpu:.2f}x" if t_gpu > 0 else "Speedup:         N/A")
        out(f"Max Diff:        {max_diff:.3e}")
        out(f"Mean Diff:       {mean_diff:.3e}")
        out(f"Relative Error:  {relative_error:.3e}")
        out(f"Valid:           {max_diff < 1e-3}")
        
        # Test inverse FFT
        if test_inverse:
            out("\nTesting Inverse FFT...")
            t2 = time.perf_counter()
            final_inv_handle = self.fft_class.fft(g_res_handle, inverse=True)
            final_inv = self.engine.fetch(final_inv_handle)
            t_inv = time.perf_counter() - t2
            
            # Compare with original input (allowing for small numerical error)
            inv_max_diff = np.max(np.abs(final_inv - x))
            inv_mean_diff = np.mean(np.abs(final_inv - x))
            inv_relative_error = inv_max_diff / (np.max(np.abs(x)) + 1e-10)
            
            out(f"IFFT Time:       {t_inv:.5f}s")
            out(f"Max Diff:        {inv_max_diff:.3e}")
            out(f"Mean Diff:       {inv_mean_diff:.3e}")
            out(f"Relative Error:  {inv_relative_error:.3e}")
            out(f"IFFT Valid:      {inv_max_diff < 1e-3}")
        
        return t_gpu, t_cpu
    
    def benchmark(self, sizes=None, num_runs=10):
        if sizes is None:
            sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        
        print("\n" + "="*90)
        print("FFT BENCHMARK - Radix-2 GPU vs NumPy CPU")
        print("="*90)
        print(f"{'Size':<10} | {'GPU Avg (s)':<12} | {'CPU Avg (s)':<12} | {'Speedup':<10} | {'Error':<10}")
        print("-"*90)
        
        results = []
        
        for size in sizes:
            # Verify it's a power of 2
            if size & (size - 1) != 0:
                print(f"Skipping {size} (not a power of 2)")
                continue
            
            gpu_times = []
            cpu_times = []
            errors = []
            
            for run in range(num_runs):
                t = np.linspace(0, 1, size, endpoint=False)
                x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
                x = x + 0.2 * np.random.randn(size)
                x = x.astype(np.complex64)
                
                # GPU
                t0 = time.perf_counter()
                g_res = self.fft_class.fft(x)
                gpu_result = self.engine.fetch(g_res)
                t_gpu = time.perf_counter() - t0
                
                # CPU
                t1 = time.perf_counter()
                cpu_result = np.fft.fft(x)
                t_cpu = time.perf_counter() - t1
                
                gpu_times.append(t_gpu)
                cpu_times.append(t_cpu)
                
                max_error = np.max(np.abs(gpu_result - cpu_result))
                errors.append(max_error)
                
                self.task_mgr.step()
            
            # Discard first run (warmup)
            gpu_times = gpu_times[1:]
            cpu_times = cpu_times[1:]
            errors = errors[1:]
            
            avg_gpu = np.mean(gpu_times)
            avg_cpu = np.mean(cpu_times)
            avg_error = np.mean(errors)
            speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
            
            print(f"{size:<10} | {avg_gpu:<12.6f} | {avg_cpu:<12.6f} | {speedup:<10.2f}x | {avg_error:<10.3e}")
            
            results.append({
                'size': size,
                'gpu_time': avg_gpu,
                'cpu_time': avg_cpu,
                'speedup': speedup,
                'error': avg_error
            })
        
        print("="*90)
        
        # Summary statistics
        if results:
            avg_speedup = np.mean([r['speedup'] for r in results if r['speedup'] > 0])
            max_speedup = max([r['speedup'] for r in results if r['speedup'] > 0])
            best_size = max(results, key=lambda r: r['speedup'])['size']
            
            print(f"\nSummary:")
            print(f"  Average Speedup: {avg_speedup:.2f}x")
            print(f"  Maximum Speedup: {max_speedup:.2f}x (at size {best_size})")
            print(f"  All errors < 1e-3: {all(r['error'] < 1e-3 for r in results)}")
        
        return results


def benchmark():
    for k, v in {
        "window-type": "none",
        "audio-library-name": "null",
        "sync-video": "#f",
    }.items():
        load_prc_file_data("", f"{k} {v}")
    
    app = Radix2Demo(headless=True)
    
    print("\n" + "="*80)
    print("RADIX-2 FFT DEMONSTRATION")
    print("="*80)
    
    # Test with small size
    print("\nTest 1: Small FFT (256 points)")
    app.run_test(N=256, output=True, test_inverse=True)
    
    # Test with medium size
    print("\n" + "="*80)
    print("\nTest 2: Medium FFT (4096 points)")
    app.run_test(N=4096, output=True, test_inverse=True)
    
    # Test with large size
    print("\n" + "="*80)
    print("\nTest 3: Large FFT (16384 points)")
    app.run_test(N=16384, output=True, test_inverse=False)
    
    app.destroy()


if __name__ == "__main__":
    benchmark()