import numpy as np
import time
import math
import textwrap
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer, 
    GeomEnums, ComputeNode, load_prc_file_data,
    GraphicsPipeSelection, FrameBufferProperties,
    WindowProperties, GraphicsPipe, Vec2
)
from gpu_math import (
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

const vec2 W16[16] = vec2[16](vec2(1.0, 0.0), vec2(0.9238795042037964, -0.3826833963394165), vec2(0.7071068286895752, -0.7071068286895752), vec2(0.3826833963394165, -0.9238795042037964), vec2(0.0, -1.0), vec2(-0.3826833963394165, -0.9238795042037964), vec2(-0.7071068286895752, -0.7071068286895752), vec2(-0.9238795042037964, -0.3826833963394165), vec2(-1.0, 0.0), vec2(-0.9238795042037964, 0.3826833963394165), vec2(-0.7071068286895752, 0.7071068286895752), vec2(-0.3826833963394165, 0.9238795042037964), vec2(0.0, 1.0), vec2(0.3826833963394165, 0.9238795042037964), vec2(0.7071068286895752, 0.7071068286895752), vec2(0.9238795042037964, 0.3826833963394165));

uniform uint L;
uniform int inverse;
uniform uint nItems;
layout(std430, binding = 0) buffer D0 { vec2 data_in[]; };
layout(std430, binding = 1) buffer D1 { vec2 data_out[]; };

void main() {
    int gid = int(gl_GlobalInvocationID.x);
    if(gid >= nItems) return;

    uint gidx = uint(gl_GlobalInvocationID.x);
    uint groups = (int(nItems) >> 4);
    bool _t3 = (gidx >= groups);
    uint j = (int(gidx) % int(L));
    uint k = (int(gidx) / int(L));
    vec2 x[16];
    for(int r = 0; r < 16; r += 1) {
        int _t9 = (r * int(groups));
        int _t10 = (int(j) + _t9);
        uint _t11 = (int(k) * int(L));
        int idx = (_t10 + int(_t11));
        vec2 _t14 = data_in[int(idx)];
        x[r] = _t14;
    }
    float PI = 3.141592653589793;
    for(int r = 0; r < 16; r += 1) {
        float _t16 = float(inverse);
        float _t17 = (-_t16);
        float _t18 = (_t17 * 2.0);
        float _t19 = (_t18 * PI);
        int _t20 = (r * int(j));
        float _t21 = float(_t20);
        float _t22 = (_t19 * _t21);
        int _t23 = (16 * int(L));
        float _t24 = float(_t23);
        float ang = (_t22 / _t24);
        float _t27 = cos(ang);
        float _t28 = sin(ang);
        vec2 tw = vec2(_t27, _t28);
        vec2 _t31 = x[int(r)];
        vec2 _t32 = vec2(_t31.x * tw.x - _t31.y * tw.y, _t31.x * tw.y + _t31.y * tw.x);
        x[r] = _t32;
    }
    vec2 y[16];
    for(int r = 0; r < 16; r += 1) {
        vec2 acc = vec2(0.0, 0.0);
        for(int m = 0; m < 16; m += 1) {
            int _t36 = (r * m);
            uint _t37 = uint(15);
            int w_idx = (_t36 & int(_t37));
            vec2 _t40 = x[int(m)];
            vec2 _t41 = W16[int(w_idx)];
            vec2 _t42 = vec2(_t40.x * _t41.x - _t40.y * _t41.y, _t40.x * _t41.y + _t40.y * _t41.x);
            vec2 _t43 = (acc + _t42);
            acc = _t43;
        }
        y[r] = acc;
    }
    for(int i = 0; i < 16; i += 1) {
        int _t45 = (i * int(L));
        int _t46 = (int(j) + _t45);
        int _t47 = (int(L) * 16);
        int _t48 = (int(k) * _t47);
        int idx_out = (_t46 + _t48);
        vec2 _t51 = y[int(i)];
        data_out[idx_out] = _t51;
    }
}
"""

@gpu_kernel({
    "data_in"   :    (Vec_GLTypes.vec2, IOTypes.array),
    "data_out"  :    (Vec_GLTypes.vec2, IOTypes.array),
    "nItems"    :    (NP_GLTypes.uint,  IOTypes.uniform),
    "L"         :    (NP_GLTypes.uint,  IOTypes.uniform),
    "inverse"   :    (NP_GLTypes.int,   IOTypes.uniform),
}) 
# Radix-16 DFT roots: exp(-i*2*pi*k/16)
@static_constant(
    "W16", "vec2", 16,
    [
        Vec2( 1.0000000,  0.0000000),
        Vec2( 0.9238795, -0.3826834),
        Vec2( 0.7071068, -0.7071068),
        Vec2( 0.3826834, -0.9238795),
        Vec2( 0.0000000, -1.0000000),
        Vec2(-0.3826834, -0.9238795),
        Vec2(-0.7071068, -0.7071068),
        Vec2(-0.9238795, -0.3826834),
        Vec2(-1.0000000,  0.0000000),
        Vec2(-0.9238795,  0.3826834),
        Vec2(-0.7071068,  0.7071068),
        Vec2(-0.3826834,  0.9238795),
        Vec2( 0.0000000,  1.0000000),
        Vec2( 0.3826834,  0.9238795),
        Vec2( 0.7071068,  0.7071068),
        Vec2( 0.9238795,  0.3826834)
    ]
)
def butterfly_shader(
    data_in :   IOTypes.array,
    data_out:   IOTypes.array,
    nItems  :   IOTypes.uniform,
    L       :   IOTypes.uniform,
    inverse :   IOTypes.uniform
) -> GLTypes.void:
    # Each thread handles 16 elements
    gidx : uint = gl_GlobalInvocationID.x
    groups : uint = nItems >> 4  # nItems / 16

    if (gidx >= groups):
        return
    
    j : uint = gidx % L
    k : uint = gidx // L
    
    # Load 16 elements
    x : vec2[16]
    for r in range(16):
        idx : uint = j + r * groups + k * L
        x[r] = data_in[idx]
    
    # Apply twiddle factors
    PI : float = 3.14159265358979323846
    for r in range(16):
        ang : float = -float(inverse) * 2.0 * PI * float(r * j) / float(16 * L)
        tw : vec2 = vec2(cos(ang), sin(ang))
        x[r] = cmul(x[r], tw)
    
    # Radix-16 DFT using precomputed W16
    y : vec2[16]
    for r in range(16):
        acc : vec2 = vec2(0.0, 0.0)
        for m in range(16):
            w_idx : uint = (r * m) & uint(15)
            acc += cmul(x[m], W16[w_idx])
        y[r] = acc
    
    # Store results
    for i in range(16):
        idx_out : uint = j + i * L + k * (L * 16)
        data_out[idx_out] = y[i]

# Complex multiplication helper function
@inline_always
def cmul(a: Vec2, b: Vec2) -> Vec2:
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

class Radix16FFT:
    def __init__(self, math_engine:GPUMath):
        self.engine = math_engine
        self.fft_function = self.engine.compile_fused(butterfly_shader, debug=True)
        
    def fft(self, data, inverse=False):
        if isinstance(data, CastBuffer):
            buf = data
        else:
            buf = self.engine.push(data)
        
        n = len(buf)
        if n & (n - 1) != 0:
            raise ValueError("FFT size must be power of 2")
        
        log16n = int(math.log(n) / math.log(16))
        inv_flag = -1 if inverse else 1
        
        # Create ping-pong buffers
        buf_a = buf
        buf_b = self.engine.push(np.zeros(n, dtype=np.complex64))
        
        current_in = buf_a
        current_out = buf_b
        
        for s in range(log16n):
            L = 16**s
            
            # Each thread processes 16 elements
            num_threads = n // 16
            
            self.fft_function(
                current_in,
                current_out,
                int(n),
                int(L),
                inv_flag,
                n_items=num_threads
            )
            
            current_in, current_out = current_out, current_in
        
        return current_in

class FFT16Demo(ShowBase):
    def __init__(self, headless=True):
        ShowBase.__init__(self)
        self.engine = GPUMath(self, headless=True)
        self.fft_class = Radix16FFT(self.engine)

    def run_test(self, N=65536, output=True, test_inverse=True):
        def out(*args, **kw):
            if output:
                print(*args, **kw)

        out(f"Testing Radix-16 {N}-point FFT...")
        
        # Generate test signal
        t = np.linspace(0, 1, N, endpoint=False)
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        x = x + 0.2 * np.random.randn(N)
        x = x.astype(np.complex64)

        # GPU
        t0 = time.perf_counter()
        g_res_handle = self.fft_class.fft(x)
        final_gpu = self.engine.fetch(g_res_handle)
        t_gpu = time.perf_counter() - t0

        # CPU (reference)
        t1 = time.perf_counter()
        final_cpu = np.fft.fft(x)
        t_cpu = time.perf_counter() - t1

        # Results
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

        out(f"GPU Time:  {t_gpu:.5f}s")
        out(f"CPU Time:  {t_cpu:.5f}s")
        out(f"Speedup:   {t_cpu/t_gpu:.2f}x")
        out(f"Max Diff:  {max_diff:.3e}")
        out(f"Mean Diff: {mean_diff:.3e}")
        out(f"Valid:     {max_diff < 1e-1}")
    
        # Inverse FFT
        if test_inverse:
            out("\nTesting Inverse FFT...")
            t2 = time.perf_counter()
            final_inv_handle = self.fft_class.fft(g_res_handle, inverse=True)
            final_inv = self.engine.fetch(final_inv_handle)
            t_inv = time.perf_counter() - t2
            inv_diff = np.max(np.abs(final_inv - x))
            out(f"IFFT Time:     {t_inv:.5f}s")
            out(f"Roundtrip Diff: {inv_diff:.3e}")
            out(f"IFFT Valid:    {inv_diff < 1e-1}")

        return t_gpu, t_cpu

def benchmark_gpu_avg(count = 20):
    for k, v in {
        "window-type":          "none",
        "audio-library-name":   "null",
        "sync-video":           "#f",
    }.items():
        load_prc_file_data("", f"{k} {v}")

    app = FFT16Demo(headless=True)
    gpu_times = []
    cpu_times = []
    
    # Test with smaller size first
    print("Testing with smaller size first...")
    app.run_test(N=256, output=True, test_inverse=True)
    
    # Then benchmark with larger size
    print("\n" + "="*60)
    print("Starting benchmark...")
    print("="*60)
    
    for i in range(count + 1):
        t_gpu, t_cpu = app.run_test(N=16**5, output=False, test_inverse=False)
        gpu_times.append(t_gpu)
        cpu_times.append(t_cpu)
        print(f"Run {i+1}: GPU={t_gpu:.5f}s, CPU={t_cpu:.5f}s, Speedup={t_cpu/t_gpu:.2f}x")
        app.task_mgr.step()

    # Discard warm-up 
    gpu_times = gpu_times[1:]
    cpu_times = cpu_times[1:]

    app.destroy()
    total_gpu   = sum(gpu_times)
    avg_gpu     = total_gpu / len(gpu_times)
    max_gpu     = max(gpu_times)
    min_gpu     = min(gpu_times)
    
    total_cpu   = sum(cpu_times)
    avg_cpu     = total_cpu / len(cpu_times)
    max_cpu     = max(cpu_times)
    min_cpu     = min(cpu_times)
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS ({len(gpu_times)} runs after warm-up):")
    print(f"{'='*60}")
    print(f"GPU: AVG={avg_gpu:.5f}s | MIN={min_gpu:.5f}s | MAX={max_gpu:.5f}s")
    print(f"CPU: AVG={avg_cpu:.5f}s | MIN={min_cpu:.5f}s | MAX={max_cpu:.5f}s")
    print(f"Speedup: {avg_cpu/avg_gpu:.2f}x")

if __name__ == "__main__":
    benchmark_gpu_avg(count=2)