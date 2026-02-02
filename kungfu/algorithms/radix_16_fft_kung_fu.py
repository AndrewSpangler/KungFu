import numpy as np
import kungfu as kf

@kf.gpu_kernel({
    "data_in"   :    (kf.Vec_GLTypes.vec2, kf.IOTypes.array),
    "data_out"  :    (kf.Vec_GLTypes.vec2, kf.IOTypes.array),
    "nItems"    :    (kf.NP_GLTypes.uint,  kf.IOTypes.uniform),
    "L"         :    (kf.NP_GLTypes.uint,  kf.IOTypes.uniform),
    "inverse"   :    (kf.NP_GLTypes.int,   kf.IOTypes.uniform)
}) 
@kf.static_constant(
    "W16", "vec2", 16,
    [
        ( 1.0000000,  0.0000000),
        ( 0.9238795, -0.3826834),
        ( 0.7071068, -0.7071068),
        ( 0.3826834, -0.9238795),
        ( 0.0000000, -1.0000000),
        (-0.3826834, -0.9238795),
        (-0.7071068, -0.7071068),
        (-0.9238795, -0.3826834),
        (-1.0000000,  0.0000000),
        (-0.9238795,  0.3826834),
        (-0.7071068,  0.7071068),
        (-0.3826834,  0.9238795),
        ( 0.0000000,  1.0000000),
        ( 0.3826834,  0.9238795),
        ( 0.7071068,  0.7071068),
        ( 0.9238795,  0.3826834)
    ]
)
def butterfly_shader(
    data_in :   kf.IOTypes.array,
    data_out:   kf.IOTypes.array,
    nItems  :   kf.IOTypes.uniform,
    L       :   kf.IOTypes.uniform,
    inverse :   kf.IOTypes.uniform
) -> kf.GLTypes.void:
    def cmul(a: vec2, b: vec2) -> vec2:
        return vec2(
            a.x * b.x - a.y * b.y,
            a.x * b.y + a.y * b.x
        )
    
    PI : float = 3.14159265358979323846
    thread_id : uint = gl_GlobalInvocationID.x
    groups : uint = nItems >> uint(4)
    
    if thread_id >= groups:
        return
    
    j : uint = thread_id % L
    k : uint = thread_id / L
    stride : uint = groups
    
    x : vec2[16]
    for r in range(16):
        v : vec2 = data_in[j + r * stride + k * L]
        ang : float = -float(inverse) * 2.0 * PI * float(r * j) / float(L * 16)
        tw : vec2 = vec2(cos(ang), sin(ang))
        x[r] = cmul(v, tw)
    
    y : vec2[16]
    for r in range(16):
        acc : vec2 = vec2(0.0, 0.0)
        for m in range(16):
            w_idx : uint = (r * m) & uint(15)
            acc = acc + cmul(x[m], W16[w_idx])
        y[r] = acc
    
    base : uint = k * (L * 16)
    data_out[j + base] = y[0]
    data_out[j + 1 * L + base] = y[1]
    data_out[j + 2 * L + base] = y[2]
    data_out[j + 3 * L + base] = y[3]
    data_out[j + 4 * L + base] = y[4]
    data_out[j + 5 * L + base] = y[5]
    data_out[j + 6 * L + base] = y[6]
    data_out[j + 7 * L + base] = y[7]
    data_out[j + 8 * L + base] = y[8]
    data_out[j + 9 * L + base] = y[9]
    data_out[j + 10 * L + base] = y[10]
    data_out[j + 11 * L + base] = y[11]
    data_out[j + 12 * L + base] = y[12]
    data_out[j + 13 * L + base] = y[13]
    data_out[j + 14 * L + base] = y[14]
    data_out[j + 15 * L + base] = y[15]

class Radix16FFT:
    def __init__(self, math_engine:kf.GPUMath):
        self.engine = math_engine
        self.fft_function = self.engine.compile_fused(butterfly_shader, debug=True)
        
    def fft(self, data, inverse=False):
        if isinstance(data, kf.CastBuffer):
            buf = data
        else:
            buf = self.engine.push(data)
        
        n = len(buf)
        if n & (n - 1) != 0:
            raise ValueError("FFT size must be power of 2")
        
        log16n = int(math.log(n) / math.log(16))
        inv_flag = -1 if inverse else 1
        
        buf_a = buf
        buf_b = self.engine.push(np.zeros(n, dtype=np.complex64))
        
        current_in = buf_a
        current_out = buf_b
        
        for s in range(log16n):
            L = 16**s
            num_threads = n
            
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