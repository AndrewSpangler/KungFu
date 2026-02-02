import numpy as np
import kungfu as kf
from direct.showbase.ShowBase import ShowBase

base = ShowBase()
engine = kf.GPUMath(base, headless=True)

@kf.gpu_kernel({
    "in_array":  (kf.NP_GLTypes.float, kf.IOTypes.array),
    "res":       (kf.NP_GLTypes.float, kf.IOTypes.buffer),
})
def in_out(in_array):
    gidx : uint = gl_GlobalInvocationID.x
    if gidx >= n_items:
        return

    return in_array[gidx]

in_out_kernel = engine.compile_fused(in_out, debug=True)

thread_count = 10000

# Create numpy data
buff = np.random.rand(thread_count).astype(np.float32)
handle = in_out_kernel(buff, nItems=thread_count)
result = engine.fetch(handle)

print(buff[:5], result[:5])