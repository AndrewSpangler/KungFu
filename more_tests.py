# [file name]: example_kernels.py
import numpy as np
from gpu_math import GPUMath, gpu_kernel, inline_always, NP_GLTypes, IOTypes, GLTypes

# Example 1: Vectorized kernel (uses IOTypes.buffer)
@gpu_kernel({
    "data_in": (NP_GLTypes.float, IOTypes.buffer),
    "data_out": (NP_GLTypes.float, IOTypes.buffer),
})
def vectorized_square(data_in, data_out):
    gid : uint = gl_GlobalInvocationID.x
    if gid >= n_items:
        return
    data_out[gid] = data_in[gid] * data_in[gid]

# Example 2: Array-style kernel (uses IOTypes.array)
@gpu_kernel({
    "data": (NP_GLTypes.float, IOTypes.array),
    "result": (NP_GLTypes.float, IOTypes.array),
    "size": (NP_GLTypes.uint, IOTypes.uniform),
})
def array_sum(data, result, size):
    # Manual indexing - no automatic gid
    acc : float = 0.0
    for i in range(size):
        acc += data[i]
    result[0] = acc

# Example 3: Mixed kernel (both buffer and array)
@gpu_kernel({
    "vector_data": (NP_GLTypes.float, IOTypes.buffer),
    "lookup_table": (NP_GLTypes.float, IOTypes.array),
    "output": (NP_GLTypes.float, IOTypes.buffer),
    "table_size": (NP_GLTypes.uint, IOTypes.uniform),
})
def table_lookup(vector_data, lookup_table, output, table_size):
    gid : uint = gl_GlobalInvocationID.x
    if gid >= n_items:
        return
    
    # Use gid for vectorized access
    idx : uint = uint(vector_data[gid] * float(table_size - 1))
    if idx < table_size:
        output[gid] = lookup_table[idx]
    else:
        output[gid] = 0.0

# Example 4: Using KungFu built-ins
@gpu_kernel({
    "data": (NP_GLTypes.float, IOTypes.buffer),
    "output": (NP_GLTypes.float, IOTypes.buffer),
})
def builtin_demo(data, output):
    # Using KungFu built-in aliases
    idx : uint = gid  # Same as gl_GlobalInvocationID.x
    wg_x : uint = wgid_x  # Same as gl_WorkGroupID.x
    lid_x : uint = lid_x  # Same as gl_LocalInvocationID.x
    
    if idx < n_items:
        # Mix of built-ins
        output[idx] = data[idx] * float(wg_x + lid_x)

# Example 5: Vector component access
@gpu_kernel({
    "points": (NP_GLTypes.vec3, IOTypes.buffer),
    "distances": (NP_GLTypes.float, IOTypes.buffer),
})
def vector_demo(points, distances):
    gid : uint = gl_GlobalInvocationID.x
    if gid >= n_items:
        return
    
    # Access vector components
    point : vec3 = points[gid]
    x : float = point.x
    y : float = point.y
    z : float = point.z
    
    # Or use swizzling
    xy : vec2 = point.xy
    
    distances[gid] = sqrt(x*x + y*y + z*z)

# Example that should raise an error (nItems in vectorized kernel)
@gpu_kernel({
    "data": (NP_GLTypes.float, IOTypes.buffer),
    "output": (NP_GLTypes.float, IOTypes.buffer),
    "nItems": (NP_GLTypes.uint, IOTypes.uniform),  # This should raise an error!
})
def invalid_kernel(data, output, nItems):
    gid : uint = gl_GlobalInvocationID.x
    if gid >= nItems:  # Error: nItems should not be a parameter
        return
    output[gid] = data[gid] * 2.0