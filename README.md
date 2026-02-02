# KungFu

> An engine for writing Python-styled code to generate graphics shaders and using the GPU for general compute. Directly integrated with Panda3D for visualization. Transpiles from specially annotated Python to GLSL. 

## Table of Contents
- [Usage](#usage)
    - [Kernels](#kernels)
    - [Shaders](#shaders)
    - [Engine Functions](#engine-functions)
- [Syntax And Typing](#syntax-and-typing)
    - [Common Syntax](#common-sytax)
        - [Annotations](#annotations)
        - [Array Creation](#array-creation)
    - [Builtins](#builtins)
        - [Panda3D Builtins](#panda3d-builtins)
        - [OpenGL Builtins](#opengl-builtins)
        - [KungFu Builtins](#kungfu-builtins)
- [Libraries](#libraries)
  - [Strings.py Library](#stringspy-library)
    - [Char Handling](#char-handling)
    - [String Handling](#string-handling)
    - [Conversion](#conversion)
    - [Fragment Shader Helper](#fragment-shader-helper)
    - [Engine Helpers](#engine-helpers)
  - [Math.py Library](#mathpy-library)
    - [Distance and Geometry](#distance-and-geometry)
    - [Bounds Checking](#bounds-checking)
    - [Clamping and Mapping](#clamping-and-mapping)
    - [Interpolation](#interpolation)
    - [Vector Operations](#vector-operations)
    - [Easing Functions](#easing-functions)
    - [Noise and RNG](#noise-and-rng)
    - [SDF Shapes](#sdf-shapes)
    - [Utility Functions](#utility-functions)
  - [Colors.py Library](#colorspy-library)
    - [Color Conversions](#color-conversions)
    - [Color Adjustments](#color-adjustments)
    - [Blending](#blending)
    - [Effects](#effects)
    - [Color Utils](#color-utils)
    - [Palettes](#palettes)

## Usage

KungFU has two usage modes:

- Kernels
    - Kernels are compute shaders used for general computations.
    - Kernels produce shader buffers that can be passed to other kernels to chain computations without reading back to the CPU. 
    - Kernels perform vectorized operations on shader buffers, and are best used for parallelized operations.
- Shaders
    - Shaders are primarily used for graphics
    - Vertex, fragment, geometry, and compute shaders are supported.
    - Compute shaders can be used to make highly customized kernels, as well as be integrated with graphics chains.

### Kernels

> Kernels were designed to handle automatically mapping input buffers by thread index. This allows Shader Buffers to be used very similarly to numpy arrays.

Example of a very basic kernel:
```py
import numpy as np
import kungfu as kf
from direct.showbase.ShowBase import ShowBase

base = ShowBase()
engine = kf.GPUMath(base)

@kf.gpu_kernel({
    #"NAME":((PY_TYPE, GLSL_TYPE),IO_TYPE))
    # Or
    #"NAME":(NP_GLTYPE,           IO_TYPE)
    # All IOTypes.buffers must be the same length for a given kernel
    "a":    (kf.NP_GLTypes.float, kf.IOTypes.buffer),
    "b":    (kf.NP_GLTypes.float, kf.IOTypes.buffer),
    # Res is the return value, automatically created on return statement in kernels 
    "res":  (kf.NP_GLTypes.float, kf.IOTypes.buffer)
}, vectorized=True)
def squared_sum(a, b): # You can use type-hints here, however they have no effect on the transpiler.
    return a*a + b*b

# Transpile to GLSL and wrap the generated shader to it can be called like a function 
squared_sum_kernel = engine.compile_fused(squared_sum, debug=True)

# Generate some test data
x = np.random.rand(10000).astype(np.float32)
y = np.random.rand(10000).astype(np.float32)

# Run kernel, get shader buffer object back
handle = squared_sum_kernel(x, y)
# Fetch result from GPU and convert to numpy array type specified in res (kf.NP_GLTypes.float)
gpu_output = engine.fetch(handle)
cpu_output = x*x + y*y # Numpy equivalent
# Set accuracy (GPU is 32 bit)
cpu_output = cpu_output.astype(np.float32)

# Calulate differences
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
```

IOTypes.buffer and IOTypes.array are both shader buffers, however sometimes you want to be able to access the whole buffer rather than one value per thread. Buffers are 1:1 per thread, whereas arrays can be accessed in full. Both can be used in the same kernel:

```py
@kf.gpu_kernel({
    "in_buffer": (kf.NP_GLTypes.float, kf.IOTypes.buffer),
    "in_array":  (kf.NP_GLTypes.float, kf.IOTypes.array),
    "res":       (kf.NP_GLTypes.float, kf.IOTypes.buffer)
}, vectorized=True)
def mixed_type(in_buffer, in_array):
    # in_buffer is accessed automatically by thread (gidx)
    # use half of that to sample in_array
    gidx : uint = gl_GlobalInvocationID.x
    return in_buffer + in_array[gidx // 2]

# in_array is half the length of in_buffer
buff = np.random.rand(10000).astype(np.float32)
arr  = np.random.rand(5000).astype(np.float32)

mixed_type_kernel = engine.compile_fused(mixed_type, debug=True)

handle = mixed_type_kernel(buff, arr)
result = engine.fetch(handle)

...
```

### Shaders
Shaders have their respective GL and Panda3D builtins provided automatically. All types of shaders are built via the same process, they are decorated with `@engine.shader("shader_type")` and then compiled with `engine.compile_shader(decorated_function)`. The compile_shader call returns the shader as a string compatible with Panda3D, and a dictionary of shader metadata. 
```py
@engine.shader('vertex')
def vertex_shader():
    position: vec4 = p3d_Vertex
    gl_Position : vec4 = p3d_ModelViewProjectionMatrix * position

@engine.shader('fragment')
def fragment_shader():
    p3d_FragColor = vec4(1, 0, 0, 1)

vertex, vertex_info = engine.compile_shader(vertex_shader, debug=True)
fragment, fragment_info = engine.compile_shader(fragment_shader, debug=True)
shader = Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)

node.setShader(shader)
node.setTransparency(TransparencyAttrib.MAlpha)
```

There are 4 supported shader types:
- fragment
- vertex
- geometry
- compute

### Engine Functions
Engine functions are shared, reusable calls that will automatically be added to the shader if needed.

Once registed with @engine.function() they become available in any @engine.shader().

Support in kernels is partial. There are issues with early return statements in kernels that may not be properly handled yet.

Engine functions are defined very similarly to shaders and kernels.

```py
@engine.function({
    'matrix': 'mat4',
    'position': 'vec4'
}, return_type='vec4')
def custom_position(matrix, position) -> Vec4:
    # Double position
    return vec4(matrix * (2.0 * position)) 
```

## Syntax And Typing

### Common Syntax
#### Annotations

GLSL is strongly typed and dislikes inferred type casts

Type annotation is used to provide the tranpiler this information.

Variables defaut to float if not declared explicitly - this may change in the future as the transpiler's type inferment improves.

The following types are available, and may be used for both setting variable types and casting. 

```py
class GLTypes:
    # Basic Types
    float           = "float"
    double          = "double"
    int             = "int"
    uint            = "uint"
    bool            = "bool"
    void            = "void"
    vec2            = "vec2"
    vec3            = "vec3"
    vec4            = "vec4"
    uvec2           = "uvec2"
    uvec3           = "uvec3"
    uvec4           = "uvec4"
    ivec2           = "ivec2"
    ivec3           = "ivec3"
    ivec4           = "ivec4"
    
    # Texture / mats / samplers
    mat             = "mat"
    mat2            = "mat2"
    mat3            = "mat3"
    sampler2D       = "sampler2D"
    sampler2DArray  = "sampler2DArray"
    sampler3D       = "sampler3D "
    samplerCube     = "samplerCube"
    sampler2DShadow = "sampler2DShadow"

    # Array Types
    float_array         = "float[]"
    double_array        = "double[]"
    int_array           = "int[]"
    uint_array          = "uint[]"
    bool_array          = "bool[]"
    void_array          = "void[]"
    vec2_array          = "vec2[]"
    vec3_array          = "vec3[]"
    vec4_array          = "vec4[]"
    uvec2_array         = "uvec2[]"
    uvec3_array         = "uvec3[]"
    uvec4_array         = "uvec4[]"
    ivec2_array         = "ivec2[]"
    ivec3_array         = "ivec3[]"
    ivec4_array         = "ivec4[]"
    mat_array           = "mat[]"
    mat2_array          = "mat2[]"
    mat3_array          = "mat3[]"
    sampler2D_array       = "sampler2D[]"
    sampler2DArray_array  = "sampler2DArray[]"
    sampler3D_array       = "sampler3D[]"
    samplerCube_array     = "samplerCube[]"
    sampler2DShadow_array = "sampler2DShadow[]"
```

```py
@engine.function({...})
def func(...):
    # Auto-casts to float
    val_float_auto = 1.2
    # Also auto-casts to float
    val_float_auto_2 = 2 

    # Will generate GLSL: uint(2.2) and end up as 2u in value
    val_uint : uint = uint(2.2)

    # Type will not be inferred by the transpiler (yet) and try to assign to float
    val_uint_wrong = uint(2.2) # will error in Panda3D compiler

    # This is not supported by python syntax.
    val_uint_wrong_2 = 2u # Results in python exception: "invalid decimal litteral"

    # Correct
    val_vec3 : vec3 = vec3(1,2,3)

    # Wrong (for now, eventually 2-4 dimensional tuples will auto-cast to their vec equivalent)
    val_vec3_wrong : vec3 = (1,2,3)

    # values can be cast in-line
    func_output : float = float(call_that_returns_int(float(abc), int(ghi)))
```

#### Array Creation

KungFu handles array declaration using a special annotation `name : type[count]`.

```py
@engine.function({...})
def func(...):
    array : float[8]
    array2d : float[8, 8]
    vec2_array : vec2[8]
    vec4_array2d : vec4[8, 8]

    # Array assignment is supported
    for i in range(3):
        for j in range(4):
            value : vec4 = vec4(i * j, i, j, 1)
            vec4_array2d[i][j] = value
```

### Builtins

KungFu supports direct usage of most GL and Panda3D builtins, and defines some aliases for ease of use.

#### Panda3D Builtins

| Built-in Variable | Type | Category |
|-------------------|------|----------|
| **Vertex Attributes** |
| p3d_Vertex | vec4 | Vertex position |
| p3d_Normal | vec3 | Vertex normal |
| p3d_Color | vec4 | Vertex color |
| **Texture Coordinates** |
| p3d_TexCoord | vec2 | Primary texture coordinate |
| p3d_TexCoord0 | vec2 | Texture coordinate set 0 |
| p3d_TexCoord1 | vec2 | Texture coordinate set 1 |
| p3d_TexCoord2 | vec2 | Texture coordinate set 2 |
| p3d_TexCoord3 | vec2 | Texture coordinate set 3 |
| p3d_TexCoord4 | vec2 | Texture coordinate set 4 |
| p3d_TexCoord5 | vec2 | Texture coordinate set 5 |
| p3d_TexCoord6 | vec2 | Texture coordinate set 6 |
| p3d_TexCoord7 | vec2 | Texture coordinate set 7 |
| **Multi-Texture Coordinates** |
| p3d_MultiTexCoord0 | vec2 | Multi-texture coordinate 0 |
| p3d_MultiTexCoord1 | vec2 | Multi-texture coordinate 1 |
| p3d_MultiTexCoord2 | vec2 | Multi-texture coordinate 2 |
| p3d_MultiTexCoord3 | vec2 | Multi-texture coordinate 3 |
| p3d_MultiTexCoord4 | vec2 | Multi-texture coordinate 4 |
| p3d_MultiTexCoord5 | vec2 | Multi-texture coordinate 5 |
| p3d_MultiTexCoord6 | vec2 | Multi-texture coordinate 6 |
| p3d_MultiTexCoord7 | vec2 | Multi-texture coordinate 7 |
| **Transformation Matrices** |
| p3d_ModelMatrix | mat4 | Model matrix |
| p3d_ViewMatrix | mat4 | View matrix |
| p3d_ProjectionMatrix | mat4 | Projection matrix |
| p3d_ModelViewMatrix | mat4 | Model-view matrix |
| p3d_ModelViewProjectionMatrix | mat4 | Model-view-projection matrix |
| p3d_ModelViewMatrixInverse | mat4 | Inverse model-view matrix |
| p3d_ModelViewProjectionMatrixInverse | mat4 | Inverse model-view-projection matrix |
| p3d_NormalMatrix | mat3 | Normal transformation matrix |
| **Textures** |
| p3d_Texture0 | sampler2D | Texture sampler 0 |
| p3d_Texture1 | sampler2D | Texture sampler 1 |
| p3d_Texture2 | sampler2D | Texture sampler 2 |
| p3d_Texture3 | sampler2D | Texture sampler 3 |
| **Fragment Outputs** |
| p3d_FragColor | vec4 | Fragment color output |
| p3d_FragData | vec4[] | Fragment data outputs |

#### OpenGL Builtins

These variabels can be used directly in their respective shaders

| Shader Type | Built-in Variable | Type | I/O |
|-------------|-------------------|------|-----|
| **Fragment** | gl_FragCoord | vec4 | Input |
| | gl_FrontFacing | bool | Input |
| | gl_PointCoord | vec2 | Input |
| | gl_ClipDistance | float[] | Input |
| | gl_PrimitiveID | int | Input |
| | gl_FragColor | vec4 | Output (GLSL < 130) |
| | gl_FragData | vec4[] | Output (GLSL < 130) |
| **Vertex** | gl_VertexID | int | Input |
| | gl_InstanceID | int | Input |
| | gl_Position | vec4 | Output |
| | gl_PointSize | float | Output |
| | gl_ClipDistance | float[] | Output |
| **Geometry** | gl_PrimitiveIDIn | int | Input |
| | gl_InvocationID | int | Input |
| | gl_Position | vec4 | Output |
| | gl_PointSize | float | Output |
| | gl_ClipDistance | float[] | Output |
| | gl_PrimitiveID | int | Output |
| | gl_Layer | int | Output |
| | gl_ViewportIndex | int | Output |
| **Compute** | gl_NumWorkGroups | uvec3 | Input |
| | gl_WorkGroupID | uvec3 | Input |
| | gl_LocalInvocationID | uvec3 | Input |
| | gl_GlobalInvocationID | uvec3 | Input |
| | gl_LocalInvocationIndex | uint | Input |
| | gl_WorkGroupSize | uvec3 | Input |
| | gl_MaxVertexAttribs | int | Constant (3.3) |
| | gl_MaxVertexOutputComponents | int | Constant (3.3) |
| | gl_MaxVertexUniformComponents | int | Constant (3.3) |
| | gl_MaxVertexTextureImageUnits | int | Constant (3.3) |
| | gl_MaxGeometryInputComponents | int | Constant (3.3) |
| | gl_MaxGeometryOutputComponents | int | Constant (3.3) |
| | gl_MaxGeometryUniformComponents | int | Constant (3.3) |
| | gl_MaxGeometryTextureImageUnits | int | Constant (3.3) |
| | gl_MaxGeometryOutputVertices | int | Constant (3.3) |
| | gl_MaxGeometryTotalOutputComponents | int | Constant (3.3) |
| | gl_MaxGeometryVaryingComponents | int | Constant (3.3) |
| | gl_MaxFragmentInputComponents | int | Constant (3.3) |
| | gl_MaxDrawBuffers | int | Constant (3.3) |
| | gl_MaxFragmentUniformComponents | int | Constant (3.3) |
| | gl_MaxTextureImageUnits1 | int | Constant (3.3) |
| | gl_MaxClipDistances | int | Constant (3.3) |
| | gl_MaxCombinedTextureImageUnits | int | Constant (3.3) |
| | gl_MaxTessControlInputComponents | int | Constant (4.0) |
| | gl_MaxTessControlOutputComponents | int | Constant (4.0) |
| | gl_MaxTessControlUniformComponents | int | Constant (4.0) |
| | gl_MaxTessControlTextureImageUnits | int | Constant (4.0) |
| | gl_MaxTessControlTotalOutputComponents | int | Constant (4.0) |
| | gl_MaxTessEvaluationInputComponents | int | Constant (4.0) |
| | gl_MaxTessEvaluationOutputComponents | int | Constant (4.0) |
| | gl_MaxTessEvaluationUniformComponents | int | Constant (4.0) |
| | gl_MaxTessEvaluationTextureImageUnits | int | Constant (4.0) |
| | gl_MaxTessPatchComponents | int | Constant (4.0) |
| | gl_MaxPatchVertices | int | Constant (4.0) |
| | gl_MaxTessGenLevel | int | Constant (4.0) |
| | gl_MaxViewports | int | Constant (4.1) |
| | gl_MaxVertexUniformVectors | int | Constant (4.1) |
| | gl_MaxFragmentUniformVectors | int | Constant (4.1) |
| | gl_MaxVaryingVectors | int | Constant (4.1) |
| | gl_MaxVertexImageUniform | int | Constant (4.2) |
| | gl_MaxVertexAtomicCounter | int | Constant (4.2) |
| | gl_MaxVertexAtomicCounterBuffer | int | Constant (4.2) |
| | gl_MaxTessControlImageUniform | int | Constant (4.2) |
| | gl_MaxTessControlAtomicCounter | int | Constant (4.2) |
| | gl_MaxTessControlAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxTessEvaluationImageUniforms | int | Constant (4.2) |
| | gl_MaxTessEvaluationAtomicCounters | int | Constant (4.2) |
| | gl_MaxTessEvaluationAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxGeometryImageUniforms | int | Constant (4.2) |
| | gl_MaxGeometryAtomicCounters | int | Constant (4.2) |
| | gl_MaxGeometryAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxFragmentImageUniforms | int | Constant (4.2) |
| | gl_MaxFragmentAtomicCounters | int | Constant (4.2) |
| | gl_MaxFragmentAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxCombinedImageUniforms | int | Constant (4.2) |
| | gl_MaxCombinedAtomicCounters | int | Constant (4.2) |
| | gl_MaxCombinedAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxImageUnits | int | Constant (4.2) |
| | gl_MaxCombinedImageUnitsAndFragmentOutputs | int | Constant (4.2) |
| | gl_MaxImageSamples | int | Constant (4.2) |
| | gl_MaxAtomicCounterBindings | int | Constant (4.2) |
| | gl_MaxAtomicCounterBufferSize | int | Constant (4.2) |
| | gl_MinProgramTexelOffset | int | Constant (4.2) |
| | gl_MaxProgramTexelOffset | int | Constant (4.2) |
| | gl_MaxVertexAtomicCounters | int | Constant (4.2) |
| | gl_MaxVertexAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxTessControlImageUniforms | int | Constant (4.2) |
| | gl_MaxTessControlAtomicCounters | int | Constant (4.2) |
| | gl_MaxTessControlAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxTessEvaluationImageUniforms | int | Constant (4.2) |
| | gl_MaxTessEvaluationAtomicCounters | int | Constant (4.2) |
| | gl_MaxTessEvaluationAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxGeometryImageUniforms | int | Constant (4.2) |
| | gl_MaxGeometryAtomicCounters | int | Constant (4.2) |
| | gl_MaxGeometryAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxFragmentImageUniforms | int | Constant (4.2) |
| | gl_MaxFragmentAtomicCounters | int | Constant (4.2) |
| | gl_MaxFragmentAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxCombinedImageUniforms | int | Constant (4.2) |
| | gl_MaxCombinedAtomicCounters | int | Constant (4.2) |
| | gl_MaxCombinedAtomicCounterBuffers | int | Constant (4.2) |
| | gl_MaxImageUnits | int | Constant (4.2) |
| | gl_MaxCombinedImageUnitsAndFragmentOutputs | int | Constant (4.2) |
| | gl_MaxImageSamples | int | Constant (4.2) |
| | gl_MaxAtomicCounterBindings | int | Constant (4.2) |
| | gl_MaxAtomicCounterBufferSize | int | Constant (4.2) |
| | gl_MinProgramTexelOffset | int | Constant (4.2) |
| | gl_MaxProgramTexelOffset | int | Constant (4.2) |
| | gl_MaxComputeWorkGroupCount | ivec3 | Constant (4.3) |
| | gl_MaxComputeWorkGroupSize | ivec3 | Constant (4.3) |
| | gl_MaxComputeUniformComponents | int | Constant (4.3) |
| | gl_MaxComputeTextureImageUnits | int | Constant (4.3) |
| | gl_MaxComputeImageUniforms | int | Constant (4.3) |
| | gl_MaxComputeAtomicCounters | int | Constant (4.3) |
| | gl_MaxComputeAtomicCounterBuffers | int | Constant (4.3) |
| | gl_MaxTransformFeedbackBuffers | int | Constant (4.4) |
| | gl_MaxTransformFeedbackInterleavedComponents | int | Constant (4.4) |

#### KungFu Builtins

These variables can be used in kernels:

| Built-in Variable | Type | Description |
|-------------------|------|-------------|
| **Global Invocation ID** |
| gid | uint | Alias for gl_GlobalInvocationID.x |
| gid_x | uint | Explicit x component access |
| gid_y | uint | Explicit y component access |
| gid_z | uint | Explicit z component access |
| gid_xyz | uvec3 | Full vector |
| **Work Group ID** |
| wgid | uvec3 | Alias for gl_WorkGroupID |
| wgid_x | uint | Work group ID x component |
| wgid_y | uint | Work group ID y component |
| wgid_z | uint | Work group ID z component |
| **Local Invocation ID** |
| lid | uvec3 | Alias for gl_LocalInvocationID |
| lid_x | uint | Local invocation ID x component |
| lid_y | uint | Local invocation ID y component |
| lid_z | uint | Local invocation ID z component |
| **Local Invocation Index** |
| lid_idx | uint | Alias for gl_LocalInvocationIndex |
| **Work Group Size** |
| wg_size | uvec3 | Alias for gl_WorkGroupSize |
| wg_size_x | uint | Work group size x component |
| wg_size_y | uint | Work group size y component |
| wg_size_z | uint | Work group size z component |
| **Number of Work Groups** |
| num_wg | uvec3 | Alias for gl_NumWorkGroups |
| num_wg_x | uint | Number of work groups x component |
| num_wg_y | uint | Number of work groups y component |
| num_wg_z | uint | Number of work groups z component |
| **Special KungFu Variables** |
| n_items | uint | Number of items to process (from uniform) |
| global_idx | uint | Calculated global index for 1D kernels |


### String Lib Implementation
    TODO

## Libraries
Below are the current libraries, and their signatures.

These signatures are in a pseudo-code format for easy reference.

See the library files for full decorators / typehinting. 

### Strings.py Library

```py
engine.import_file("./shader_libraries/strings.py")
```

#### Char handling
```py
def is_whitespace(char_code: uint) -> bool:
    """Check if character is whitespace (space, tab, newline, etc.)"""

def is_digit(char_code: uint) -> bool:
    """Check if character is a digit (0-9)"""

def is_alpha(char_code: uint) -> bool:
    """Check if character is alphabetic (A-Z or a-z)"""

def is_alnum(char_code: uint) -> bool:
    """Check if character is alphanumeric (A-Z, a-z, or 0-9)"""

def is_upper(char_code: uint) -> bool:
    """Check if character is uppercase (A-Z)"""

def is_lower(char_code: uint) -> bool:
    """Check if character is lowercase (a-z)"""

def to_upper(char_code: uint) -> uint:
    """Convert character to uppercase"""

def to_lower(char_code: uint) -> uint:
    """Convert character to lowercase"""
```

#### String Handling
```py
def string_length(str_array: uint[255]) -> uint:
    """Get the length of a string (up to first terminator or array length)"""

def is_empty_string(str_array: uint[255]) -> bool:
    """Check if string is empty (first character is terminator)"""

def string_equal(str1: uint[255], str2: uint[255]) -> bool:
    """Compare two strings for equality"""

def string_compare(str1: uint[255], str2: uint[255]) -> int:
    """
    Compare two strings lexicographically.
    Returns:
        0 if equal
        <0 if str1 < str2
        >0 if str1 > str2
    """

def string_to_upper(str_array: uint[255]) -> uint[255]:
    """Convert string to uppercase"""

def string_to_lower(str_array: uint[255]) -> uint[255]:
    """Convert string to lowercase"""

def string_reverse(str_array: uint[255]) -> uint[255]:
    """Reverse a string"""

def string_find(str_array: uint[255], char_code: uint) -> uint:
    """Find first occurrence of character. Returns index or max uint if not found"""

def string_rfind(str_array: uint[255], char_code: uint) -> uint:
    """Find last occurrence of character. Returns index or max uint if not found"""

def string_count(str_array: uint[255], char_code: uint) -> uint:
    """Count occurrences of character"""

def string_replace_char(str_array: uint[255], old_char_code: uint, new_char_code: uint) -> uint[255]:
    """Replace all occurrences of old_char_code with new_char_code"""

def string_concat(str1: uint[255], str2: uint[255]) -> uint[255]:
    """Concatenate two strings"""

def string_substring(str_array: uint[255], start: uint, length: uint) -> uint[255]:
    """Extract substring by index and length"""
```

#### Conversion
```py
def string_to_int(str_array: uint[255]) -> int:
    """Convert string to integer"""

def string_to_float(string_to_float: uint[255]) -> float:
    """Convert string to float"""

def string_is_numeric(str_array: uint[255]) -> bool:
    """Check if string contains only digits and optionally one dot and minus sign"""

def string_hash(str_array: uint[255]) -> uint:
    """Simple string hash function (djb2 algorithm)"""

def int_to_string(value: int) -> uint[255]:
    """Convert integer to string"""

def float_to_string(value: float, precision: uint) -> uint[255]:
    """Convert float to string with given precision"""
```

#### Fragment Shader Helper
```py
def render_text(
    text_array:      uint[{255}],
    color:           vec4,
    cols:            uint,
    rows:            uint,
    x:               float,
    y:               float,
    charmap_texture: sampler2D,
    char_uvs:        vec4[48]
) -> vec4:
    """Render text using character map texture"""
```

#### Engine Helpers    
```py
def encode_string(val: str) -> list:
    """
    Helper function to encode strings to a list of integers for Panda3D
    Pads with -1 (which is 0xFFFFFFFF in two's complement / unsigned interpretation)
    This works with Panda3D's C++ backend which uses signed int conversion
    """

def encode_string_glsl(val: str) -> str:
    """Helper function to encode strings to GLSL format"""


"""Engine helpers available with:
- engine.encode_string
- engine.encode_string_glsl
"""
```

### Math.py Library

```py
engine.import_file("./shader_libraries/math.py")
```

#### Distance and Geometry
```py
def dist(a:float, b:float) -> float:
    """2D Euclidean distance from origin"""

def dist_vec2(p1: vec2, p2: vec2) -> float:
    """Distance between two 2D points"""

def dist_vec3(p1: vec3, p2: vec3) -> float:
    """Distance between two 3D points"""

def manhattan_dist_vec2(p1: vec2, p2: vec2) ->float:
    """Manhattan distance between two 2D points"""

def chebyshev_dist(x: float, y: float) -> float:
    """Chebyshev distance (L-infinity norm)"""
```

#### Bounds Checking
```py
def in_bounds(position:vec2, bound_a:vec2, bound_b:vec2) -> bool:
    """Check if position is within rectangular bounds"""

def in_circle(point: vec2, center: vec2, radius: float) -> bool:
    """Check if point is inside circle"""

def in_sphere( point: vec3, center: vec3, radius: float) -> bool:
    """Check if point is inside sphere"""
```

#### Clamping and Mapping
```py
def clamp_float(value: float, min_val: float, max_val: float) -> float:
    """Clamp floating value between min and max"""

def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map value from one range to another"""

def smooth_step(value: float, edge0: float, edge1: float) -> float:
    """Hermite interpolation between 0 and 1"""

def smoother_step(value: float, edge0: float, edge1: float) -> float:
    """Smoother Hermite interpolation"""
```

#### Interpolation
```py
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation"""

def lerp_vec2(a: vec2, b: vec2, t: float) -> vec2:
    """Linear interpolation with vec2s"""

def lerp_vec3(a: vec3, b: vec3, t: float) -> vec3:
    """Linear interpolation for vec3s"""

def bezier_quadratic(p0: vec2, p1: vec2, p2: vec2, t: float) -> vec2:
    """Quadratic Bezier curve"""

def bezier_cubic(p0: vec2, p1: vec2, p2: vec2, p3: vec2, t: float) -> vec2:
    """Cubic Bezier curve"""
```

#### Vector Operations
```py
def magnitude_vec2(v: vec2) -> float:
    """Get magnitude/length of 2D vector"""

def magnitude_vec3(v: vec3) -> float:
    """Get magnitude/length of 3D vector"""

def normalize_vec2(v: vec2) -> vec2:
    """Normalize 2D vector to unit length"""

def normalize_vec3(v: vec3) -> vec3:
    """Normalize 3D vector to unit length"""

def rotate_vec2(v: vec2, angle: float) -> vec2:
    """Rotate 2D vector by angle (in radians)"""

def angle_between_vec2(v: vec2, v2: vec2) -> float:
    """Get angle between two 2D vectors"""

def perpendicular(v: vec2) -> vec2:
    """Get perpendicular vector (rotated 90 degrees)"""

def reflect_vec2(incident: vec2, normal: vec2) -> vec2:
    """Reflect vector across normal"""

def reflect_vec3(incident: vec3, normal: vec3) -> vec3:
    """Reflect 3D vector across normal"""
```

#### Easing Functions
```py
def ease_in_quad(t: float) -> float:
    """Quadratic ease-in"""

def ease_out_quad(t: float) -> float:
    """Quadratic ease-out"""

def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out"""

def ease_in_cubic(t: float) -> float:
    """Cubic ease-in"""

def ease_out_cubic(t: float) -> float:
    """Cubic ease-out"""

def ease_in_sine(t: float) -> float:
    """Sine ease-in"""

def ease_out_sine(t: float) -> float:
    """Sine ease-out"""

def ease_in_out_sine(t: float) -> float:
    """Sine ease-in-out"""

def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out"""

def ease_in_out_sine(t: float) -> float:
    """Sine ease-in-out"""
```

#### Noise and RNG
```py
def pseudo_random(x: float, y: float) -> float:
    """Pseudo-random hash function for 2D coordinates"""

def pseudo_random_vec2(v: vec2) -> float:
    """Pseudo-random hash function for vec2"""

def pseudo_random_vec3(v: vec3) -> float:
    """Pseudo-random hash function for vec3"""
```

#### SDF Shapes
```py
def sdf_circle(p: vec2, radius: float) -> float:
    """Signed distance to circle"""

def sdf_box(p: vec2, size: vec2) -> float:
    """Signed distance to box"""

def sdf_line(p: vec2, a: vec2, b: vec2, radius: float) -> float:
    """Signed distance to line segment"""
```

#### Utility Functions
```py
def sign_non_zero(value : float) -> float:
    """Sign function that returns 1.0 for 0 instead of 0"""

def snap_to_grid(value: float, step_size: float) -> float:
    """Snap value to nearest grid point"""

def wrap(value: float, period: float) -> float:
    """Wrap value to period [0, period)"""

def step_threshold(value: float, threshold: float) -> float:
    """Step function: 0 if value < threshold, 1 otherwise"""

def approximately_equal(a: float, b: float, tolerance: float) -> bool:
    """Check if two floats are approximately equal"""
```

### Colors.py Library

```py
engine.import_file("./shader_libraries/colors.py")
```

#### Color Conversions

```py
def grayscale_rgb(r: float, g: float, b: float) -> float:
    """Convert RGB to grayscale using standard luminance weights"""

def grayscale_rgba(r: float, g: float, b: float, a: float) -> float:
    """Convert RGBA to grayscale, preserving alpha"""

def grayscale_vec3(color: vec3) -> float:
    """Convert vec3 color to grayscale"""

def grayscale_vec4(color: vec4) -> float:
    """Convert vec4 color to grayscale, preserving alpha"""

def rgb_to_hsv(r: float, g: float, b: float) -> vec3:
    """Convert RGB to HSV. Returns vec3(hue, saturation, value)"""

def rgb_to_hsv_vec3(color: vec3) -> vec3:
    """Convert vec3 RGB to HSV"""

def hsv_to_rgb(h: float, s: float, v: float) -> vec3:
    """Convert HSV to RGB. H in [0, 360], S and V in [0, 1]"""
    
def hsv_to_rgb_vec3(hsv: vec3) -> vec3:
    """Convert vec3 HSV to RGB"""

def rgb_to_hsl(r: float, g: float, b: float) -> vec3:
    """Convert RGB to HSL. Returns vec3(hue, saturation, lightness)"""

def rgb_to_hsl_vec3(color: vec3) -> vec3:
    """Convert vec3 RGB to HSL"""

def hsl_to_rgb(h: float, s: float, l: float) -> vec3:
    """Convert HSL to RGB. H in [0, 360], S and L in [0, 1]"""

def hsl_to_rgb_vec3(hsl: vec3) -> vec3:
    """Convert vec3 HSL to RGB"""
```

#### Color Adjustments

```py
def brightness(color: vec3, amount: float) -> vec3:
    """Adjust brightness by adding amount to each channel"""

def contrast(color: vec3, amount: float) -> vec3:
    """Adjust contrast. amount = 1.0 is no change, < 1.0 reduces, > 1.0 increases"""

def saturation(color: vec3, amount: float) -> vec3:
    """Adjust saturation. amount = 1.0 is no change, 0.0 is grayscale"""

def hue_shift(color: vec3, degrees: float) -> vec3:
    """Shift hue by degrees (0-360)"""

def invert(color: vec3) -> vec3:
    """Invert color"""
```

#### Blending

```py
def blend_add(base: vec3, blend: vec3, opacity: float) -> vec3:
    """Additive blend mode"""

def blend_subtract(base: vec3, blend: vec3, opacity: float) -> vec3:
    """Subtractive blend mode"""

def blend_difference(base: vec3, blend: vec3, opacity: float) -> vec3:
    """Difference blend mode"""
```

#### Effects

```py
def sepia(color: vec3) -> vec3:
    """Apply sepia effect"""

def posterize(color: vec3, threshold: float) -> vec3:
    """Posterize color to threshold levels"""

def apply_tint(color: vec3, tint: vec3, amount: float) -> vec3:
    """Apply a tint to the color"""
```

#### Color Utils

```py
def luminance(color: vec3) -> float:
    """Calculate luminance (same as grayscale_vec3)"""

def perceived_brightness(color: vec3) -> float:
    """Calculate perceived brightness using sRGB weights"""

def color_distance(c1: vec3, c2: vec3) -> float:
    """Calculate Euclidean distance between two colors"""

def threshold_color(color: vec3, threshold: float) -> vec3:
    """Threshold color to black or white based on luminance"""
```

#### Palettes

```py
def rainbow_gradient(t: float) -> vec3:
    """Generate rainbow color from t [0, 1]"""

def heat_map(t: float) -> vec3:
    """Generate heat map color from t [0, 1] (black -> red -> yellow -> white)"""

def gradient_3_colors(t: float, c1: vec3, c2: vec3, c3: vec3) -> vec3:
    """Interpolate between three colors"""

def gradient_4_colors(t: float, c1: vec3, c2: vec3, c3: vec3, c4: vec3) -> vec3:
    """Interpolate between four colors"""
```