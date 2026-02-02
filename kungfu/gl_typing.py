import ast
import numpy as np
from enum import Enum
from typing import Dict, List
from panda3d.core import (
    Vec2,
    Vec3,
    Vec4,
    LVecBase2f,
    LVecBase3f,
    LVecBase4f,
    Texture,
    SamplerState
)

from .constants import (
    TextureType,
    TextureFilter,
    TextureFormat,
    WrapMode,
    TextureComponentType
)

class IOTypes:
    buffer  = "buffer"      # Auto-indexed vectorized buffer (element-wise operations)
    array   = "array"       # Non-indexed buffer (manual indexing) - accessible as an array rather than element
    uniform = "uniform"
    texture = "texture"

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
    
    # Texture / mats / samplers (no promotion)
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

GLSL_CONSTRUCTORS = set()
for attr_name in dir(GLTypes):
    if not attr_name.startswith('__'):
        attr_value = getattr(GLTypes, attr_name)
        if isinstance(attr_value, str):
            GLSL_CONSTRUCTORS.add(attr_value)

class ShaderType(Enum):
    """Supported shader types"""
    COMPUTE = "compute"
    FRAGMENT = "fragment"
    VERTEX = "vertex"
    GEOMETRY = "geometry"

class ShaderStage(Enum):
    """Shader pipeline stages"""
    VERTEX = "vertex"
    GEOMETRY = "geometry"
    FRAGMENT = "fragment"

SHADER_BUILTINS = {
    ShaderType.FRAGMENT: {
        # Inputs
        'gl_FragCoord': 'vec4',
        'gl_FrontFacing': 'bool',
        'gl_PointCoord': 'vec2',
        'gl_ClipDistance': 'float[]',
        'gl_PrimitiveID': 'int',
        # Outputs
        'gl_FragColor': 'vec4',  # GLSL < 130
        'gl_FragData': 'vec4[]', # GLSL < 130
    },
    ShaderType.VERTEX: {
        # Inputs
        'gl_VertexID': 'int',
        'gl_InstanceID': 'int',
        # Outputs
        'gl_Position': 'vec4',
        'gl_PointSize': 'float',
        'gl_ClipDistance': 'float[]',
    },
    ShaderType.GEOMETRY: {
        # Inputs
        'gl_PrimitiveIDIn': 'int',
        'gl_InvocationID': 'int',
        # Outputs
        'gl_Position': 'vec4',
        'gl_PointSize': 'float',
        'gl_ClipDistance': 'float[]',
        'gl_PrimitiveID': 'int',
        'gl_Layer': 'int',
        'gl_ViewportIndex': 'int',
    },
}

class GLComputeShaderInputs:
    gl_NumWorkGroups        = GLTypes.uvec3
    gl_WorkGroupID          = GLTypes.uvec3
    gl_LocalInvocationID    = GLTypes.uvec3
    gl_GlobalInvocationID   = GLTypes.uvec3
    gl_LocalInvocationIndex = GLTypes.uint
    gl_WorkGroupSize        = GLTypes.uvec3

    # 3.3
    gl_MaxVertexAttribs = GLTypes.int
    gl_MaxVertexOutputComponents = GLTypes.int
    gl_MaxVertexUniformComponents = GLTypes.int
    gl_MaxVertexTextureImageUnits = GLTypes.int
    gl_MaxGeometryInputComponents = GLTypes.int
    gl_MaxGeometryOutputComponents = GLTypes.int
    gl_MaxGeometryUniformComponents = GLTypes.int
    gl_MaxGeometryTextureImageUnits = GLTypes.int
    gl_MaxGeometryOutputVertices = GLTypes.int
    gl_MaxGeometryTotalOutputComponents = GLTypes.int
    gl_MaxGeometryVaryingComponents = GLTypes.int
    gl_MaxFragmentInputComponents = GLTypes.int
    gl_MaxDrawBuffers = GLTypes.int
    gl_MaxFragmentUniformComponents = GLTypes.int
    gl_MaxTextureImageUnits1 = GLTypes.int
    gl_MaxClipDistances = GLTypes.int
    gl_MaxCombinedTextureImageUnits = GLTypes.int

    # 4.0
    gl_MaxTessControlInputComponents = GLTypes.int
    gl_MaxTessControlOutputComponents = GLTypes.int
    gl_MaxTessControlUniformComponents = GLTypes.int
    gl_MaxTessControlTextureImageUnits = GLTypes.int
    gl_MaxTessControlTotalOutputComponents = GLTypes.int
    gl_MaxTessEvaluationInputComponents = GLTypes.int
    gl_MaxTessEvaluationOutputComponents = GLTypes.int
    gl_MaxTessEvaluationUniformComponents = GLTypes.int
    gl_MaxTessEvaluationTextureImageUnits = GLTypes.int
    gl_MaxTessPatchComponents = GLTypes.int
    gl_MaxPatchVertices = GLTypes.int
    gl_MaxTessGenLevel = GLTypes.int

    # 4.1
    gl_MaxViewports = GLTypes.int
    gl_MaxVertexUniformVectors = GLTypes.int
    gl_MaxFragmentUniformVectors = GLTypes.int
    gl_MaxVaryingVectors = GLTypes.int

    # 4.2
    gl_MaxVertexImageUniform = GLTypes.int
    gl_MaxVertexAtomicCounter = GLTypes.int
    gl_MaxVertexAtomicCounterBuffer = GLTypes.int
    gl_MaxTessControlImageUniform = GLTypes.int
    gl_MaxTessControlAtomicCounter = GLTypes.int
    gl_MaxTessControlAtomicCounterBuffers = GLTypes.int
    gl_MaxTessEvaluationImageUniforms = GLTypes.int
    gl_MaxTessEvaluationAtomicCounters = GLTypes.int
    gl_MaxTessEvaluationAtomicCounterBuffers = GLTypes.int
    gl_MaxGeometryImageUniforms = GLTypes.int
    gl_MaxGeometryAtomicCounters = GLTypes.int
    gl_MaxGeometryAtomicCounterBuffers = GLTypes.int
    gl_MaxFragmentImageUniforms = GLTypes.int
    gl_MaxFragmentAtomicCounters = GLTypes.int
    gl_MaxFragmentAtomicCounterBuffers = GLTypes.int
    gl_MaxCombinedImageUniforms = GLTypes.int
    gl_MaxCombinedAtomicCounters = GLTypes.int
    gl_MaxCombinedAtomicCounterBuffers = GLTypes.int
    gl_MaxImageUnits = GLTypes.int
    gl_MaxCombinedImageUnitsAndFragmentOutputs = GLTypes.int
    gl_MaxImageSamples = GLTypes.int
    gl_MaxAtomicCounterBindings = GLTypes.int
    gl_MaxAtomicCounterBufferSize = GLTypes.int
    gl_MinProgramTexelOffset = GLTypes.int
    gl_MaxProgramTexelOffset = GLTypes.int
    gl_MaxVertexAtomicCounters = GLTypes.int
    gl_MaxVertexAtomicCounterBuffers = GLTypes.int
    gl_MaxTessControlImageUniforms = GLTypes.int
    gl_MaxTessControlAtomicCounters = GLTypes.int
    gl_MaxTessControlAtomicCounterBuffers = GLTypes.int
    gl_MaxTessEvaluationImageUniforms = GLTypes.int
    gl_MaxTessEvaluationAtomicCounters = GLTypes.int
    gl_MaxTessEvaluationAtomicCounterBuffers = GLTypes.int
    gl_MaxGeometryImageUniforms = GLTypes.int
    gl_MaxGeometryAtomicCounters = GLTypes.int
    gl_MaxGeometryAtomicCounterBuffers = GLTypes.int
    gl_MaxFragmentImageUniforms = GLTypes.int
    gl_MaxFragmentAtomicCounters = GLTypes.int
    gl_MaxFragmentAtomicCounterBuffers = GLTypes.int
    gl_MaxCombinedImageUniforms = GLTypes.int
    gl_MaxCombinedAtomicCounters = GLTypes.int
    gl_MaxCombinedAtomicCounterBuffers = GLTypes.int
    gl_MaxImageUnits = GLTypes.int
    gl_MaxCombinedImageUnitsAndFragmentOutputs = GLTypes.int
    gl_MaxImageSamples = GLTypes.int
    gl_MaxAtomicCounterBindings = GLTypes.int
    gl_MaxAtomicCounterBufferSize = GLTypes.int
    gl_MinProgramTexelOffset = GLTypes.int
    gl_MaxProgramTexelOffset = GLTypes.int
    # 4.3
    gl_MaxComputeWorkGroupCount = GLTypes.ivec3
    gl_MaxComputeWorkGroupSize = GLTypes.ivec3
    gl_MaxComputeUniformComponents = GLTypes.int
    gl_MaxComputeTextureImageUnits = GLTypes.int
    gl_MaxComputeImageUniforms = GLTypes.int
    gl_MaxComputeAtomicCounters = GLTypes.int
    gl_MaxComputeAtomicCounterBuffers = GLTypes.int
    # 4.4
    gl_MaxTransformFeedbackBuffers = GLTypes.int
    gl_MaxTransformFeedbackInterleavedComponents = GLTypes.int

# Generate BUILTIN_VARIABLES from GLComputeShaderInputs
BUILTIN_VARIABLES = {}
for attr_name in dir(GLComputeShaderInputs):
    if not attr_name.startswith('__'):
        attr_value = getattr(GLComputeShaderInputs, attr_name)
        if isinstance(attr_value, str):  # Only include GLTypes strings
            BUILTIN_VARIABLES[attr_name] = attr_value

# Add KungFu built-ins and aliases
KUNGFU_BUILTINS = {
    # Global invocation ID components - most commonly used
    'gid': GLTypes.uint,  # Alias for gl_GlobalInvocationID.x
    'gid_x': GLTypes.uint,  # Explicit component access
    'gid_y': GLTypes.uint,
    'gid_z': GLTypes.uint,
    'gid_xyz': GLTypes.uvec3,  # Full vector
    
    # Work group ID components
    'wgid': GLTypes.uvec3,  # Alias for gl_WorkGroupID
    'wgid_x': GLTypes.uint,
    'wgid_y': GLTypes.uint,
    'wgid_z': GLTypes.uint,
    
    # Local invocation ID components
    'lid': GLTypes.uvec3,  # Alias for gl_LocalInvocationID
    'lid_x': GLTypes.uint,
    'lid_y': GLTypes.uint,
    'lid_z': GLTypes.uint,
    
    # Local invocation index
    'lid_idx': GLTypes.uint,  # Alias for gl_LocalInvocationIndex
    
    # Work group size
    'wg_size': GLTypes.uvec3,  # Alias for gl_WorkGroupSize
    'wg_size_x': GLTypes.uint,
    'wg_size_y': GLTypes.uint,
    'wg_size_z': GLTypes.uint,
    
    # Number of work groups
    'num_wg': GLTypes.uvec3,  # Alias for gl_NumWorkGroups
    'num_wg_x': GLTypes.uint,
    'num_wg_y': GLTypes.uint,
    'num_wg_z': GLTypes.uint,
    
    # Special KungFu variables
    'n_items': GLTypes.uint,  # Number of items to process (from uniform)
    'global_idx': GLTypes.uint,  # Calculated global index for 1D kernels
}

# Add KungFu built-ins to BUILTIN_VARIABLES
BUILTIN_VARIABLES.update(KUNGFU_BUILTINS)

# GLSL type mappings for Python type names
# Python -> GLSL
GLSL_TYPE_MAP = {
    'int': GLTypes.int,
    'uint': GLTypes.uint,
    'float': GLTypes.float,
    'double': GLTypes.double,
    'bool': GLTypes.bool,
    'vec2': GLTypes.vec2,
    'vec3': GLTypes.vec3,
    'vec4': GLTypes.vec4,
    'uvec2': GLTypes.uvec2,
    'uvec3': GLTypes.uvec3,
    'uvec4': GLTypes.uvec4,
    'ivec2': GLTypes.ivec2,
    'ivec3': GLTypes.ivec3,
    'ivec4': GLTypes.ivec4,
}

# GLSL versions
# Todo: add a way to overwrite on engine start eventually
GLSL_VERSION = {
    "default": 430,
    "fragment": 430,
    "vertex": 430,
    "geometry": 430,
    "compute": 430
}

# Reverse mapping from GLSL type to Python type name
GLSL_TYPE_REVERSE_MAP = {v: k for k, v in GLSL_TYPE_MAP.items()}

GLSL_TO_NP = {
    GLTypes.float:  np.float32,
    GLTypes.double: np.float64,
    GLTypes.int:    np.int32,
    GLTypes.uint:   np.uint32,
    GLTypes.bool:   np.bool_,
    GLTypes.vec2:   np.float32,
    GLTypes.vec3:   np.float32,
    GLTypes.vec4:   np.float32,
    GLTypes.uvec2:  np.uint32,
    GLTypes.uvec3:  np.uint32,
    GLTypes.uvec4:  np.uint32,
    GLTypes.ivec2:  np.int32,
    GLTypes.ivec3:  np.int32,
    GLTypes.ivec4:  np.int32,
    GLTypes.void:   None,
}

NP_TO_GLSL = {
    bool:       GLTypes.bool,
    np.bool_:   GLTypes.bool,
    np.int32:   GLTypes.int,
    np.uint32:  GLTypes.uint,
    np.float32: GLTypes.float,
    np.float64: GLTypes.double,
    np.complex64: GLTypes.vec2,
    np.complex128: GLTypes.vec2,
}

VEC_TO_GLSL = {
    Vec2:       GLTypes.vec2,
    Vec3:       GLTypes.vec3,
    Vec4:       GLTypes.vec4,
    LVecBase2f: GLTypes.vec2,
    LVecBase3f: GLTypes.vec3,
    LVecBase4f: GLTypes.vec4,
}

AST_BIN_OP_MAP = {
    ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mult', ast.Div: 'div',
    ast.FloorDiv: 'floordiv', ast.Mod: 'mod', ast.BitAnd: 'and', 
    ast.BitOr: 'or', ast.BitXor: 'xor', ast.LShift: 'lsh', 
    ast.RShift: 'rsh', ast.Pow: 'pow',
}

AST_BIN_SYMBOL_MAP = {
    ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
    ast.FloorDiv: '/', ast.Mod: '%', ast.BitAnd: '&', ast.BitOr: '|',
    ast.BitXor: '^', ast.LShift: '<<', ast.RShift: '>>',
    ast.Pow: '**',
}

AST_UNARY_SYMBOL_MAP = {
    ast.USub: '-', ast.UAdd: '+', ast.Not: '!', ast.Invert: '~',
}

AST_COMPARISON_SYMBOL_MAP = {
    ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>',
    ast.GtE: '>=', ast.Eq: '==', ast.NotEq: '!=',  
}

AST_BOOL_SYMBOL_MAP = {
    ast.And: '&&', ast.Or: '||'
}

SWIZZLES = [
    'x', 'y', 'z', 'w', 'r', 'g', 'b', 'a', 
    'xy', 'xyz', 'xyzw', 'rgb', 'rgba'
]

GLSL_MATH_FUNCTIONS = {
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'pow', 'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
    'abs', 'sign', 'floor', 'ceil', 'fract', 'mod', 'min', 'max',
    'clamp', 'mix', 'step', 'smoothstep', 'length', 'distance',
    'dot', 'cross', 'normalize', 'reflect', 'refract',
}

GLSL_MATRIX_FUNCTIONS = {
    'matrixCompMult', 'transpose', 'determinant', 'inverse',
}

GLSL_TEXTURE_FUNCTIONS = {
    'texture', 'texture2D', 'textureCube', 'textureLod',
}

GLSL_LOGIC_FUNCTIONS = {
    'lessThan', 'lessThanEqual', 'greaterThan', 'greaterThanEqual',
    'equal', 'notEqual', 'any', 'all', 'not',
}

GLSL_TYPE_CONSTRUCTORS = {
    'vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4', 
    'uvec2', 'uvec3', 'uvec4', 'mat2', 'mat3', 'mat4',
    'int', 'float', 'bool', 'uint',
}

ALL_GLSL_FUNCTIONS = (
    GLSL_MATH_FUNCTIONS | GLSL_MATRIX_FUNCTIONS | 
    GLSL_TEXTURE_FUNCTIONS | GLSL_LOGIC_FUNCTIONS
)

OP_TO_GLSL = {
    # Arithmetic
    'add': lambda inputs: f"({' + '.join(inputs)})",
    'sub': lambda inputs: f"({inputs[0]} - {inputs[1]})",
    'mult': lambda inputs: f"({' * '.join(inputs)})",
    'div': lambda inputs: f"({inputs[0]} / {inputs[1]})",
    'floordiv': lambda inputs: f"({inputs[0]} / {inputs[1]})",
    'neg': lambda inputs: f"(-{inputs[0]})",
    'square': lambda inputs: f"({inputs[0]} * {inputs[0]})",
    
    # Comparisons
    'gt': lambda inputs: f"({inputs[0]} > {inputs[1]})",
    'lt': lambda inputs: f"({inputs[0]} < {inputs[1]})",
    'eq': lambda inputs: f"({inputs[0]} == {inputs[1]})",
    'gte': lambda inputs: f"({inputs[0]} >= {inputs[1]})",
    'lte': lambda inputs: f"({inputs[0]} <= {inputs[1]})",
    'neq': lambda inputs: f"({inputs[0]} != {inputs[1]})",
    
    # Logic
    'and': lambda inputs: f"({inputs[0]} & {inputs[1]})",
    'or': lambda inputs: f"({inputs[0]} | {inputs[1]})",
    'logical_and': lambda inputs: f"({inputs[0]} && {inputs[1]})",
    'logical_or': lambda inputs: f"({inputs[0]} || {inputs[1]})",
    'xor': lambda inputs: f"({inputs[0]} ^ {inputs[1]})",
    'bool_not': lambda inputs: f"(!{inputs[0]})",
    'bitwise_not': lambda inputs: f"(~{inputs[0]})",
    
    # Math
    'mod': lambda inputs: f"({inputs[0]} % {inputs[1]})",
    'clamp': lambda inputs: f"clamp({inputs[0]}, {inputs[1]}, {inputs[2]})",
    'avg': lambda inputs: f"(({' + '.join(inputs)}) / {len(inputs)}.0)",
    'is_zero': lambda inputs: f"({inputs[0]} == 0)",
    'lsh': lambda inputs: f"({inputs[0]} << {inputs[1]})",
    'rsh': lambda inputs: f"({inputs[0]} >> {inputs[1]})",
    'abs': lambda inputs: f"abs({inputs[0]})",
    'sqrt': lambda inputs: f"sqrt({inputs[0]})",
    'pow': lambda inputs: f"pow({inputs[0]}, {inputs[1]})",
    'min': lambda inputs: f"min({inputs[0]}, {inputs[1]})",
    'max': lambda inputs: f"max({inputs[0]}, {inputs[1]})",
    'mix': lambda inputs: f"mix({inputs[0]}, {inputs[1]}, {inputs[2]})",
    'step': lambda inputs: f"step({inputs[0]}, {inputs[1]})",
    'smoothstep': lambda inputs: f"smoothstep({inputs[0]}, {inputs[1]}, {inputs[2]})",
    
    # Type operations
    'cast': lambda inputs: f"{inputs[1]}({inputs[0]})",
    'int': lambda inputs: f"int({inputs[0]})",
    'float': lambda inputs: f"float({inputs[0]})",
    'uint': lambda inputs: f"uint({inputs[0]})",
    'bool': lambda inputs: f"bool({inputs[0]})",
    'vec2': lambda inputs: f"vec2({', '.join(inputs)})" if len(inputs) == 2 else f"vec2({inputs[0]})",
    'vec3': lambda inputs: f"vec3({', '.join(inputs)})" if len(inputs) == 3 else f"vec3({inputs[0]})",
    'vec4': lambda inputs: f"vec4({', '.join(inputs)})" if len(inputs) == 4 else f"vec4({inputs[0]})",
    'uvec2': lambda inputs: f"uvec2({', '.join(inputs)})" if len(inputs) == 2 else f"uvec2({inputs[0]})",
    'uvec3': lambda inputs: f"uvec3({', '.join(inputs)})" if len(inputs) == 3 else f"uvec3({inputs[0]})",
    'uvec4': lambda inputs: f"uvec4({', '.join(inputs)})" if len(inputs) == 4 else f"uvec4({inputs[0]})",
    'ivec2': lambda inputs: f"ivec2({', '.join(inputs)})" if len(inputs) == 2 else f"ivec2({inputs[0]})",
    'ivec3': lambda inputs: f"ivec3({', '.join(inputs)})" if len(inputs) == 3 else f"ivec3({inputs[0]})",
    'ivec4': lambda inputs: f"ivec4({', '.join(inputs)})" if len(inputs) == 4 else f"ivec4({inputs[0]})",
    
    # Access operations
    'subscript': lambda inputs: f"{inputs[0]}[int({inputs[1]})]",
    'subscript_2d': lambda inputs: f"{inputs[0]}[int({inputs[1]})][int({inputs[2]})]",
    'swizzle': lambda inputs: f"{inputs[0]}.{inputs[1]}",
    'subscript_assign': lambda inputs: f"{inputs[0]}[int({inputs[1]})] = {inputs[2]}",
    'subscript_assign_2d': lambda inputs: f"{inputs[0]}[int({inputs[1]})][int({inputs[2]})] = {inputs[3]}",
    'function_call': lambda inputs: f"{inputs[0]}({', '.join(inputs[1:])})",
    
    # Complex math
    'cmul': lambda inputs: f"vec2({inputs[0]}.x * {inputs[1]}.x - {inputs[0]}.y * {inputs[1]}.y, {inputs[0]}.x * {inputs[1]}.y + {inputs[0]}.y * {inputs[1]}.x)",
}

class TypeRules:
    """Type system rules for GLSL type inference and promotion"""
    
    TYPE_HIERARCHY = {
        'bool': 0,
        'int': 1,
        'uint': 2,
        'float': 3,
        'double': 4,
        'vec2': 5,
        'vec3': 6,
        'vec4': 7,
        'ivec2': 8,
        'ivec3': 9,
        'ivec4': 10,
        'uvec2': 11,
        'uvec3': 12,
        'uvec4': 13,
    }
    
    TYPE_PROMOTION_MATRIX = {
    (GLTypes.bool,    GLTypes.bool  )   : GLTypes.bool,
    (GLTypes.bool,    GLTypes.int   )   : GLTypes.int,
    (GLTypes.bool,    GLTypes.uint  )   : GLTypes.uint,
    (GLTypes.bool,    GLTypes.float )   : GLTypes.float,
    (GLTypes.bool,    GLTypes.double)   : GLTypes.double, 
    (GLTypes.int,     GLTypes.int   )   : GLTypes.int,
    (GLTypes.int,     GLTypes.uint  )   : GLTypes.uint,
    (GLTypes.int,     GLTypes.float )   : GLTypes.float,
    (GLTypes.int,     GLTypes.double)   : GLTypes.double,
    (GLTypes.uint,    GLTypes.uint  )   : GLTypes.uint,
    (GLTypes.uint,    GLTypes.float )   : GLTypes.float,
    (GLTypes.uint,    GLTypes.double)   : GLTypes.double,
    (GLTypes.float,   GLTypes.float )   : GLTypes.float,
    (GLTypes.float,   GLTypes.double)   : GLTypes.double,
    (GLTypes.double,  GLTypes.double)   : GLTypes.double,
    # Vectors - basic promotion (not 100% coverage)
    (GLTypes.float,   GLTypes.vec2)     : GLTypes.vec2,
    (GLTypes.float,   GLTypes.vec3)     : GLTypes.vec3,
    (GLTypes.float,   GLTypes.vec4)     : GLTypes.vec4,
    (GLTypes.vec2,    GLTypes.vec2)     : GLTypes.vec2,
    (GLTypes.vec3,    GLTypes.vec3)     : GLTypes.vec3,
    (GLTypes.vec4,    GLTypes.vec4)     : GLTypes.vec4,
    # Integer vectors
    (GLTypes.int,     GLTypes.ivec2)    : GLTypes.ivec2,
    (GLTypes.int,     GLTypes.ivec3)    : GLTypes.ivec3,
    (GLTypes.int,     GLTypes.ivec4)    : GLTypes.ivec4,
    (GLTypes.uint,    GLTypes.uvec2)    : GLTypes.uvec2,
    (GLTypes.uint,    GLTypes.uvec3)    : GLTypes.uvec3,
    (GLTypes.uint,    GLTypes.uvec4)    : GLTypes.uvec4,
    (GLTypes.ivec2,   GLTypes.ivec2)    : GLTypes.ivec2,
    (GLTypes.ivec3,   GLTypes.ivec3)    : GLTypes.ivec3,
    (GLTypes.ivec4,   GLTypes.ivec4)    : GLTypes.ivec4,
    (GLTypes.uvec2,   GLTypes.uvec2)    : GLTypes.uvec2,
    (GLTypes.uvec3,   GLTypes.uvec3)    : GLTypes.uvec3,
    (GLTypes.uvec4,   GLTypes.uvec4)    : GLTypes.uvec4,

    # Vector-scalar promotions
    (GLTypes.vec2, GLTypes.float): GLTypes.vec2,
    (GLTypes.vec3, GLTypes.float): GLTypes.vec3,
    (GLTypes.vec4, GLTypes.float): GLTypes.vec4,
    (GLTypes.uvec2, GLTypes.uint): GLTypes.uvec2,
    (GLTypes.uvec3, GLTypes.uint): GLTypes.uvec3,
    (GLTypes.uvec4, GLTypes.uint): GLTypes.uvec4,
    (GLTypes.ivec2, GLTypes.int): GLTypes.ivec2,
    (GLTypes.ivec3, GLTypes.int): GLTypes.ivec3,
    (GLTypes.ivec4, GLTypes.int): GLTypes.ivec4,
    
    # Vector-vector promotions (must be same dimension)
    (GLTypes.vec2, GLTypes.vec2): GLTypes.vec2,
    (GLTypes.vec3, GLTypes.vec3): GLTypes.vec3,
    (GLTypes.vec4, GLTypes.vec4): GLTypes.vec4,
    (GLTypes.uvec2, GLTypes.uvec2): GLTypes.uvec2,
    (GLTypes.uvec3, GLTypes.uvec3): GLTypes.uvec3,
    (GLTypes.uvec4, GLTypes.uvec4): GLTypes.uvec4,
    (GLTypes.ivec2, GLTypes.ivec2): GLTypes.ivec2,
    (GLTypes.ivec3, GLTypes.ivec3): GLTypes.ivec3,
    (GLTypes.ivec4, GLTypes.ivec4): GLTypes.ivec4,
    
    # Reverse direction promotions for commutative operations
    (GLTypes.uint,    GLTypes.int   )   : GLTypes.uint,
    (GLTypes.int,     GLTypes.bool  )   : GLTypes.int,
    (GLTypes.uint,    GLTypes.bool  )   : GLTypes.uint,
    (GLTypes.float,   GLTypes.bool  )   : GLTypes.float,
    (GLTypes.double,  GLTypes.bool  )   : GLTypes.double,
    (GLTypes.float,   GLTypes.int   )   : GLTypes.float,
    (GLTypes.double,  GLTypes.int   )   : GLTypes.double,
    (GLTypes.float,   GLTypes.uint  )   : GLTypes.float,
    (GLTypes.double,  GLTypes.uint  )   : GLTypes.double,
    (GLTypes.double,  GLTypes.float )   : GLTypes.double,
    (GLTypes.vec2,    GLTypes.float )   : GLTypes.vec2,
    (GLTypes.vec3,    GLTypes.float )   : GLTypes.vec3,
    (GLTypes.vec4,    GLTypes.float )   : GLTypes.vec4,
    (GLTypes.ivec2,   GLTypes.int   )   : GLTypes.ivec2,
    (GLTypes.ivec3,   GLTypes.int   )   : GLTypes.ivec3,
    (GLTypes.ivec4,   GLTypes.int   )   : GLTypes.ivec4,
    (GLTypes.uvec2,   GLTypes.uint  )   : GLTypes.uvec2,
    (GLTypes.uvec3,   GLTypes.uint  )   : GLTypes.uvec3,
    (GLTypes.uvec4,   GLTypes.uint  )   : GLTypes.uvec4,
    }
    
    # Operators that always return bool
    BOOL_RETURN_OPS = {
        'gt', 'lt', 'eq', 'gte',
        'lte', 'neq', 'is_zero', 'bool', 'bool_not',
        'logical_and', 'logical_or'
    }
    
    # Operators that return the same type as input (or promoted)
    SAME_TYPE_OPS = {
        'add', 'sub', 'mult', 'neg', 'square',
        'abs', 'sign', 'bitwise_not', 'floor',
        'ceil', 'fract', 'round', 'min', 'max'
    }
    
    # Operators that always return float (or double for vectors)
    FLOAT_RETURN_OPS = {
        'avg', 'sqrt', 'sin', 'cos',
        'tan', 'asin', 'acos', 'atan', 'exp',
        'log', 'pow', 'mix', 'step', 'smoothstep'
    }
    
    # Bitwise operators (return int/uint based on input)
    BITWISE_OPS = {
        'and', 'or', 'xor', 'lsh', 'rsh', 
        'bitwise_and', 'bitwise_or', 'bitwise_xor',
        'bitshift_left', 'bitshift_right'
    }
    
    # Integer-only operators (return int)
    INTEGRAL_OPS = {'mod', 'floordiv'}
    
    @classmethod
    def promote_types(cls, type1: str, type2: str) -> str:
        """Promote two types to a common type following GLSL rules"""
        type1 = type1.lower()
        type2 = type2.lower()
        
        if type1 == type2:
            return type1
        
        # Try both orderings in the promotion matrix
        key1 = (type1, type2)
        key2 = (type2, type1)
        
        if key1 in cls.TYPE_PROMOTION_MATRIX:
            return cls.TYPE_PROMOTION_MATRIX[key1]
        elif key2 in cls.TYPE_PROMOTION_MATRIX:
            return cls.TYPE_PROMOTION_MATRIX[key2]
        
        # Handle vector-scalar promotions
        if type1.startswith(('vec', 'uvec', 'ivec')) and type2 in ['float', 'int', 'uint']:
            # scalar with vector - return vector type
            return type1
        elif type2.startswith(('vec', 'uvec', 'ivec')) and type1 in ['float', 'int', 'uint']:
            # scalar with vector - return vector type
            return type2
        
        rank1 = cls.TYPE_HIERARCHY.get(type1, 3)
        rank2 = cls.TYPE_HIERARCHY.get(type2, 3)
        return type1 if rank1 > rank2 else type2
    
    @classmethod
    def infer_operator_type(cls, op_name: str, input_types: List[str]) -> str:
        """Infer the result type of an operation based on input types"""
        
        # Handle array operations
        
        # Handle complex multiplication (returns vec2)
        if op_name == 'cmul':
            return 'vec2'
        
        # Handle vector constructors
        if op_name in ['vec2', 'vec3', 'vec4', 'uvec2', 'uvec3', 'uvec4', 'ivec2', 'ivec3', 'ivec4']:
            return op_name 

        if op_name in ['subscript', 'swizzle']:
            # For subscript, return element type of array
            if input_types and cls.is_array_type(input_types[0]):
                return cls.get_array_element_type(input_types[0])
            return 'float'  # Default

        # Boolean return operators
        if op_name in cls.BOOL_RETURN_OPS:
            return 'bool'
        
        # Bitwise operators
        if op_name in cls.BITWISE_OPS:
            if input_types:
                # For bitwise ops, return uint if any operand is uint, otherwise int
                has_uint = any(t in ["uint", "uvec2", "uvec3", "uvec4"] for t in input_types)
                if has_uint:
                    return "uint"
                elif input_types[0] in ["int", "uint", "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4"]:
                    return input_types[0]
            return "int"  # Default to int for bitwise operations
        
        # Integer-only operators
        if op_name in cls.INTEGRAL_OPS:
            if input_types and input_types[0] in ['int', 'uint', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4']:
                return input_types[0]
            return 'int'
        
        # Division - special handling for integer division
        if op_name == 'div':
            # If both operands are integer types, keep them as integer
            if (len(input_types) >= 2 and
                input_types[0] in ['int', 'uint', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4'] and
                input_types[1] in ['int', 'uint', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4']):
                # Integer division - promote types
                result_type = input_types[0]
                for t in input_types[1:]:
                    result_type = cls.promote_types(result_type, t)
                return result_type
            # Otherwise, float division
            for t in input_types:
                if t.startswith('vec'):
                    return t
            return 'float'
        
        # Float return operators
        if op_name in cls.FLOAT_RETURN_OPS:
            # For vector operations, return vector type if any input is vector
            for t in input_types:
                if t.startswith('vec'):
                    return t
            return 'float'
        
        # Matrix multiplication special cases
        if op_name == 'mult' and len(input_types) >= 2:
            type1, type2 = input_types[0], input_types[1]
            # mat4 * vec4 = vec4
            if type1.startswith('mat') and type2.startswith('vec'):
                return type2
            # vec4 * mat4 = vec4
            if type1.startswith('vec') and type2.startswith('mat'):
                return type1
            # mat * mat = mat
            if type1.startswith('mat') and type2.startswith('mat'):
                return type1
        
        # Same type operators (promote as needed)
        if op_name in cls.SAME_TYPE_OPS:
            result_type = input_types[0] if input_types else 'float'
            for t in input_types[1:]:
                result_type = cls.promote_types(result_type, t)
            return result_type
        
        # Special cases
        if op_name == 'cast' and len(input_types) > 1:
            return input_types[1]  # Target type
        
        # Default to float
        return 'float'
    
    @classmethod
    def is_array_type(cls, type_str: str) -> bool:
        """Check if a type string represents an array"""
        return '[' in type_str and ']' in type_str
    
    @classmethod
    def get_array_element_type(cls, type_str: str) -> str:
        """Extract the base element type from an array type"""
        if cls.is_array_type(type_str):
            return type_str.split('[')[0]
        return type_str
    
    @classmethod
    def get_array_dimensions(cls, type_str: str) -> List[str]:
        """Extract dimensions from an array type"""
        if not cls.is_array_type(type_str):
            return []
        
        # Extract everything between brackets
        import re
        dims = re.findall(r'\[([^\]]+)\]', type_str)
        return dims
    
    @classmethod
    def is_vector_type(cls, type_str: str) -> bool:
        """Check if a type string represents a vector type"""
        return type_str.startswith(('vec', 'uvec', 'ivec'))
    
    @classmethod
    def get_vector_dimension(cls, type_str: str) -> int:
        """Get the dimension of a vector type"""
        if cls.is_vector_type(type_str):
            try:
                return int(type_str[-1])
            except:
                return 0
        return 0

class Vec_GLTypes:
    vec2    = (Vec2, GLTypes.vec2)
    vec3    = (Vec3, GLTypes.vec3)
    vec4    = (Vec4, GLTypes.vec4)

class NP_GLTypes:
    float   = (np.float32,  GLTypes.float)
    double  = (np.float64,  GLTypes.double)
    int     = (np.int32,    GLTypes.int)
    uint    = (np.uint32,   GLTypes.uint)
    bool    = (np.bool_,    GLTypes.bool)
    vec2    = (np.float32,  GLTypes.vec2)  # Actually needs 2 floats
    vec3    = (np.float32,  GLTypes.vec3)  # Actually needs 2 floats
    vec4    = (np.float32,  GLTypes.vec4)  # Actually needs 2 floats
    complex = (np.complex64, GLTypes.vec2)  # complex maps to vec2

PANDA3D_BUILTINS = {
    'p3d_Vertex': 'vec4',
    'p3d_Normal': 'vec3',
    'p3d_Color': 'vec4',
    'p3d_TexCoord': 'vec2',
    'p3d_TexCoord0': 'vec2',
    'p3d_TexCoord1': 'vec2',
    'p3d_TexCoord2': 'vec2',
    'p3d_TexCoord3': 'vec2',
    'p3d_TexCoord4': 'vec2',
    'p3d_TexCoord5': 'vec2',
    'p3d_TexCoord6': 'vec2',
    'p3d_TexCoord7': 'vec2',
    'p3d_MultiTexCoord0': 'vec2',
    'p3d_MultiTexCoord1': 'vec2',
    'p3d_MultiTexCoord2': 'vec2',
    'p3d_MultiTexCoord3': 'vec2',
    'p3d_MultiTexCoord4': 'vec2',
    'p3d_MultiTexCoord5': 'vec2',
    'p3d_MultiTexCoord6': 'vec2',
    'p3d_MultiTexCoord7': 'vec2',
    
    'p3d_ModelMatrix': 'mat4',
    'p3d_ViewMatrix': 'mat4',
    'p3d_ProjectionMatrix': 'mat4',
    'p3d_ModelViewMatrix': 'mat4',
    'p3d_ModelViewProjectionMatrix': 'mat4',
    'p3d_ModelViewMatrixInverse': 'mat4',
    'p3d_ModelViewProjectionMatrixInverse': 'mat4',
    'p3d_NormalMatrix': 'mat3',
    
    'p3d_Texture0': 'sampler2D',
    'p3d_Texture1': 'sampler2D',
    'p3d_Texture2': 'sampler2D',
    'p3d_Texture3': 'sampler2D',
    
    'p3d_FragColor': 'vec4',
    'p3d_FragData': 'vec4[]',
}

GLSL_BUILTINS = {
    'vertex': {
        'gl_VertexID': 'int',
        'gl_InstanceID': 'int',
        'gl_Position': 'vec4',
        'gl_PointSize': 'float',
        'gl_ClipDistance': 'float[]',
    },
    'fragment': {
        'gl_FragCoord': 'vec4',
        'gl_FrontFacing': 'bool',
        'gl_PointCoord': 'vec2',
        'gl_ClipDistance': 'float[]',
        'gl_PrimitiveID': 'int',
        'gl_FragColor': 'vec4',
        'gl_FragData': 'vec4[]',
    },
    'geometry': {
        'gl_PrimitiveIDIn': 'int',
        'gl_InvocationID': 'int',
        'gl_Position': 'vec4',
        'gl_PointSize': 'float',
        'gl_ClipDistance': 'float[]',
        'gl_PrimitiveID': 'int',
        'gl_Layer': 'int',
        'gl_ViewportIndex': 'int',
    }
}

SHADER_TEMPLATES = {
    ShaderType.FRAGMENT: """#version {version}

{declarations}

void main() {{
{body}
}}""",
    
    ShaderType.VERTEX: """#version {version}

{declarations}

void main() {{
{body}
}}""",
    
    ShaderType.GEOMETRY: """#version {version}

{declarations}

void main() {{
{body}
}}""",
}