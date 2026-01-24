import numpy as np
from typing import Dict, List
from panda3d.core import Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f

class IOTypes:
    buffer  = "buffer"      # Auto-indexed vectorized buffer (element-wise operations)
    array   = "array"       # Non-indexed buffer (manual indexing) - accessible as an array rather than element
    uniform = "uniform"
    texture = "texture"

class GLTypes:
    float   = "float"
    double  = "double"
    int     = "int"
    uint    = "uint"
    bool    = "bool"
    void    = "void"
    vec2    = "vec2"
    vec3    = "vec3"
    vec4    = "vec4"
    uvec2   = "uvec2"
    uvec3   = "uvec3"
    uvec4   = "uvec4"
    ivec2   = "ivec2"
    ivec3   = "ivec3"
    ivec4   = "ivec4"

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

# Reverse mapping from GLSL type to Python type name
GLSL_TYPE_REVERSE_MAP = {v: k for k, v in GLSL_TYPE_MAP.items()}

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
    (GLTypes.int,     GLTypes.uint  )   : GLTypes.int,
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
    }
    
    # Operators that always return bool
    BOOL_RETURN_OPS = {
        'gt', 'lt', 'eq', 'gte',
        'lte', 'neq', 'is_zero', 'bool', 'bool_not'
    }
    
    # Operators that return the same type as input (or promoted)
    SAME_TYPE_OPS = {
        'add', 'sub', 'mult', 'neg', 'square',
        'abs', 'sign', 'bitwise_not', 'floor',
        'ceil', 'fract', 'round', 'min', 'max'
    }
    
    # Operators that always return float (or double for vectors)
    FLOAT_RETURN_OPS = {
        'div', 'avg', 'sqrt', 'sin', 'cos',
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
        
        key = tuple(sorted([type1, type2]))
        if key in cls.TYPE_PROMOTION_MATRIX:
            return cls.TYPE_PROMOTION_MATRIX[key]
        
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
            if input_types and input_types[0] in ['int', 'uint', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4']:
                # Return the same integer/vector type as input
                return input_types[0]
            return 'int'  # Default to int for bitwise operations
        
        # Integer-only operators
        if op_name in cls.INTEGRAL_OPS:
            if input_types and input_types[0] in ['int', 'uint', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4']:
                return input_types[0]
            return 'int'
        
        # Float return operators
        if op_name in cls.FLOAT_RETURN_OPS:
            # For vector operations, return vector type if any input is vector
            for t in input_types:
                if t.startswith('vec'):
                    return t
            return 'float'
        
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

GLSL_TO_NP = {
    GLTypes.float:  np.float32,
    GLTypes.double: np.float64,
    GLTypes.int:    np.int32,
    GLTypes.uint:   np.uint32,
    GLTypes.bool:   np.bool_,
    GLTypes.vec2:   np.float32,  # Note: vec2 is 2 float32s
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

# Helper function to check if a variable name is a KungFu built-in
def is_kungfu_builtin(var_name: str) -> bool:
    """Check if a variable name is a KungFu built-in"""
    return var_name in KUNGFU_BUILTINS

# Helper function to get GLSL expression for KungFu built-ins
def get_kungfu_builtin_glsl(var_name: str) -> str:
    """Get the GLSL expression for a KungFu built-in variable"""
    mappings = {
        'gid': 'gl_GlobalInvocationID.x',
        'gid_x': 'gl_GlobalInvocationID.x',
        'gid_y': 'gl_GlobalInvocationID.y',
        'gid_z': 'gl_GlobalInvocationID.z',
        'gid_xyz': 'gl_GlobalInvocationID',
        'wgid': 'gl_WorkGroupID',
        'wgid_x': 'gl_WorkGroupID.x',
        'wgid_y': 'gl_WorkGroupID.y',
        'wgid_z': 'gl_WorkGroupID.z',
        'lid': 'gl_LocalInvocationID',
        'lid_x': 'gl_LocalInvocationID.x',
        'lid_y': 'gl_LocalInvocationID.y',
        'lid_z': 'gl_LocalInvocationID.z',
        'lid_idx': 'gl_LocalInvocationIndex',
        'wg_size': 'gl_WorkGroupSize',
        'wg_size_x': 'gl_WorkGroupSize.x',
        'wg_size_y': 'gl_WorkGroupSize.y',
        'wg_size_z': 'gl_WorkGroupSize.z',
        'num_wg': 'gl_NumWorkGroups',
        'num_wg_x': 'gl_NumWorkGroups.x',
        'num_wg_y': 'gl_NumWorkGroups.y',
        'num_wg_z': 'gl_NumWorkGroups.z',
        'n_items': 'nItems',
        'global_idx': 'gid',  # For 1D kernels, global_idx is just gid
    }
    return mappings.get(var_name, var_name)

class Vec_GLTypes:
    vec2    = (Vec2, GLTypes.vec2)
    vec3    = (Vec3, GLTypes.vec3)
    vec4    = (Vec4, GLTypes.vec4)

def numpy_to_glsl_type(dtype):
    if dtype == np.complex64:
        return GLTypes.vec2
    elif dtype == np.complex128:
        return GLTypes.vec2
    else:
        return NP_TO_GLSL.get(dtype.type if hasattr(dtype, 'type') else dtype, GLTypes.float)

class NP_GLTypes:
    float   = (np.float32,  GLTypes.float)
    double  = (np.float64,  GLTypes.double)
    int     = (np.int32,    GLTypes.int)
    uint    = (np.uint32,   GLTypes.uint)
    bool    = (np.bool_,    GLTypes.bool)
    vec2    = (np.float32,  GLTypes.vec2)  # Actually needs 2 floats
    complex = (np.complex64, GLTypes.vec2)  # complex maps to vec2