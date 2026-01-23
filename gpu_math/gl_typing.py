import numpy as np
from typing import Dict, List
from panda3d.core import Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f


    


class IOTypes:
    buffer  = "buffer"      # Auto-indexed vectorized buffer (element-wise operations)
    array   = "array"       # Non-indexed buffer (manual indexing)
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

GL_GLOBALS = {
    "gl_GlobalInvocationID" : GLTypes.vec3,
}

class NP_GLTypes:
    float   = (np.float32,  GLTypes.float)
    double  = (np.float64,  GLTypes.double)
    int     = (np.int32,    GLTypes.int)
    uint    = (np.uint32,   GLTypes.uint)
    bool    = (np.bool_,    GLTypes.bool)

class Vec_GLTypes:
    vec2    = (Vec2, GLTypes.vec2)
    vec3    = (Vec3, GLTypes.vec3)
    vec4    = (Vec4, GLTypes.vec4)

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
        'vec4': 7
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
        
        rank1 = cls.TYPE_HIERARCHY.get(type1, 3)
        rank2 = cls.TYPE_HIERARCHY.get(type2, 3)
        return type1 if rank1 > rank2 else type2
    
    @classmethod
    def infer_operator_type(cls, op_name: str, input_types: List[str]) -> str:
        """Infer the result type of an operation based on input types"""
        
        # Handle array operations
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
            if input_types and input_types[0] in ['int', 'uint']:
                return 'uint' if input_types[0] == 'uint' else 'int'
            return 'int'  # Default to int for bitwise operations
        
        # Integer-only operators
        if op_name in cls.INTEGRAL_OPS:
            if input_types and input_types[0] in ['int', 'uint']:
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

GLSL_TO_NP = {
    GLTypes.float:  np.float32,
    GLTypes.double: np.float64,
    GLTypes.int:    np.int32,
    GLTypes.uint:   np.uint32,
    GLTypes.bool:   np.bool_,
    GLTypes.vec2:   np.float32,
    GLTypes.vec3:   np.float32,
    GLTypes.vec4:   np.float32,
    GLTypes.void:   None,
}

NP_TO_GLSL = {
    bool:       GLTypes.bool,
    np.bool_:   GLTypes.bool,
    np.int32:   GLTypes.int,
    np.uint32:  GLTypes.uint,
    np.float32: GLTypes.float,
    np.float64: GLTypes.double,
}

VEC_TO_GLSL = {
    Vec2:       GLTypes.vec2,
    Vec3:       GLTypes.vec3,
    Vec4:       GLTypes.vec4,
    LVecBase2f: GLTypes.vec2,
    LVecBase3f: GLTypes.vec3,
    LVecBase4f: GLTypes.vec4,
}