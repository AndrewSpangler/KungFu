from .gpu_math import GPUMath
from .engine import GPUEngine, get_engine

from .cast_buffer import CastBuffer
from .graph_compiler import (
    gpu_kernel, inline_always, static_constant,
    _transpile_kernel, shader as shader_decorator
)
from .ast_utils import ASTVisitorBase
from .shader_compiler import ShaderCompiler

from .base_transpiler import BaseTranspiler
from .shader_inputs import ShaderInputManager
from .function_registry import FunctionRegistry
from .shader_functions import ShaderFunction, ShaderFunctionTranspiler
from .graph_compiler import function

from .gl_typing import (
    Vec_GLTypes, NP_GLTypes, IOTypes, GLTypes, 
    GLComputeShaderInputs, BUILTIN_VARIABLES,
    GLSL_MATH_FUNCTIONS, GLSL_TYPE_CONSTRUCTORS,
    ALL_GLSL_FUNCTIONS, OP_TO_GLSL, ShaderType
)

from .helpers import (
    KernelValidator,
    get_op_glsl,
    is_panda3d_builtin,
    is_kungfu_builtin,
    numpy_to_glsl_type,
    get_shader_version,
    get_builtin_variables
)

# Export all GLSL operations
for op_name in [
    'add', 'sub', 'mul', 'div', 'mod', 'pow',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh',
    'exp', 'log', 'exp2', 'log2', 'sqrt', 'inversesqrt',
    'abs', 'sign', 'floor', 'ceil', 'fract',
    'min', 'max', 'clamp', 'mix', 'step', 'smoothstep',
    'length', 'distance', 'dot', 'cross', 'normalize',
    'reflect', 'refract',
    'transpose', 'determinant', 'inverse',
    'vec2', 'vec3', 'vec4',
    'ivec2', 'ivec3', 'ivec4',
    'uvec2', 'uvec3', 'uvec4',
    'int', 'uint', 'float'
]:
    globals()[op_name] = getattr(GPUEngine, op_name) if hasattr(GPUEngine, op_name) else None

from .algorithms import Radix16FFT, Radix2FFT

class ADDONS:
    Radix2FFT = Radix2FFT
    Radix16FFT = Radix16FFT