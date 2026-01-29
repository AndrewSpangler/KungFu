"""
Enhanced GPU engine with full GLSL operation support
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer,
    GeomEnums, ComputeNode, GraphicsPipeSelection,
    FrameBufferProperties, WindowProperties, GraphicsPipe,
    Vec2, Vec3, Vec4
)
from .gpu_math import GPUMath
from .graph_compiler import gpu_kernel, function
from .gl_typing import GLTypes, NP_GLTypes, Vec_GLTypes

class GPUEngine:
    """Enhanced GPU engine with full GLSL operation support"""
    
    def __init__(self, base=None, headless=False):
        self.base = base
        self.gpu_math = GPUMath(base, headless)
        
        # Operation registry for enhanced operations
        self.ops = {
            # Arithmetic operations
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'mod': lambda a, b: a % b,
            'pow': lambda a, b: a ** b,
            
            # Trigonometric functions
            'sin': lambda x: self._call_unary('sin', x),
            'cos': lambda x: self._call_unary('cos', x),
            'tan': lambda x: self._call_unary('tan', x),
            'asin': lambda x: self._call_unary('asin', x),
            'acos': lambda x: self._call_unary('acos', x),
            'atan': lambda x: self._call_unary('atan', x),
            'atan2': lambda y, x: self._call_binary('atan2', y, x),
            
            # Hyperbolic functions
            'sinh': lambda x: self._call_unary('sinh', x),
            'cosh': lambda x: self._call_unary('cosh', x),
            'tanh': lambda x: self._call_unary('tanh', x),
            
            # Exponential and logarithmic functions
            'exp': lambda x: self._call_unary('exp', x),
            'log': lambda x: self._call_unary('log', x),
            'exp2': lambda x: self._call_unary('exp2', x),
            'log2': lambda x: self._call_unary('log2', x),
            'sqrt': lambda x: self._call_unary('sqrt', x),
            'inversesqrt': lambda x: self._call_unary('inversesqrt', x),
            
            # Common functions
            'abs': lambda x: self._call_unary('abs', x),
            'sign': lambda x: self._call_unary('sign', x),
            'floor': lambda x: self._call_unary('floor', x),
            'ceil': lambda x: self._call_unary('ceil', x),
            'fract': lambda x: self._call_unary('fract', x),
            'min': lambda a, b: self._call_binary('min', a, b),
            'max': lambda a, b: self._call_binary('max', a, b),
            'clamp': lambda x, a, b: self._call_ternary('clamp', x, a, b),
            'mix': lambda x, y, a: self._call_ternary('mix', x, y, a),
            'step': lambda edge, x: self._call_binary('step', edge, x),
            'smoothstep': lambda edge0, edge1, x: self._call_ternary('smoothstep', edge0, edge1, x),
            
            # Geometric functions
            'length': lambda x: self._call_unary('length', x),
            'distance': lambda a, b: self._call_binary('distance', a, b),
            'dot': lambda a, b: self._call_binary('dot', a, b),
            'cross': lambda a, b: self._call_binary('cross', a, b),
            'normalize': lambda x: self._call_unary('normalize', x),
            'reflect': lambda I, N: self._call_binary('reflect', I, N),
            'refract': lambda I, N, eta: self._call_ternary('refract', I, N, eta),
            
            # Matrix functions
            'transpose': lambda m: self._call_unary('transpose', m),
            'determinant': lambda m: self._call_unary('determinant', m),
            'inverse': lambda m: self._call_unary('inverse', m),
            
            # Vector operations
            'vec2': lambda x, y=None: self._create_vec('vec2', x, y),
            'vec3': lambda x, y=None, z=None: self._create_vec('vec3', x, y, z),
            'vec4': lambda x, y=None, z=None, w=None: self._create_vec('vec4', x, y, z, w),

            'ivec2': lambda x, y=None: self._create_vec('ivec2', x, y),
            'ivec3': lambda x, y=None, z=None: self._create_vec('ivec3', x, y, z),
            'ivec4': lambda x, y=None, z=None, w=None: self._create_vec('ivec4', x, y, z, w),

            'uvec2': lambda x, y=None: self._create_vec('uvec2', x, y),
            'uvec3': lambda x, y=None, z=None: self._create_vec('uvec3', x, y, z),
            'uvec4': lambda x, y=None, z=None, w=None: self._create_vec('uvec4', x, y, z, w),
        }
        
        # Register all operations as methods
        for op_name, op_func in self.ops.items():
            setattr(self, op_name, op_func)
    
    def _call_unary(self, op_name, x):
        """Call a unary GLSL operation"""
        @gpu_kernel()
        def kernel(a, res):
            res = getattr(self.gpu_math, op_name)(a)
            return res
        
        return self.gpu_math.compile_fused(kernel)(x)
    
    def _call_binary(self, op_name, a, b):
        """Call a binary GLSL operation"""
        @gpu_kernel()
        def kernel(a, b, res):
            res = getattr(self.gpu_math, op_name)(a, b)
            return res
        
        return self.gpu_math.compile_fused(kernel)(a, b)
    
    def _call_ternary(self, op_name, a, b, c):
        """Call a ternary GLSL operation"""
        @gpu_kernel()
        def kernel(a, b, c, res):
            res = getattr(self.gpu_math, op_name)(a, b, c)
            return res
        
        return self.gpu_math.compile_fused(kernel)(a, b, c)
    
    def _create_vec(self, vec_type, *args):
        """Create a vector"""
        @gpu_kernel()
        def kernel(*components, res):
            if vec_type == 'vec2':
                res = vec2(components[0], components[1])
            elif vec_type == 'uvec2':
                res = uvec2(components[0], components[1])
            elif vec_type == 'ivec2':
                res = ivec2(components[0], components[1])
            elif vec_type == 'vec3':
                res = vec3(components[0], components[1], components[2])
            elif vec_type == 'uvec3':
                res = uvec3(components[0], components[1], components[2])
            elif vec_type == 'ivec3':
                res = ivec3(components[0], components[1], components[2])
            elif vec_type == 'vec4':
                res = vec4(components[0], components[1], components[2], components[3])
            elif vec_type == 'uvec4':
                res = uvec4(components[0], components[1], components[2], components[3])
            elif vec_type == 'ivec4':
                res = ivec4(components[0], components[1], components[2], components[3])
            return res
        
        # Flatten args
        flat_args = []
        for arg in args:
            if arg is not None:
                flat_args.append(arg)
        
        return self.gpu_math.compile_fused(kernel)(*flat_args)
        
    
    def kernel(self, func=None, **kwargs):
        if func is None:
            return lambda f: self.gpu_math.kernel(f, **kwargs)
        return self.gpu_math.kernel(func, **kwargs)
    
    def shader(self, shader_type='fragment', **kwargs):
        """Create a shader"""
        return self.gpu_math.shader(shader_type, **kwargs)
    
    def compile_shader(self, func, debug=False):
        """Compile a shader function"""
        return self.gpu_math.compile_shader(func, debug)
    
    def function(self, param_types=None, return_type=None):
        """Decorator for creating reusable shader functions"""
        return function(param_types, return_type)
    
    def push(self, data):
        """Push data to GPU"""
        return self.gpu_math.push(data)
    
    def fetch(self, handle):
        """Fetch data from GPU"""
        return self.gpu_math.fetch(handle)
    
    def zeros(self, shape, dtype=np.float32):
        """Create a zero array on GPU"""
        data = np.zeros(shape, dtype=dtype)
        return self.push(data)
    
    def ones(self, shape, dtype=np.float32):
        """Create a ones array on GPU"""
        data = np.ones(shape, dtype=dtype)
        return self.push(data)
    
    def arange(self, start, stop=None, step=1, dtype=np.float32):
        """Create a range array on GPU"""
        if stop is None:
            data = np.arange(start, dtype=dtype)
        else:
            data = np.arange(start, stop, step, dtype=dtype)
        return self.push(data)
    
    def linspace(self, start, stop, num=50, dtype=np.float32):
        """Create a linearly spaced array on GPU"""
        data = np.linspace(start, stop, num, dtype=dtype)
        return self.push(data)
    
    def random_uniform(self, shape, low=0.0, high=1.0, dtype=np.float32):
        """Create a random uniform array on GPU"""
        data = np.random.uniform(low, high, shape).astype(dtype)
        return self.push(data)
    
    def random_normal(self, shape, mean=0.0, std=1.0, dtype=np.float32):
        """Create a random normal array on GPU"""
        data = np.random.normal(mean, std, shape).astype(dtype)
        return self.push(data)

# Create a global engine instance
_engine = None

def get_engine(base=None, headless=False):
    """Get or create a global GPU engine instance"""
    global _engine
    if _engine is None:
        if not base:
            raise ValueError("Engine must be supplied a ShowBase instance")
        _engine = GPUEngine(base, headless)
    return _engine

shader = lambda shader_type='fragment', **kwargs: get_engine().shader(shader_type, **kwargs)