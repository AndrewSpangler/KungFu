import numpy as np
from typing import Dict, Optional, Union
from panda3d.core import (
    NodePath, Shader, ShaderAttrib, ShaderBuffer,
    GeomEnums, ComputeNode, GraphicsPipeSelection,
    FrameBufferProperties, WindowProperties, GraphicsPipe,
    Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f
)
from .composition import create_shader
from .declaration import CONFIG
from .cast_buffer import CastBuffer
from .fft.fft_radix16 import Radix16FFT
from .shader_compiler import ShaderCompiler
from .graph_compiler import gpu_kernel, inline_always
from .gl_typing import GLTypes, NP_GLTypes, Vec_GLTypes, GLSL_TO_NP, NP_TO_GLSL, VEC_TO_GLSL

class GPUMath:
    def __init__(self, base, headless=False, addons=[Radix16FFT]):
        self.base = base
        self.op_registry = {}
        self.fused_cache = {}
        self.code_cache = {}

        self.addons = [addon(base, headless=headless) for addon in addons]

        for name, arity_map in CONFIG.items():
            self.op_registry[name] = {}
            for arity, (expr, overloads) in arity_map.items():
                self.op_registry[name][arity] = {}
                for arg_types, res_type in overloads:
                    arg_types_with_storage = [(t, 'buffer') for t in arg_types]
                    code = create_shader(expr, arg_types_with_storage, res_type)
                    node = self._compile(code, res_type)
                    self.op_registry[name][arity][tuple(arg_types)] = node
            
            setattr(self, name, lambda *args, n=name: self._dispatch(n, *args))
        
        if headless:
            self._setup_headless()

    def _compile(self, code, res_type):
        shader = Shader.make_compute(Shader.SL_GLSL, code)
        node = NodePath(ComputeNode("math_node"))
        node.set_shader(shader)
        if res_type != 'void':
            node.set_python_tag("dtype", GLSL_TO_NP.get(res_type, np.float32))
        return node

    def _dispatch(self, op_name, *args, out: Optional[Union[np.ndarray, CastBuffer]] = None):
        n_items = 1
        for a in args:
            if isinstance(a, (CastBuffer, np.ndarray)):
                n_items = len(a)
                break

        buffers = []
        for a in args:
            if isinstance(a, (int, float, bool, np.generic)):
                buffers.append(self.push(np.full(n_items, a)))
            elif isinstance(a, np.ndarray):
                buffers.append(self.push(a))
            else:
                buffers.append(a)

        arity = len(buffers)
        sig = tuple(NP_TO_GLSL.get(b.cast, "float") for b in buffers)

        if arity not in self.op_registry[op_name]:
            raise TypeError(f"Operator '{op_name}' does not support {arity} arguments.")
        
        if sig not in self.op_registry[op_name][arity]:
            sig = tuple("float" for _ in range(arity))
            if sig not in self.op_registry[op_name][arity]:
                raise TypeError(f"No variant for '{op_name}' matching {sig}")

        node = self.op_registry[op_name][arity][sig]
        res_dtype = node.get_python_tag("dtype")

        if out is not None:
            if isinstance(out, CastBuffer):
                result_buffer = out.buffer
            elif isinstance(out, np.ndarray):
                if len(out) != n_items:
                    raise ValueError(f"Output array length {len(out)} doesn't match input length {n_items}")
                result_buffer = self.push(out).buffer
            else:
                raise TypeError("out must be CastBuffer or numpy array")
        else:
            res_size = n_items * np.dtype(res_dtype).itemsize
            result_buffer = ShaderBuffer("DR", res_size, GeomEnums.UH_stream)

        for i, b in enumerate(buffers):
            node.set_shader_input(f"D{i}", b.buffer)
        
        node.set_shader_input("DR", result_buffer)
        node.set_shader_input("nItems", int(n_items))

        self.base.graphics_engine.dispatch_compute(
            ((n_items + 63) // 64, 1, 1), 
            node.get_attrib(ShaderAttrib), 
            self.base.win.get_gsg()
        )
        
        if out is None:
            return CastBuffer(result_buffer, n_items, cast=res_dtype)
        elif isinstance(out, CastBuffer):
            return out
        else:
            return self.fetch(CastBuffer(result_buffer, n_items, cast=res_dtype))

    def compile_fused(self, func, debug=False):
        import inspect
        
        # Handle bound methods by getting the underlying function
        is_method = inspect.ismethod(func)
        actual_func = func.__func__ if is_method else func
        
        if not hasattr(actual_func, '_compute_graph') and not hasattr(actual_func, '_needs_transpilation'):
            raise ValueError("Function must be decorated with @gpu_kernel")
        
        # Transpile if not already done (For kernels wrapped in static_constant decorators)
        if getattr(actual_func, '_needs_transpilation', False):
            from .graph_compiler import _transpile_kernel
            _transpile_kernel(func)
        
        graph = actual_func._compute_graph
        arg_names = actual_func._arg_names
        type_hints = getattr(actual_func, '_type_hints', {})
        storage_hints = getattr(actual_func, '_storage_hints', {})
        explicit_types = getattr(actual_func, '_explicit_types', {})
        
        cache_key = (actual_func.__name__, id(graph))
        
        if cache_key not in self.fused_cache:
            input_types = {name: type_hints.get(name, 'float') for name in arg_names}
            
            compiler = ShaderCompiler(graph, input_types, explicit_types)
            code, res_type = compiler.compile()
            
            if 'res' in type_hints:
                res_type = type_hints['res']
                        
            shader = Shader.make_compute(Shader.SL_GLSL, code)
            node = NodePath(ComputeNode(f"fused_{actual_func.__name__}"))
            node.set_shader(shader)
            
            if res_type != 'void':
                node.set_python_tag("dtype", GLSL_TO_NP.get(res_type, np.float32))
            
            self.fused_cache[cache_key] = (node, arg_names, res_type, storage_hints)
            self.code_cache[cache_key] = code
        
        node, arg_names, res_type, storage_hints = self.fused_cache[cache_key]
        
        def executor(*args, out: Optional[Union[np.ndarray, CastBuffer]] = None, n_items: Optional[int] = None, **uniform_values):
            if len(args) != len(arg_names):
                raise ValueError(f"Expected {len(arg_names)} arguments, got {len(args)}")
            
            auto_n_items = None
            buffers = []
            
            for i, (arg_name, arg) in enumerate(zip(arg_names, args)):
                storage = storage_hints.get(arg_name, 'buffer')
                
                if storage == 'uniform':
                    continue
                
                if storage == 'array':
                    # Array storage - no auto n_items determination, just pass buffer
                    if isinstance(arg, CastBuffer):
                        buffers.append((i, arg))
                    elif isinstance(arg, np.ndarray):
                        buffers.append((i, self.push(arg)))
                    else:
                        raise TypeError(f"Array parameter '{arg_name}' must be CastBuffer or numpy array, got {type(arg)}")
                
                elif storage == 'buffer':
                    # Vectorized buffer storage - determine n_items from length
                    if isinstance(arg, CastBuffer):
                        if auto_n_items is None:
                            auto_n_items = len(arg)
                        buffers.append((i, arg))
                    elif isinstance(arg, np.ndarray):
                        if auto_n_items is None:
                            auto_n_items = len(arg)
                        buffers.append((i, self.push(arg)))
                    elif not isinstance(arg, (int, float, bool, np.generic, Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f)):
                        raise TypeError(f"Unsupported argument type for '{arg_name}': {type(arg)}")
            
            # Determine actual n_items to use
            if n_items is not None:
                # Explicit n_items takes precedence
                actual_n_items = n_items
            elif auto_n_items is not None:
                # Auto-determined from buffer lengths
                actual_n_items = auto_n_items
            else:
                # No buffers and no explicit n_items - single work item
                actual_n_items = 1
            
            # Expand scalar arguments for vectorized buffers
            for i, (arg_name, arg) in enumerate(zip(arg_names, args)):
                if storage_hints.get(arg_name, 'buffer') == 'buffer':
                    if any(idx == i for idx, _ in buffers):
                        continue
                    
                    if isinstance(arg, (int, float, bool, np.generic)):
                        if isinstance(arg, (bool, np.bool_)):
                            dtype = np.bool_
                        elif isinstance(arg, (int, np.integer)):
                            dtype = np.int32
                        elif isinstance(arg, (float, np.floating)):
                            dtype = np.float32
                        else:
                            dtype = type(arg)
                        
                        if res_type.startswith('vec'):
                            dim = int(res_type[3])
                            scalar_array = np.full((actual_n_items, dim), arg, dtype=dtype)
                        else:
                            scalar_array = np.full(actual_n_items, arg, dtype=dtype)
                        
                        buffers.append((i, self.push(scalar_array)))
            
            buffers.sort(key=lambda x: x[0])
            buffer_objects = [buf for _, buf in buffers]
            
            if res_type != 'void':
                res_dtype = GLSL_TO_NP.get(res_type, np.float32)
                
                if out is not None:
                    if isinstance(out, CastBuffer):
                        result_buffer = out.buffer
                    elif isinstance(out, np.ndarray):
                        if len(out) != actual_n_items:
                            if res_type.startswith('vec'):
                                dim = int(res_type[3])
                                if len(out) != actual_n_items * dim:
                                    raise ValueError(f"Output array length {len(out)} doesn't match expected length {actual_n_items * dim}")
                            else:
                                raise ValueError(f"Output array length {len(out)} doesn't match expected length {actual_n_items}")
                        result_buffer = self.push(out).buffer
                    else:
                        raise TypeError("out must be CastBuffer or numpy array")
                else:
                    if res_type.startswith('vec'):
                        dim = int(res_type[3])
                        res_size = actual_n_items * dim * np.dtype(res_dtype).itemsize
                    else:
                        res_size = actual_n_items * np.dtype(res_dtype).itemsize
                    
                    result_buffer = ShaderBuffer("DR", res_size, GeomEnums.UH_stream)
            
            for i, buf in enumerate(buffer_objects):
                node.set_shader_input(f"D{i}", buf.buffer)
            
            node.set_shader_input("nItems", int(actual_n_items))
            
            for uniform_name, uniform_value in uniform_values.items():
                node.set_shader_input(uniform_name, uniform_value)
            
            for i, arg_name in enumerate(arg_names):
                if storage_hints.get(arg_name) == 'uniform':
                    if i < len(args):
                        uniform_val = args[i]
                        if isinstance(uniform_val, (Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f)):
                            node.set_shader_input(arg_name, uniform_val)
                        elif isinstance(uniform_val, (int, float, bool, np.generic)):
                            node.set_shader_input(arg_name, uniform_val)
                        else:
                            node.set_shader_input(arg_name, uniform_val)
            
            if res_type != 'void':
                node.set_shader_input("DR", result_buffer)
            
            try:
                self.base.graphics_engine.dispatch_compute(
                    ((actual_n_items + 63) // 64, 1, 1),
                    node.get_attrib(ShaderAttrib),
                    self.base.win.get_gsg()
                )
            except Exception as e:
                if debug:
                    print("Shader execution error:", e)
                    print(self.code_cache[cache_key])
                raise
            
            if res_type == 'void':
                return None
            elif out is not None:
                if isinstance(out, CastBuffer):
                    return out
                else:
                    return self.fetch(CastBuffer(result_buffer, actual_n_items, cast=res_dtype))
            else:
                if res_type.startswith('vec'):
                    dim = int(res_type[3])
                    return CastBuffer(result_buffer, actual_n_items * dim, cast=res_dtype)
                else:
                    return CastBuffer(result_buffer, actual_n_items, cast=res_dtype)
        
        if debug:
            print("Generated GLSL code:")
            print(self.code_cache[cache_key])

        return executor

    def _compile_fused_shader(self, code: str, res_type: str, uniform_types: Dict[str, str]):
        shader = Shader.make_compute(Shader.SL_GLSL, code)
        node = NodePath(ComputeNode("fused_kernel"))
        node.set_shader(shader)
        if res_type != 'void':
            node.set_python_tag("dtype", GLSL_TO_NP.get(res_type, np.float32))
        node.set_python_tag("uniform_types", uniform_types)
        return node

    def kernel(self, func):
        compiled = None
        
        def wrapper(*args, out=None, **kwargs):
            nonlocal compiled
            if compiled is None:
                compiled = self.compile_fused(func)
            return compiled(*args, out=out, **kwargs)
        
        wrapper._original_func = func
        return wrapper

    def push(self, data):
        if not isinstance(data, np.ndarray):
            data = np.array([data])
        sbuf = ShaderBuffer("Data", data.tobytes(), GeomEnums.UH_stream)
        return CastBuffer(sbuf, len(data), cast=data.dtype.type)

    def fetch(self, handle):
        gsg = self.base.win.get_gsg()
        raw = self.base.graphics_engine.extract_shader_buffer_data(handle.buffer, gsg)
        if handle.cast == np.bool_:
            return np.frombuffer(raw, dtype=np.int32).astype(np.bool_)
        return np.frombuffer(raw, dtype=handle.cast)

    def _setup_headless(self):
        pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
        fb_prop = FrameBufferProperties()
        win_prop = WindowProperties.size(1, 1)
        self.base.win = self.base.graphics_engine.make_output(
            pipe, "math_headless", 0, fb_prop, win_prop, GraphicsPipe.BF_refuse_window
        )