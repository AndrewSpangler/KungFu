import numpy as np
import inspect
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
from .shader_compiler import ShaderCompiler
from .graph_compiler import (
    gpu_kernel, inline_always, function,
    _transpile_shader, _transpile_kernel,
    shader as shader_decorator,
    function as function_decorator
)
from .function_registry import FunctionRegistry
from .gl_typing import (
    GLTypes, NP_GLTypes, Vec_GLTypes, GLSL_TO_NP,
    NP_TO_GLSL, VEC_TO_GLSL, ShaderType, IOTypes
)
from .shader_functions import ShaderFunctionTranspiler
from .module_loader import ModuleLoader, import_file

class GPUMath:
    def __init__(self, base, headless=False):
        self.base = base
        self.op_registry = {}
        self.fused_cache = {}
        self.code_cache = {}
        self.function_registry = FunctionRegistry()
                
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
        
        try:
            import kungfu as kungfu_module
        except ImportError:
            raise RuntimeError("Can't import kungfu")
        self.module_loader = ModuleLoader(self, kungfu_module)

        if headless:
            self._setup_headless()

    def import_file(self, filepath: str, **kwargs) -> 'ModuleType':
        return self.module_loader.import_file(filepath, extra_modules=kwargs)

    def reload_file(self, filepath: str) -> 'ModuleType':
        return self.module_loader.reload(filepath)

    def get_imported_module(self, filepath: str) -> 'Optional[ModuleType]':
        return self.module_loader.get_module(filepath)

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
        # Handle bound methods by getting the underlying function
        is_method = inspect.ismethod(func)
        actual_func = func.__func__ if is_method else func
        
        if not hasattr(actual_func, '_compute_graph') and not hasattr(actual_func, '_needs_transpilation'):
            raise ValueError("Function must be decorated with @gpu_kernel")
        
        if getattr(actual_func, '_needs_transpilation', False):
            # Get hints from function attribute
            hints = getattr(actual_func, '_gpu_kernel_hints', {})
            _transpile_kernel(func, hints)
        
        graph = actual_func._compute_graph
        arg_names = actual_func._arg_names
        type_hints = getattr(actual_func, '_type_hints', {})
        storage_hints = getattr(actual_func, '_storage_hints', {})
        explicit_types = getattr(actual_func, '_explicit_types', {})
        
        cache_key = (actual_func.__name__, id(graph))
        
        if cache_key not in self.fused_cache:
            input_types = {name: type_hints.get(name, 'float') for name in arg_names}
            
            local_functions = getattr(actual_func, "_local_functions", {})
            compiler = ShaderCompiler(graph, input_types, explicit_types, local_functions)
            code = compiler.compile()
            
            # Infer result type from graph output or type hints
            if 'res' in type_hints:
                res_type = type_hints['res']
            elif graph.output_var and not graph.has_void_return:
                res_type = compiler.compiler._get_var_type(graph.output_var)
            else:
                res_type = 'void' 
                        
            shader = Shader.make_compute(Shader.SL_GLSL, code)
            node = NodePath(ComputeNode(f"fused_{actual_func.__name__}"))
            node.set_shader(shader)
            
            if res_type != 'void':
                node.set_python_tag("dtype", GLSL_TO_NP.get(res_type, np.float32))
            
            # Store storage hints in the cache
            self.fused_cache[cache_key] = (node, arg_names, res_type, storage_hints, graph.storage_hints)
            self.code_cache[cache_key] = code
        
        node, arg_names, res_type, storage_hints, graph_storage_hints = self.fused_cache[cache_key]
        
        def executor(*args, out: Optional[Union[np.ndarray, CastBuffer]] = None, n_items: Optional[int] = None, **uniform_values):
            if len(args) != len(arg_names):
                raise ValueError(f"Expected {len(arg_names)} arguments, got {len(args)}")
            
            auto_n_items = None
            buffers = []
            uniforms = {}
            
            # Process arguments based on their storage type from the graph
            for i, (arg_name, arg) in enumerate(zip(arg_names, args)):
                storage = graph_storage_hints.get(arg_name, IOTypes.buffer)
                
                if storage == IOTypes.uniform:
                    # Uniforms are handled separately
                    uniforms[arg_name] = arg
                    continue
                
                elif storage == IOTypes.array:
                    # Array storage - no auto n_items determination, just pass buffer
                    if isinstance(arg, CastBuffer):
                        buffers.append((i, f"D{i}", arg))  # Use D{i} for binding
                    elif isinstance(arg, np.ndarray):
                        buffers.append((i, f"D{i}", self.push(arg)))
                    else:
                        raise TypeError(f"Array parameter '{arg_name}' must be CastBuffer or numpy array, got {type(arg)}")
                
                elif storage == IOTypes.buffer:
                    # Vectorized buffer storage - determine n_items from length
                    if isinstance(arg, CastBuffer):
                        if auto_n_items is None:
                            auto_n_items = len(arg)
                        buffers.append((i, f"D{i}", arg))
                    elif isinstance(arg, np.ndarray):
                        if auto_n_items is None:
                            auto_n_items = len(arg)
                        buffers.append((i, f"D{i}", self.push(arg)))
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
                storage = graph_storage_hints.get(arg_name, IOTypes.buffer)
                if storage == IOTypes.buffer:
                    # Check if this buffer was already processed
                    if any(idx == i for idx, _, _ in buffers):
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
                        
                        buffers.append((i, f"D{i}", self.push(scalar_array)))
            
            buffers.sort(key=lambda x: x[0])
            
            # Handle result buffer
            if res_type != 'void':
                res_dtype = GLSL_TO_NP.get(res_type, np.float32)
                
                if out is not None:
                    if isinstance(out, CastBuffer):
                        result_buffer = out.buffer
                    elif isinstance(out, np.ndarray):
                        if res_type.startswith('vec'):
                            dim = int(res_type[3])
                            expected_len = actual_n_items * dim
                            if len(out) != expected_len:
                                raise ValueError(f"Output array length {len(out)} doesn't match expected length {expected_len}")
                        else:
                            if len(out) != actual_n_items:
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
            
            # Set up buffer inputs
            buffer_count = 0
            for i, buffer_name, buf in buffers:
                node.set_shader_input(buffer_name, buf.buffer)
                buffer_count += 1
            
            # Set up uniforms
            for uniform_name, uniform_value in uniforms.items():
                node.set_shader_input(uniform_name, uniform_value)
            
            # Set any additional uniforms passed as keyword arguments
            for uniform_name, uniform_value in uniform_values.items():
                node.set_shader_input(uniform_name, uniform_value)
            
            # Always set nItems uniform for compute shaders
            node.set_shader_input("nItems", int(actual_n_items))
            
            # Set result buffer if needed
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
        
        # Handle complex numbers as vec2
        if data.dtype == np.complex64 or data.dtype == np.complex128:
            # Convert complex to interleaved float32 pairs
            data_flat = np.empty(data.size * 2, dtype=np.float32)
            data_flat[0::2] = data.real.ravel()
            data_flat[1::2] = data.imag.ravel()
            sbuf = ShaderBuffer("Data", data_flat.tobytes(), GeomEnums.UH_stream)
            return CastBuffer(sbuf, len(data), cast=np.complex64)
        else:
            sbuf = ShaderBuffer("Data", data.tobytes(), GeomEnums.UH_stream)
            return CastBuffer(sbuf, len(data), cast=data.dtype.type)

    def fetch(self, handle):
        gsg = self.base.win.get_gsg()
        raw = self.base.graphics_engine.extract_shader_buffer_data(handle.buffer, gsg)
        
        if handle.cast == np.bool_:
            return np.frombuffer(raw, dtype=np.int32).astype(np.bool_)
        elif handle.cast == np.complex64:
            # Convert interleaved float32 pairs back to complex64
            data = np.frombuffer(raw, dtype=np.float32)
            result = np.empty(len(handle), dtype=np.complex64)
            result.real = data[0::2]
            result.imag = data[1::2]
            return result
        elif handle.cast == np.complex128:
            data = np.frombuffer(raw, dtype=np.float64)
            result = np.empty(len(handle), dtype=np.complex128)
            result.real = data[0::2]
            result.imag = data[1::2]
            return result
        else:
            return np.frombuffer(raw, dtype=handle.cast)

    def _setup_headless(self):
        pipe = GraphicsPipeSelection.get_global_ptr().make_default_pipe()
        fb_prop = FrameBufferProperties()
        win_prop = WindowProperties.size(1, 1)
        self.base.win = self.base.graphics_engine.make_output(
            pipe, "math_headless", 0, fb_prop, win_prop, GraphicsPipe.BF_refuse_window
        )
    
    def shader(self, shader_type: str, uniforms : Dict = None):
        return shader_decorator(shader_type, uniforms = uniforms)
    
    def compile_shader(self, func, debug: bool = False):
        is_method = inspect.ismethod(func)
        actual_func = func.__func__ if is_method else func
        
        if not hasattr(actual_func, '_needs_shader_transpilation') and not hasattr(actual_func, '_shader_code'):
            raise ValueError("Function must be decorated with @shader")
        
        if getattr(actual_func, '_needs_shader_transpilation', False):
            shader_code = _transpile_shader(func)
        else:
            shader_code = getattr(actual_func, '_shader_code', None)
        
        shader_info = getattr(actual_func, '_shader_info', {})
        
        # Add function declarations if needed
        if shader_code and hasattr(shader_info, 'functions'):
            called_functions = shader_info.get('functions', [])
            if called_functions:
                # Get all dependencies
                all_deps = set()
                for func_name in called_functions:
                    all_deps.update(ShaderFunctionTranspiler.get_all_dependencies(func_name))
                all_deps.update(called_functions)
                
                # Transpile each function
                shader_type = shader_info.get('type', ShaderType.FRAGMENT)
                function_decls = []
                for func_name in all_deps:
                    func_metadata = FunctionRegistry.get(func_name)
                    if func_metadata:
                        glsl_code = ShaderFunctionTranspiler.transpile_function(func_metadata, shader_type)
                        function_decls.append(glsl_code)
                
                # Insert function declarations after version directive
                lines = shader_code.split('\n')
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if line.startswith('#version'):
                        # Add function declarations after version
                        for func_decl in function_decls:
                            new_lines.append(func_decl)
                
                shader_code = '\n'.join(new_lines)
        
        if debug and shader_code:
            print(f"Generated {shader_info.get('type', 'unknown')} shader code:")
            print(shader_code)
            print("-" * 60)
        
        return shader_code, shader_info

    def function(self, param_types: Dict[str, str] = None, return_type: str = None):
        return function_decorator(param_types, return_type)