import ast
import inspect
from typing import Dict, List, Tuple, Optional
from .graph_transpiler import GraphTranspiler
from .unified_compiler import UnifiedCompiler
from .helpers import CompilationError, CompilationErrorInfo
from .gl_typing import NP_GLTypes, ShaderType

def _extract_hints(hints: Dict) -> Tuple[Dict, Dict]:
    """Extract type and storage hints"""
    type_hints = {}
    storage_hints = {}
    if hints:
        for key, value in hints.items():
            if isinstance(value, tuple):
                # Handle (type, storage) tuples like (NP_GLTypes.float, IOTypes.array)
                type_info = value[0]
                storage = value[1] if len(value) > 1 else "buffer"
                
                # Check if type_info is from NP_GLTypes (tuple of (numpy_type, glsl_type))
                if isinstance(type_info, tuple) and len(type_info) >= 2:
                    # NP_GLTypes format: (numpy_type, glsl_type_string)
                    glsl_type = type_info[1]
                elif isinstance(type_info, str):
                    # Already a GLSL type string
                    glsl_type = type_info
                else:
                    glsl_type = 'float'
                
                type_hints[key] = glsl_type
                storage_hints[key] = storage
            elif isinstance(value, str):
                # Just a type string
                type_hints[key] = value
                storage_hints[key] = "buffer"
            else:
                # Could be a type constant from NP_GLTypes or similar
                type_hints[key] = value
                storage_hints[key] = "buffer"

    return type_hints, storage_hints

def _transpile_kernel(func, hints: Dict = None, layout: Tuple[int, int, int] = (64, 1, 1)):
    """Transpile a kernel function to compute shader"""
    is_method = inspect.ismethod(func)
    actual_func = func.__func__ if is_method else func

    if hints is None:
        hints = getattr(actual_func, '_gpu_kernel_hints', {})

    if not getattr(actual_func, '_needs_transpilation', True):
        return
    
    try:
        file_path = inspect.getfile(actual_func)
    except:
        file_path = "<unknown>"
    
    source = inspect.getsource(actual_func)
    
    lines = source.split('\n')
    func_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def '):
            func_start = i
            break
    
    func_lines = lines[func_start:]
    if func_lines:
        min_indent = min(len(line) - len(line.lstrip()) 
                        for line in func_lines if line.strip())
        func_lines = [line[min_indent:] if line.strip() else line 
                     for line in func_lines]
    
    source = '\n'.join(func_lines)
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        error_info = CompilationErrorInfo(
            file_path=file_path,
            line_number=e.lineno,
            column_offset=e.offset,
            function_name=actual_func.__name__,
            source_code=e.text,
            error_message=str(e.msg)
        )
        raise CompilationError(f"Syntax error: {e.msg}", error_info) from e
    
    func_def = tree.body[0]
    arg_names = [arg.arg for arg in func_def.args.args]
    
    if is_method and arg_names and arg_names[0] == 'self':
        arg_names = arg_names[1:]
    
    static_constants = getattr(actual_func, '_static_constants', None)
    vectorized = getattr(actual_func, '_gpu_kernel_vectorized', None)
    
    type_hints, storage_hints = _extract_hints(hints)
    
    transpiler = GraphTranspiler(
        arg_names, ShaderType.COMPUTE, type_hints, storage_hints, source, actual_func.__name__, 
        file_path, static_constants, layout, vectorized
    )
    
    try:
        for stmt in func_def.body:
            transpiler.visit(stmt)
    except CompilationError:
        raise
    except Exception as e:
        error_info = CompilationErrorInfo(
            file_path=file_path,
            function_name=actual_func.__name__,
            error_message=str(e)
        )
        raise CompilationError(f"Transpilation error: {e}", error_info) from e
    
    graph = transpiler.graph    
    actual_func._original_hints = hints
    actual_func._compute_graph = graph
    actual_func._arg_names = arg_names
    actual_func._type_hints = type_hints
    actual_func._storage_hints = storage_hints
    actual_func._explicit_types = transpiler.explicit_types
    actual_func._local_functions = transpiler.local_functions
    actual_func._needs_transpilation = False

def gpu_kernel(hints: Optional[Dict[str, tuple]] = None, layout: Tuple[int, int, int] = (64, 1, 1), vectorized: Optional[bool] = None):
    """Decorator for GPU compute kernels"""
    def decorator(func):
        func._gpu_kernel_hints = hints
        func._gpu_kernel_layout = layout
        func._gpu_kernel_vectorized = vectorized
        func._needs_transpilation = True
        return func
    
    if callable(hints):
        func = hints
        hints = None
        return decorator(func)
    
    return decorator

def shader(shader_type: str, uniforms: Dict = None, textures: Dict = None):
    """Decorator for fragment, vertex, or geometry shaders"""
    try:
        shader_type_enum = ShaderType(shader_type.lower())
    except ValueError:
        raise ValueError(f"Invalid shader type: {shader_type}. Must be one of: {[t.value for t in ShaderType]}")
    
    uniforms = uniforms or {}
    textures = textures or {}
    
    def decorator(func):
        func._shader_type = shader_type_enum
        func._custom_uniforms = uniforms
        func._custom_textures = textures
        func._needs_shader_transpilation = True
        return func
    
    return decorator

def _transpile_shader(func):
    """Transpile a shader function to GLSL"""
    is_method = inspect.ismethod(func)
    actual_func = func.__func__ if is_method else func
    
    if not getattr(actual_func, '_needs_shader_transpilation', False):
        return getattr(actual_func, '_shader_code', None)
    
    shader_type = getattr(actual_func, '_shader_type', ShaderType.FRAGMENT)
    custom_uniforms = getattr(actual_func, '_custom_uniforms', {})
    custom_textures = getattr(actual_func, '_custom_textures', {})
    
    try:
        file_path = inspect.getfile(actual_func)
    except:
        file_path = "<unknown>"
    
    source = inspect.getsource(actual_func)
    
    lines = source.split('\n')
    func_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def '):
            func_start = i
            break
    
    func_lines = lines[func_start:]
    if func_lines:
        min_indent = min(len(line) - len(line.lstrip()) 
                        for line in func_lines if line.strip())
        func_lines = [line[min_indent:] if line.strip() else line 
                     for line in func_lines]
    
    source = '\n'.join(func_lines)
    
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        error_info = CompilationErrorInfo(
            file_path=file_path,
            line_number=e.lineno,
            column_offset=e.offset,
            function_name=actual_func.__name__,
            source_code=e.text,
            error_message=str(e.msg)
        )
        raise CompilationError(f"Syntax error: {e.msg}", error_info) from e
    
    func_def = tree.body[0]
    arg_names = [arg.arg for arg in func_def.args.args]
    
    if arg_names and arg_names[0] == 'self':
        arg_names = arg_names[1:]
    
    transpiler = GraphTranspiler(
        arg_names, shader_type, {}, {}, source, actual_func.__name__, 
        file_path, None, (1, 1, 1), None, False, custom_uniforms, custom_textures
    )
    
    try:
        for stmt in func_def.body:
            transpiler.visit(stmt)
    except Exception as e:
        error_info = CompilationErrorInfo(
            file_path=file_path,
            function_name=actual_func.__name__,
            error_message=str(e)
        )
        raise CompilationError(f"Shader transpilation error: {e}", error_info) from e
    
    compiler = UnifiedCompiler(
        transpiler.graph, shader_type, {}, transpiler.explicit_types,
        transpiler.used_builtins, custom_uniforms, custom_textures
    )
    
    shader_code = compiler.compile()
    
    actual_func._shader_code = shader_code
    actual_func._shader_info = {'code': shader_code, 'type': shader_type}
    actual_func._needs_shader_transpilation = False
    
    return shader_code

def function(param_types: Dict[str, str] = None, return_type: str = None):
    """Decorator for reusable shader functions"""
    def decorator(func):
        from .function_registry import FunctionRegistry
        FunctionRegistry.register(func.__name__, func, param_types, return_type)
        return func
    
    return decorator

def inline_always(func):
    """Mark a function for inlining"""
    func._inline_always = True
    return func

def static_constant(name: str, glsl_type: str, size: int, values: list):
    """Decorator to add static constants to a kernel"""
    def decorator(func):
        if not hasattr(func, '_static_constants'):
            func._static_constants = []
        func._static_constants.append((name, glsl_type, size, values))
        return func
    return decorator