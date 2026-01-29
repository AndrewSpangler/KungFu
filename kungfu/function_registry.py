import ast
import inspect
import textwrap
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from .helpers import CompilationError

class FunctionRegistry:
    _instance = None
    _functions: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FunctionRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, func: Callable, param_types: Dict[str, str] = None,
                 return_type: str = None) -> Dict:
        """Register a function for use in shaders (store metadata only)"""
        if name in cls._functions:
            return cls._functions[name]
        
        # Get source code
        source = inspect.getsource(func)
        
        # Remove decorator lines and dedent
        lines = source.split('\n')
        start_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                start_line = i
                break
        
        source = '\n'.join(lines[start_line:])
        source = textwrap.dedent(source)
        
        # Parse AST
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            raise CompilationError(f"Syntax error in function {name}: {e}")
        
        func_def = tree.body[0]
        
        # Extract parameter names
        param_names = [arg.arg for arg in func_def.args.args]
        
        # Extract parameter types from annotations
        inferred_param_types = param_types or {}
        for i, arg in enumerate(func_def.args.args):
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    type_name = arg.annotation.id
                    if type_name in [
                        'float',    'int',      'bool', 
                        'double',   'mat3',     'mat4',
                        'vec2',     'vec3',     'vec4',
                        'uvec2',    'uvec3',    'uvec4',
                        'ivec2',    'ivec3',    'ivec4',
                    ]:
                        inferred_param_types[arg.arg] = type_name
        
        # Extract return type
        inferred_return_type = return_type or 'float'
        if func_def.returns:
            if isinstance(func_def.returns, ast.Name):
                return_type_name = func_def.returns.id
                if return_type_name in [
                    'float',    'int',      'bool', 
                    'double',   'mat3',     'mat4',
                    'vec2',     'vec3',     'vec4',
                    'uvec2',    'uvec3',    'uvec4',
                    'ivec2',    'ivec3',    'ivec4',
                    'void'
                ]:
                    inferred_return_type = return_type_name
        
        # Store function metadata
        func_metadata = {
            'name': name,
            'source_code': source,
            'ast_node': func_def,
            'param_names': param_names,
            'param_types': inferred_param_types,
            'return_type': inferred_return_type,
            'original_func': func
        }
        
        cls._functions[name] = func_metadata
        return func_metadata
    
    @classmethod
    def get(cls, name: str) -> Optional[Dict]:
        """Get a registered function's metadata"""
        return cls._functions.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, Dict]:
        """Get all registered functions"""
        return cls._functions.copy()
    
    @classmethod
    def _clear(cls):
        """Clear the registry for testing (don't use in production)"""
        cls._functions.clear()