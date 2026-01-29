import ast
import inspect
from typing import Dict, List, Tuple, Optional, Set, Any
from .function_registry import FunctionRegistry
from .gl_typing import ShaderType
from .helpers import CompilationError, create_error_context

class ShaderFunctionTranspiler:
    """Transpiler for shader functions that reuses the main transpiler infrastructure"""
    
    @staticmethod
    def transpile_function(func_metadata: Dict, shader_type: ShaderType) -> str:
        """Transpile a function to GLSL using the main transpiler infrastructure"""
        from .graph_transpiler import GraphTranspiler
        from .unified_compiler import UnifiedCompiler
        
        name = func_metadata['name']
        param_names = func_metadata['param_names']
        param_types = func_metadata.get('param_types', {})
        return_type = func_metadata.get('return_type', 'float')
        ast_node = func_metadata['ast_node']
        source_code = func_metadata.get('source_code', '')
        
        # Build parameter signature
        param_signature = []
        for param in param_names:
            param_type = param_types.get(param, 'float')
            param_signature.append(f"{param_type} {param}")
        
        try:
            # Get the file path from the original function
            original_func = func_metadata.get('original_func')
            file_path = inspect.getfile(original_func) if original_func else "<unknown>"
            
            # Create a GraphTranspiler for the function
            # For functions, we treat all parameters as 'uniform' storage since they're just inputs
            storage_hints = {param: 'uniform' for param in param_names}
            
            # Create a special GraphTranspiler for functions
            transpiler = FunctionGraphTranspiler(
                arg_names=param_names,
                shader_type=ShaderType.COMPUTE,
                type_hints=param_types,
                storage_hints=storage_hints,
                source_code=source_code,
                function_name=name,
                file_path=file_path,
                is_shader_function=True
            )
            
            # Process each statement in the function body
            for stmt in ast_node.body:
                transpiler.visit(stmt)
            
            graph = transpiler.graph
            
            # If there's a return statement but no output variable, set it
            if graph.output_var is None and return_type != 'void':
                # Check if the last statement was a return
                last_stmt = ast_node.body[-1] if ast_node.body else None
                if isinstance(last_stmt, ast.Return) and last_stmt.value:
                    # The return value should be in the last operation
                    if graph.operations:
                        last_op = graph.operations[-1]
                        if last_op[0] != 'return' and last_op[2]:  # op_name, inputs, output_var
                            graph.output_var = last_op[2]
            
            # Compile the function graph to GLSL
            compiler = UnifiedCompiler(
                graph=graph,
                shader_type=shader_type,
                input_types=param_types,
                explicit_types=transpiler.explicit_types,
                used_builtins=set(),
                custom_uniforms={},
                custom_textures={},
                is_function=True  # New flag to indicate this is a function compilation
            )
            
            # Generate the function signature and body
            signature = f"{return_type} {name}({', '.join(param_signature)})"
            
            # Get the compiled body from the compiler
            glsl_body = compiler.compile_function_body()
            
            # Build the complete function
            glsl_code = f"{signature} {{\n{glsl_body}\n}}"
            
            return glsl_code
            
        except Exception as e:
            error_info = create_error_context(
                ast_node, source_code, name, func_metadata.get('file_path', '<unknown>')
            )
            error_info.error_message = f"Failed to transpile function {name}: {e}"
            raise CompilationError(f"Failed to transpile function {name}: {e}", error_info)
    
    @staticmethod
    def get_all_dependencies(function_name: str) -> Set[str]:
        """Get all functions that a function depends on"""
        visited = set()
        to_visit = [function_name]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            
            visited.add(current)
            func_metadata = FunctionRegistry.get(current)
            if func_metadata and 'ast_node' in func_metadata:
                calls = ShaderFunctionTranspiler._find_function_calls(func_metadata['ast_node'])
                for called_func in calls:
                    if called_func in FunctionRegistry.get_all() and called_func not in visited:
                        to_visit.append(called_func)
        
        return visited
    
    @staticmethod
    def _find_function_calls(node: ast.AST) -> List[str]:
        """Find all function calls in an AST node"""
        calls = []
        
        class FunctionCallFinder(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    # Check if it's a type constructor (vec3, etc.)
                    if node.func.attr in ['vec2', 'vec3', 'vec4', 'ivec2', 'ivec3', 'ivec4',
                                         'uvec2', 'uvec3', 'uvec4', 'mat3', 'mat4']:
                        # These are type constructors, not function calls
                        pass
                    else:
                        # Could be a method call, but we don't support those in functions
                        pass
                self.generic_visit(node)
        
        finder = FunctionCallFinder()
        finder.visit(node)
        return calls

class FunctionGraphTranspiler:
    """Specialized GraphTranspiler for functions that handles function-specific logic"""
    
    def __init__(self, arg_names: List[str], shader_type: ShaderType = ShaderType.COMPUTE,
                type_hints: Dict[str, str] = None, storage_hints: Dict[str, str] = None,
                source_code: str = "", function_name: str = "", file_path: str = "",
                is_shader_function: bool = True):
        
        # Import here to avoid circular imports
        from .graph_transpiler import GraphTranspiler
        
        # Create a regular GraphTranspiler instance
        self.transpiler = GraphTranspiler(
            arg_names=arg_names,
            shader_type=shader_type,
            type_hints=type_hints,
            storage_hints=storage_hints,
            source_code=source_code,
            function_name=function_name,
            file_path=file_path,
            is_shader_function=is_shader_function
        )
        
        # Expose important attributes
        self.graph = self.transpiler.graph
        self.explicit_types = self.transpiler.explicit_types
        self.var_types = self.transpiler.var_types
        self.var_map = self.transpiler.var_map
    
    def visit(self, node):
        """Delegate to the underlying transpiler"""
        return self.transpiler.visit(node)
    
    def visit_Return(self, node):
        """Handle return statements specially for functions"""
        if node.value is None:
            self.transpiler.graph.set_void_return()
            # Add a return operation
            self.transpiler.graph.add_operation('return', [], 
                                              in_loop=bool(self.transpiler.graph.current_scope))
        else:
            # Visit the return value
            result = self.transpiler.visit(node.value)
            self.transpiler.graph.set_output(result)
            
            # Also add a return operation with the result
            self.transpiler.graph.add_operation('return', [result],
                                              in_loop=bool(self.transpiler.graph.current_scope))