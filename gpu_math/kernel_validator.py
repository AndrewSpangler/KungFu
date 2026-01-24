from typing import Dict, List, Set
from .gl_typing import IOTypes, is_kungfu_builtin

class KernelValidator:
    """Validate kernel function signatures and usage"""
    
    @staticmethod
    def validate_kernel_signature(arg_names: List[str], hints: Dict[str, tuple], 
                                 function_name: str) -> Dict[str, str]:
        """
        Validate kernel signature and return kernel type info
        
        Returns:
            Dict with keys: 'type' ('vectorized', 'array', or 'mixed'),
                           'has_buffer_inputs' (bool),
                           'has_array_inputs' (bool),
                           'uses_gid' (bool)
        """
        has_buffer_inputs = False
        has_array_inputs = False
        uses_gid = False
        
        for arg_name in arg_names:
            hint = hints.get(arg_name, (None, IOTypes.buffer))
            storage = hint[1] if len(hint) > 1 else IOTypes.buffer
            
            if storage == IOTypes.buffer:
                has_buffer_inputs = True
            elif storage == IOTypes.array:
                has_array_inputs = True
            
            # Check for nItems parameter in vectorized kernels
            if arg_name == 'nItems' and has_buffer_inputs:
                raise ValueError(
                    f"Error in function '{function_name}': "
                    "nItems should not be passed as a parameter to vectorized kernels. "
                    "It's automatically available as 'n_items' or 'nItems' uniform."
                )
        
        # Determine kernel type
        if has_buffer_inputs and not has_array_inputs:
            kernel_type = 'vectorized'
        elif has_array_inputs and not has_buffer_inputs:
            kernel_type = 'array'
        elif has_buffer_inputs and has_array_inputs:
            kernel_type = 'mixed'
        else:
            kernel_type = 'uniform_only'  # Only uniform inputs
        
        return {
            'type': kernel_type,
            'has_buffer_inputs': has_buffer_inputs,
            'has_array_inputs': has_array_inputs,
            'uses_gid': uses_gid
        }
    
    @staticmethod
    def get_required_builtins(kernel_type: str, ast_tree, source_code: str) -> Set[str]:
        """
        Analyze AST to determine which built-in variables are needed
        """
        import ast
        
        required_builtins = set()
        
        # Check for usage of built-in variables in the AST
        class BuiltinFinder(ast.NodeVisitor):
            def __init__(self):
                self.builtins = set()
            
            def visit_Name(self, node):
                if is_kungfu_builtin(node.id):
                    self.builtins.add(node.id)
                self.generic_visit(node)
            
            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and is_kungfu_builtin(node.value.id):
                    self.builtins.add(node.value.id)
                self.generic_visit(node)
        
        finder = BuiltinFinder()
        finder.visit(ast_tree)
        
        # Add required builtins based on kernel type
        if kernel_type in ['vectorized', 'mixed']:
            # Vectorized kernels always need gid and n_items
            required_builtins.update(['gid', 'n_items'])
        
        # Add any found builtins
        required_builtins.update(finder.builtins)
        
        return required_builtins
    
    @staticmethod
    def validate_builtin_usage(kernel_info: Dict, required_builtins: Set[str]) -> List[str]:
        """
        Validate that built-in variables are used correctly
        """
        warnings = []
        
        if kernel_info['type'] == 'array' and 'gid' in required_builtins:
            warnings.append(
                "Warning: Using 'gid' in array-style kernel. "
                "gid is only automatically available in vectorized kernels. "
                "Make sure to pass thread indices as parameters."
            )
        
        if kernel_info['type'] == 'vectorized' and 'n_items' not in required_builtins:
            warnings.append(
                "Warning: Vectorized kernel doesn't use 'n_items' for bounds checking. "
                "Consider adding bounds check: if(gid < n_items) { ... }"
            )
        
        return warnings