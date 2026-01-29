import ast
from typing import Dict, List, Set, Any, Optional, Callable
from .helpers import CompilationError, create_error_context
from .gl_typing import AST_BIN_OP_MAP

class ASTVisitorBase:
    """Base class with common AST visitor utilities"""
    
    def __init__(
        self,
        source_code: str = "",
        function_name: str = "", 
        file_path: str = ""
    ):
        self.source_code = source_code
        self.function_name = function_name
        self.file_path = file_path
        self.var_map = {}
        self.var_types = {}
        self.local_functions = {}
        self.inline_always = set()
        self.explicit_types = {}
        self.temp_var_usage_count = {}
        self.builtin_variables = {}

    def _create_error(self, message: str, node: ast.AST) -> CompilationError:
        """Create a CompilationError with context"""
        error_info = create_error_context(
            node, self.source_code, self.function_name, self.file_path
        )
        error_info.error_message = message
        return CompilationError(message, error_info)
    
    def _get_var_type(self, var_name: str) -> str:
        """Get type of variable"""
        if self._is_literal(var_name):
            return self._get_literal_type(var_name)
        elif var_name in self.explicit_types:
            return self.explicit_types[var_name]
        elif var_name in self.var_types:
            return self.var_types[var_name]
        elif var_name in self.builtin_variables:
            return self.builtin_variables[var_name]
        else:
            return 'float'
    
    def _is_literal(self, val: str) -> bool:
        """Check if value is a literal"""
        try:
            float(val)
            return True
        except ValueError:
            return val.lower() in ['true', 'false', 'null', 'none']
    
    def _get_literal_type(self, val: str) -> str:
        """Get type of literal value"""
        if val.lower() in ['true', 'false']:
            return 'bool'
        elif '.' in val or 'e' in val.lower():
            return 'float'
        else:
            try:
                int(val)
                return 'int'
            except ValueError:
                return 'float'
    
    def _increment_temp_usage(self, var_name: str):
        """Track usage of temporary variables"""
        if var_name.startswith('_t'):
            self.temp_var_usage_count[var_name] = self.temp_var_usage_count.get(var_name, 0) + 1
    
    def _record_variable_use(self, var_name: str):
        """Record when a variable is used as input"""
        # Don't record literals or builtins
        if self._is_literal(var_name) or var_name in self.builtin_variables:
            return
            
        if var_name in self.temp_var_usage_count:
            self._increment_temp_usage(var_name)

        # Check if it's a mapped variable
        elif var_name in self.var_map:
            mapped_name = self.var_map[var_name]
            if mapped_name.startswith('_t'):
                self._increment_temp_usage(mapped_name)
    
    def _visit_constant(self, node: ast.Constant) -> str:
        """Handle constant values"""
        try:
            if node.value is True:
                return "true"
            elif node.value is False:
                return "false"
            elif node.value is None:
                return "0"
            elif isinstance(node.value, (int, float)):
                return str(node.value)
            else:
                return f'"{node.value}"'
        except Exception as e:
            raise self._create_error(f"Invalid constant: {e}", node)
    
    def _visit_name(self, node: ast.Name, builtin_check: Callable[[str], Optional[str]] = None) -> str:
        """Handle variable names"""
        try:
            if node.id == 'self':
                return None
            
            # Check for built-in variables
            if builtin_check:
                builtin_expr = builtin_check(node.id)
                if builtin_expr is not None:
                    return builtin_expr
            
            if node.id in self.builtin_variables:
                return node.id
            
            if node.id in self.var_map:
                var_name = self.var_map[node.id]
                self._record_variable_use(var_name)
                return var_name
            
            raise self._create_error(
                f"Variable '{node.id}' used before assignment", node
            )
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error accessing variable '{node.id}': {e}", node)
    
    def _visit_binop(self, node: ast.BinOp, add_operation: Callable) -> str:
        """Handle binary operations"""
        try:
            left = self.visit(node.left)
            if left not in self.builtin_variables:
                self._record_variable_use(left)
            
            right = self.visit(node.right)

            op_type = type(node.op)
            if op_type in AST_BIN_OP_MAP:
                result = add_operation(AST_BIN_OP_MAP[op_type], [left, right])
                self._increment_temp_usage(result)
                return result
            else:
                raise self._create_error(
                    f"Unsupported binary operation: {op_type.__name__}", node.op
                )
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in binary operation: {e}", node)
    
    def _visit_compare(self, node: ast.Compare, add_operation: Callable) -> str:
        """Handle comparison operations"""
        try:
            left = self.visit(node.left)
            if left not in self.builtin_variables:
                self._record_variable_use(left)
            
            comparisons = []
            
            for op, right in zip(node.ops, node.comparators):
                right_var = self.visit(right)
                if right_var not in self.builtin_variables:
                    self._record_variable_use(right_var)
                
                op_map = {
                    ast.Lt: 'lt', ast.LtE: 'lte', ast.Gt: 'gt',
                    ast.GtE: 'gte', ast.Eq: 'eq', ast.NotEq: 'neq',
                }
                
                op_type = type(op)
                if op_type in op_map:
                    result = add_operation(op_map[op_type], [left, right_var])
                    self._increment_temp_usage(result)
                    comparisons.append(result)
                else:
                    raise self._create_error(
                        f"Unsupported comparison: {op_type.__name__}", op
                    )
                left = right_var
            
            if len(comparisons) == 1:
                return comparisons[0]
            
            # Chain comparisons
            result = comparisons[0]
            for comp in comparisons[1:]:
                result = add_operation('and', [result, comp])
                self._increment_temp_usage(result)
            return result
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in comparison: {e}", node)
    
    def _visit_call(
        self,
        node: ast.Call,
        add_operation: Callable, 
        glsl_functions: Set[str],
        type_functions: Set[str],
        function_handler: Callable = None
    ) -> str:
        """Handle function calls"""
        try:
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            else:
                raise self._create_error(
                    f"Unsupported function call type: {type(node.func).__name__}", node.func
                )
            
            args = [self.visit(arg) for arg in node.args]
            for arg in args:
                if arg not in self.builtin_variables:
                    self._record_variable_use(arg)
            
            # Handle type constructors and casts
            if func_name in type_functions:
                if add_operation is not None:
                    result = add_operation(func_name, args)
                    self._increment_temp_usage(result)
                    return result
                else:
                    # BaseTranspiler mode - just return the GLSL expression
                    return f"{func_name}({', '.join(args)})"
            
            # Handle GLSL built-in functions
            if func_name in glsl_functions:
                if add_operation is not None:
                    result = add_operation(func_name, args)
                    self._increment_temp_usage(result)
                    return result
                else:
                    # BaseTranspiler mode - just return the GLSL expression
                    return f"{func_name}({', '.join(args)})"
            
            # Handle custom function calls
            if function_handler:
                result = function_handler(func_name, args, add_operation)
                if result:
                    return result
            
            raise self._create_error(
                f"Function '{func_name}' not recognized", node
            )
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in function call '{func_name}': {e}", node)