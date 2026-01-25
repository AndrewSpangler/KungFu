import ast
import sys
import traceback
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from panda3d.core import Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f
from .composition import get_standard_heading, _buff_line
from .cast_buffer import CastBuffer
from .gl_typing import (
    GLTypes, NP_GLTypes, Vec_GLTypes, VEC_TO_GLSL,
    BUILTIN_VARIABLES, GLSL_TYPE_MAP, is_kungfu_builtin,
    get_kungfu_builtin_glsl
)
from .compute_graph import ComputeGraph
from .error_handler import (
    CompilationError, CompilationErrorInfo, 
    create_error_context, get_node_location
)

class PythonToGLSLTranspiler(ast.NodeVisitor):
    def __init__(self, arg_names: List[str], hints: Dict[str, tuple] = None,
                 source_code: str = "", function_name: str = "", file_path: str = "",
                 static_constants: List[Tuple[str, str, int, list]] = None,
                 layout: Tuple[int, int, int] = (64, 1, 1), vectorized: Optional[bool] = None):
        self.graph = ComputeGraph()
        self.arg_names = arg_names
        self.var_map = {}
        self.var_types = {}
        self.local_functions = {}
        self.inline_always = set()
        self.explicit_types = {}
        self.hints = hints or {}
        self.source_code = source_code
        self.function_name = function_name
        self.file_path = file_path
        self.temp_var_usage_count = {}
        self.static_constants = static_constants or []
        self.layout = layout
        self.graph.layout = layout  # Store layout in graph
        
        # Track whether this kernel is vectorized (uses IOTypes.buffer)
        self.has_vectorized_inputs = False
        self.has_array_inputs = False
        
        # Check inputs to determine kernel type (only if not explicitly set)
        if vectorized is None:
            for arg in arg_names:
                hint = self.hints.get(arg, (NP_GLTypes.float, "buffer"))
                storage = hint[1] if len(hint) > 1 else "buffer"
                if storage == "buffer":
                    self.has_vectorized_inputs = True
                elif storage == "array":
                    self.has_array_inputs = True
        else:
            # Explicitly set by decorator
            self.has_vectorized_inputs = vectorized
        
        # Store in graph for later use
        self.graph.is_vectorized = self.has_vectorized_inputs
        
        # Add built-in variables to explicit types
        for builtin_name, builtin_type in BUILTIN_VARIABLES.items():
            self.explicit_types[builtin_name] = builtin_type
            self.var_map[builtin_name] = builtin_name  # Map to itself
        
        # Add static constants to graph and make them available immediately
        if static_constants:
            for name, glsl_type, size, values in static_constants:
                self.graph.add_static_constant(name, glsl_type, size, values)
                self.var_map[name] = name
                self.explicit_types[name] = f"{glsl_type}[{size}]"
        
        # Add function arguments to var_map
        for arg in arg_names:
            hint = self.hints.get(arg, (NP_GLTypes.float, "buffer"))
            storage = hint[1] if len(hint) > 1 else "buffer"
            self.graph.add_input(arg, storage)
            self.var_map[arg] = arg
            
            # Check for nItems parameter in vectorized kernels
            if arg == 'nItems' and self.has_vectorized_inputs:
                raise CompilationError(
                    "nItems should not be passed as a parameter to vectorized kernels. "
                    "It's automatically available as 'n_items' or 'nItems' uniform."
                )
    
        self.explicit_types['vec2'] = GLTypes.vec2
        self.explicit_types['vec3'] = GLTypes.vec3
        self.explicit_types['vec4'] = GLTypes.vec4

    def _create_error(self, message: str, node: ast.AST) -> CompilationError:
        """Create a CompilationError with context"""
        error_info = create_error_context(
            node, self.source_code, self.function_name, self.file_path
        )
        error_info.error_message = message
        return CompilationError(message, error_info)
    
    def _get_var_type(self, var_name: str) -> str:
        if self._is_literal(var_name):
            if var_name.lower() in ['true', 'false']:
                return 'bool'
            elif '.' in var_name or 'e' in var_name.lower():
                return 'float'
            else:
                try:
                    val = int(var_name)
                    return 'int' if val < 0 else 'uint'
                except ValueError:
                    return 'float'
        elif var_name in self.explicit_types:
            return self.explicit_types[var_name]
        elif var_name in BUILTIN_VARIABLES:
            return BUILTIN_VARIABLES[var_name]
        elif var_name in self.var_types:  # ADD THIS CHECK
            return self.var_types[var_name]
        else:
            return 'float'

    def _is_literal(self, val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return val.lower() in ['true', 'false']
    
    def _increment_temp_usage(self, var_name: str):
        """Track usage of temporary variables for elimination"""
        if var_name.startswith('_t'):
            self.temp_var_usage_count[var_name] = self.temp_var_usage_count.get(var_name, 0) + 1
    
    def _record_variable_use(self, var_name: str):
        """Record when a variable is used as input"""
        if var_name in self.temp_var_usage_count:
            self._increment_temp_usage(var_name)

    def _handle_chained_subscript(self, node, is_store=False):
        """Handle multi-dimensional array subscript like array[i][j]"""
        # Collect all indices in reverse order
        indices = []
        current = node
        while isinstance(current, ast.Subscript):
            indices.insert(0, self.visit(current.slice))
            current = current.value
        
        array_var = self.visit(current)
        
        if is_store:
            # For assignment, we don't need an output variable yet
            return array_var, indices
        else:
            # For value access, create an operation
            if len(indices) == 1:
                result = self.graph.add_operation('subscript', [array_var, indices[0]], 
                                            in_loop=bool(self.graph.current_scope))
                # Track the type of the subscript result
                array_type = self._get_var_type(array_var)
                if array_type.startswith(('vec', 'uvec', 'ivec')):
                    self.var_types[result] = array_type
                    self.graph.var_types[result] = array_type
                self._increment_temp_usage(result)
                return result
            elif len(indices) == 2:
                result = self.graph.add_operation('subscript_2d', [array_var, indices[0], indices[1]], 
                                            in_loop=bool(self.graph.current_scope))
                self._increment_temp_usage(result)
                return result
            else:
                raise self._create_error(
                    f"Array subscript with {len(indices)} dimensions not supported", node
                )

    def visit_Constant(self, node):
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

    def visit_Name(self, node):
        try:
            # Skip 'self' references (for methods)
            if node.id == 'self':
                return None
            
            # Check if it's a KungFu built-in
            if is_kungfu_builtin(node.id):
                # For vectorized kernels with buffer inputs, gid should be automatically available
                if node.id == 'gid' and self.has_vectorized_inputs:
                    # In vectorized kernels, gid is gl_GlobalInvocationID.x
                    return 'gl_GlobalInvocationID.x'
                elif node.id == 'n_items' and self.has_vectorized_inputs:
                    # In vectorized kernels, n_items is available as uniform
                    return 'nItems'
                else:
                    # Return the GLSL expression for the built-in
                    return get_kungfu_builtin_glsl(node.id)
            
            # Check if it's a built-in variable
            if node.id in BUILTIN_VARIABLES:
                return node.id
            
            if node.id in self.var_map:
                # Record that this variable is being used
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
            
    def visit_Subscript(self, node):
        try:
            # Handle both single and multi-dimensional subscripts
            if isinstance(node.ctx, ast.Store):
                # We return a marker or handle it in visit_Assign
                # This logic is handled by visit_Assign calling _handle_chained_subscript(is_store=True)
                return node
            else:
                # This is a value access
                return self._handle_chained_subscript(node, is_store=False)
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in array subscript: {e}", node)
    
    def visit_Attribute(self, node):
        try:
            value = self.visit(node.value)
            
            # Handle KungFu built-in component access
            if value in ['gl_GlobalInvocationID', 'gl_WorkGroupID', 'gl_LocalInvocationID',
                        'gl_WorkGroupSize', 'gl_NumWorkGroups', 'gid_xyz', 'wgid', 'lid',
                        'wg_size', 'num_wg'] or value in BUILTIN_VARIABLES:
                # These are uvec3/ivec3 types
                attr = node.attr
                if attr in ['x', 'y', 'z', 'r', 'g', 'b']:
                    return f"{value}.{attr}"
                elif attr in ['xy', 'xyz', 'rgb']:
                    return f"{value}.{attr}"
            
            # Check if value is a built-in variable
            if value in BUILTIN_VARIABLES:
                # Built-in variables don't need usage tracking
                pass
            else:
                self._record_variable_use(value)
            
            attr = node.attr
            
            # Handle swizzling for vec types (.x, .y, .z, .w) and color equivalents
            if attr in ['x', 'y', 'z', 'w', 'xy', 'xyz', 'xyzw', 'rgb', 'rgba', 'r', 'g', 'b', 'a']:
                result = self.graph.add_operation('swizzle', [value, attr], 
                                               in_loop=bool(self.graph.current_scope))
                self._increment_temp_usage(result)
                return result
            
            raise self._create_error(f"Unsupported attribute: {attr}", node)
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in attribute access: {e}", node)

    def visit_Assign(self, node):
        try:
            if len(node.targets) != 1:
                raise self._create_error(
                    "Multiple assignment targets not supported", node
                )
            
            target = node.targets[0]
            
            # Check if target is a built-in variable (read-only)
            if isinstance(target, ast.Name) and target.id in BUILTIN_VARIABLES:
                raise self._create_error(
                    f"Cannot assign to built-in variable '{target.id}'", target
                )
            
            # Handle Array Subscripts (like result[i][j] = sum_val)
            if isinstance(target, ast.Subscript):
                array_var, indices = self._handle_chained_subscript(target, is_store=True)
                value_var = self.visit(node.value)
                
                # We use a dummy output name for assignments to ensure they are 
                # treated as statements rather than expressions in the compiler
                if len(indices) == 1:
                    self.graph.add_operation('subscript_assign', [array_var, indices[0], value_var], 
                                            output_var="_void_", # Use a marker
                                            in_loop=bool(self.graph.current_scope))
                elif len(indices) == 2:
                    self.graph.add_operation('subscript_assign_2d', [array_var, indices[0], indices[1], value_var], 
                                            output_var="_void_", # Use a marker
                                            in_loop=bool(self.graph.current_scope))
                return None # Return None to signify this assignment is complete

            # Case 2: Standard Variable Assignment (e.g., a = val)
            if not isinstance(target, ast.Name):
                raise self._create_error(
                    "Only simple variable assignment supported", target
                )
            
            target_name = target.id
            result_var = self.visit(node.value)
            self.var_map[target_name] = target_name
            
            # Try to infer the type from the result
            result_type = self._get_var_type(result_var)
            self.var_types[target_name] = result_type  # ADD THIS LINE
            
            # Check if we can eliminate the temporary variable
            if result_var.startswith('_t') and result_var in self.temp_var_usage_count:
                # Check if this temp is only used here
                if self.temp_var_usage_count.get(result_var, 0) <= 1:
                    # Replace the temp with target directly
                    self.var_map[result_var] = target_name
                    # Store type mapping
                    self.var_types[result_var] = result_type  # ADD THIS LINE
                    # Store special operation for compiler to handle
                    self.graph.add_operation('direct_assign', [result_var, target_name], 
                                            in_loop=bool(self.graph.current_scope))
                    return target_name
            
            self.graph.add_operation('assign', [target_name, result_var], 
                                    in_loop=bool(self.graph.current_scope))
            return target_name
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in assignment: {e}", node)
            
    def visit_AugAssign(self, node):
        try:
            if not isinstance(node.target, ast.Name):
                raise self._create_error(
                    "Only simple variable assignment supported", node.target
                )
            
            target = node.target.id
            
            # Check if target is a built-in variable (read-only)
            if target in BUILTIN_VARIABLES:
                raise self._create_error(
                    f"Cannot assign to built-in variable '{target}'", node.target
                )
            
            left = self.visit(node.target)
            self._record_variable_use(left)
            right = self.visit(node.value)
            
            op_map = {
                ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mult',
                ast.Div: 'div', ast.Pow: 'pow', ast.Mod: 'mod',
                ast.BitAnd: 'and', ast.BitOr: 'or', ast.BitXor: 'xor',
                ast.LShift: 'lsh', ast.RShift: 'rsh'
            }
            
            op_name = op_map.get(type(node.op))
            if op_name is None:
                raise self._create_error(
                    f"Unsupported augmented assignment: {type(node.op).__name__}", node.op
                )
            
            temp_result = self.graph.add_operation(op_name, [left, right], 
                                                  in_loop=bool(self.graph.current_scope))
            self._increment_temp_usage(temp_result)
            self.graph.add_operation('assign', [target, temp_result], 
                                    in_loop=bool(self.graph.current_scope))
            self.var_map[target] = target
            return target
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in augmented assignment: {e}", node)

    def visit_BinOp(self, node):
        try:
            left = self.visit(node.left)
            # Only track usage if not a built-in
            if left not in BUILTIN_VARIABLES:
                self._record_variable_use(left)
            
            right = self.visit(node.right)
            
            if isinstance(node.op, ast.Pow):
                result = self.graph.add_operation('pow', [left, right], 
                                               in_loop=bool(self.graph.current_scope))
                self._increment_temp_usage(result)
                return result
            
            op_map = {
                ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mult', ast.Div: 'div',
                ast.FloorDiv: 'floordiv', ast.Mod: 'mod', ast.BitAnd: 'and', ast.BitOr: 'or',
                ast.BitXor: 'xor', ast.LShift: 'lsh', ast.RShift: 'rsh',
            }
            
            op_type = type(node.op)
            if op_type not in op_map:
                raise self._create_error(
                    f"Unsupported binary operation: {op_type.__name__}", node.op
                )
            
            result = self.graph.add_operation(op_map[op_type], [left, right], 
                                           in_loop=bool(self.graph.current_scope))
            self._increment_temp_usage(result)
            return result
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in binary operation: {e}", node)
    
    def visit_UnaryOp(self, node):
        """Handle unary operations like negation (-x), positive (+x), and bitwise not (~x)"""
        try:
            operand = self.visit(node.operand)
            # Only track usage if not a built-in
            if operand not in BUILTIN_VARIABLES:
                self._record_variable_use(operand)
            
            op_map = {
                ast.USub: 'neg',      # Unary minus: -x
                ast.UAdd: 'full',     # Unary plus: +x (identity)
                ast.Not: 'bool_not',  # Logical not: not x
                ast.Invert: 'bitwise_not'  # Bitwise not: ~x
            }
            
            op_type = type(node.op)
            if op_type not in op_map:
                raise self._create_error(
                    f"Unsupported unary operation: {op_type.__name__}", node.op
                )
            
            result = self.graph.add_operation(op_map[op_type], [operand], 
                                           in_loop=bool(self.graph.current_scope))
            self._increment_temp_usage(result)
            return result
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in unary operation: {e}", node)
    
    def visit_Call(self, node):
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
                # Only track usage if not a built-in
                if arg not in BUILTIN_VARIABLES:
                    self._record_variable_use(arg)
            
            # Check if it's a type casting function
            if func_name in GLSL_TYPE_MAP:
                target_type = GLSL_TYPE_MAP[func_name]
                # Special handling for vector constructors
                if target_type.startswith(('vec', 'uvec', 'ivec')):
                    # This is a vector constructor
                    result = self.graph.add_operation(target_type, args, 
                                                    in_loop=bool(self.graph.current_scope))
                    self._increment_temp_usage(result)
                    return result
                else:
                    # Regular type cast
                    result = self.graph.add_operation('cast', [args[0], target_type], 
                                                    in_loop=bool(self.graph.current_scope))
                    self._increment_temp_usage(result)
                    return result
                        
            if func_name in self.inline_always:
                func_node, param_names = self.local_functions[func_name]
                saved_var_map = self.var_map.copy()
                saved_explicit_types = self.explicit_types.copy()
                
                for param, arg in zip(param_names, args):
                    self.var_map[param] = arg
                
                result_var = None
                for stmt in func_node.body:
                    if isinstance(stmt, ast.Return):
                        if stmt.value:
                            result_var = self.visit(stmt.value)
                    else:
                        self.visit(stmt)
                
                self.var_map = saved_var_map
                self.explicit_types = saved_explicit_types
                return result_var
            
            result = self.graph.add_operation(func_name, args, 
                                           in_loop=bool(self.graph.current_scope))
            self._increment_temp_usage(result)
            return result
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in function call '{func_name}': {e}", node)
    
    def visit_Compare(self, node):
        try:
            left = self.visit(node.left)
            # Only track usage if not a built-in
            if left not in BUILTIN_VARIABLES:
                self._record_variable_use(left)
            
            comparisons = []
            
            for op, right in zip(node.ops, node.comparators):
                right_var = self.visit(right)
                # Only track usage if not a built-in
                if right_var not in BUILTIN_VARIABLES:
                    self._record_variable_use(right_var)
                
                op_map = {
                    ast.Lt: 'lt', ast.LtE: 'lte', ast.Gt: 'gt',
                    ast.GtE: 'gte', ast.Eq: 'eq', ast.NotEq: 'neq',
                }
                
                op_type = type(op)
                if op_type not in op_map:
                    raise self._create_error(
                        f"Unsupported comparison: {op_type.__name__}", op
                    )
                
                result = self.graph.add_operation(op_map[op_type], [left, right_var], 
                                                in_loop=bool(self.graph.current_scope))
                self._increment_temp_usage(result)
                comparisons.append(result)
                left = right_var
            
            if len(comparisons) == 1:
                return comparisons[0]
            
            result = comparisons[0]
            for comp in comparisons[1:]:
                result = self.graph.add_operation('and', [result, comp], 
                                                in_loop=bool(self.graph.current_scope))
                self._increment_temp_usage(result)
            return result
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in comparison: {e}", node)
    
    def visit_If(self, node):
        try:
            condition = self.visit(node.test)
            # Only track usage if not a built-in
            if condition not in BUILTIN_VARIABLES:
                self._record_variable_use(condition)
            
            if_step = {
                'type': 'if',
                'condition': condition,
                'then_body': [],
                'else_body': [] if node.orelse else None
            }
            
            # Push the if step to graph's current_scope
            self.graph.current_scope.append(if_step)
            
            # Visit then body
            for stmt in node.body:
                self.visit(stmt)
            
            # Pop the if step
            self.graph.current_scope.pop()
            
            # If there's an else, do the same for else body
            if node.orelse:
                else_step = {
                    'type': 'else',
                    'body': []
                }
                self.graph.current_scope.append(else_step)
                
                for stmt in node.orelse:
                    self.visit(stmt)
                
                self.graph.current_scope.pop()
                if_step['else_body'] = else_step['body']
            
            # Add the if step to the graph
            if self.graph.current_scope:
                # We are inside another scope
                current_scope = self.graph.current_scope[-1]
                if 'body' not in current_scope:
                    current_scope['body'] = []
                current_scope['body'].append(if_step)
            else:
                self.graph.steps.append(if_step)
            
            return None
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in if statement: {e}", node)
    
        
    def visit_For(self, node):
        try:
            if not isinstance(node.target, ast.Name):
                raise self._create_error(
                    "Only simple loop variables supported", node.target
                )
            
            loop_var = node.target.id
            
            # Check if it's a range() call
            if not isinstance(node.iter, ast.Call):
                raise self._create_error(
                    "Only range() loops are supported", node.iter
                )
            
            if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != 'range':
                raise self._create_error(
                    "Only range() loops are supported", node.iter
                )
            
            # Validate range arguments
            args = node.iter.args
            if len(args) < 1 or len(args) > 3:
                raise self._create_error(
                    "range() requires 1-3 arguments", node.iter
                )
            
            # Validate each argument - allow variables as loop bounds
            processed_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, ast.Constant):
                    if not isinstance(arg.value, (int, float)):
                        raise self._create_error(
                            f"range() argument {i+1} must be numeric", arg
                        )
                    processed_args.append(str(arg.value))
                elif isinstance(arg, ast.Name):
                    # Variable as loop bound - check if it's defined
                    var_name = arg.id
                    if var_name not in self.var_map and var_name not in BUILTIN_VARIABLES:
                        raise self._create_error(
                            f"Variable '{var_name}' used as loop bound before definition", arg
                        )
                    processed_args.append(self.var_map.get(var_name, var_name))
                elif isinstance(arg, ast.UnaryOp) and isinstance(arg.op, (ast.USub, ast.UAdd)):
                    # Handle negative constants
                    if isinstance(arg.operand, ast.Constant):
                        if not isinstance(arg.operand.value, (int, float)):
                            raise self._create_error(
                                f"range() argument {i+1} must be numeric", arg
                            )
                        value = arg.operand.value
                        if isinstance(arg.op, ast.USub):
                            value = -value
                        processed_args.append(str(value))
                    elif isinstance(arg.operand, ast.Name):
                        var_name = arg.operand.id
                        if var_name not in self.var_map and var_name not in BUILTIN_VARIABLES:
                            raise self._create_error(
                                f"Variable '{var_name}' used as loop bound before definition", arg
                            )
                        var_ref = self.var_map.get(var_name, var_name)
                        if isinstance(arg.op, ast.USub):
                            processed_args.append(f"-{var_ref}")
                        else:
                            processed_args.append(var_ref)
                    else:
                        raise self._create_error(
                            f"range() argument {i+1} must be a simple numeric expression", arg
                        )
                elif isinstance(arg, ast.BinOp):
                    # Allow simple binary expressions like n_items + 1
                    try:
                        expr = self.visit(arg)
                        processed_args.append(expr)
                    except Exception:
                        raise self._create_error(
                            f"range() argument {i+1} must be a simple numeric expression", arg
                        )
                else:
                    raise self._create_error(
                        f"range() argument {i+1} must be a simple numeric expression", arg
                    )
            
            if len(processed_args) == 1:
                start, end, step = '0', processed_args[0], '1'
            elif len(processed_args) == 2:
                start, end, step = processed_args[0], processed_args[1], '1'
            elif len(processed_args) == 3:
                start, end, step = processed_args[0], processed_args[1], processed_args[2]
            
            loop_info = {
                'type': 'for',
                'var': loop_var,
                'start': start,
                'end': end,
                'step': step,
                'dynamic': any(not self._is_literal(arg) for arg in [start, end, step])
            }
            
            saved_var_map = self.var_map.copy()
            self.var_map[loop_var] = loop_var
            self.explicit_types[loop_var] = 'int'
            
            self.graph.start_loop(loop_info)
            
            for stmt in node.body:
                self.visit(stmt)
            
            self.graph.end_loop()
            
            self.var_map = saved_var_map
            if loop_var in self.explicit_types:
                del self.explicit_types[loop_var]
            
            return None
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in for loop: {e}", node)

    def visit_AnnAssign(self, node):
        try:
            if not isinstance(node.target, ast.Name):
                raise self._create_error(
                    "Only simple variable assignment supported", node.target
                )
            
            target = node.target.id
            
            # Check if target is a built-in variable (read-only)
            if target in BUILTIN_VARIABLES:
                raise self._create_error(
                    f"Cannot assign to built-in variable '{target}'", node.target
                )
            
            type_name = None
            array_dims = []
            
            # Parse type annotation
            if isinstance(node.annotation, ast.Subscript):
                # Array type like float[16] or float[16, 18]
                if isinstance(node.annotation.value, ast.Name):
                    type_name = node.annotation.value.id
                elif isinstance(node.annotation.value, ast.Attribute):
                    type_name = node.annotation.value.attr
                
                # Parse array dimensions
                slice_node = node.annotation.slice
                if isinstance(slice_node, ast.Tuple):
                    # Multiple dimensions: float[16, 18]
                    for dim in slice_node.elts:
                        if isinstance(dim, ast.Constant):
                            array_dims.append(str(dim.value))
                        elif isinstance(dim, ast.Name):
                            # Variable dimension
                            if dim.id not in self.var_map and dim.id not in BUILTIN_VARIABLES:
                                raise self._create_error(
                                    f"Variable '{dim.id}' used as array dimension before definition", dim
                                )
                            array_dims.append(self.var_map.get(dim.id, dim.id))
                        else:
                            raise self._create_error(
                                f"Invalid array dimension: {ast.dump(dim)}", dim
                            )
                elif isinstance(slice_node, ast.Constant):
                    # Single dimension: float[16]
                    array_dims.append(str(slice_node.value))
                elif isinstance(slice_node, ast.Name):
                    # Variable single dimension
                    if slice_node.id not in self.var_map and slice_node.id not in BUILTIN_VARIABLES:
                        raise self._create_error(
                            f"Variable '{slice_node.id}' used as array dimension before definition", slice_node
                        )
                    array_dims.append(self.var_map.get(slice_node.id, slice_node.id))
                else:
                    raise self._create_error(
                        f"Invalid array dimension specification", node.annotation
                    )
                    
            elif isinstance(node.annotation, ast.Name):
                type_name = node.annotation.id
            elif isinstance(node.annotation, ast.Attribute):
                type_name = node.annotation.attr
            
            # Use GLSL_TYPE_MAP from gl_typing
            glsl_type = GLSL_TYPE_MAP.get(type_name, 'float')
            
            if array_dims:
                # Array declaration
                full_type = glsl_type
                for dim in array_dims:
                    full_type += f"[{dim}]"
                
                self.explicit_types[target] = full_type
                self.var_types[target] = full_type  # ADD THIS LINE
                
                # Check if we have an initializer
                if node.value:
                    if isinstance(node.value, ast.List):
                        if len(node.value.elts) == 0:
                            # Empty list - uninitialized array
                            self.graph.add_operation('array_decl', [target, glsl_type] + array_dims, 
                                                    in_loop=bool(self.graph.current_scope))
                        else:
                            # List with values - initialize array
                            values = []
                            for elem in node.value.elts:
                                if isinstance(elem, ast.Constant):
                                    values.append(str(elem.value))
                                elif isinstance(elem, ast.Name):
                                    if elem.id in self.var_map or elem.id in BUILTIN_VARIABLES:
                                        values.append(self.var_map.get(elem.id, elem.id))
                                    else:
                                        raise self._create_error(
                                            f"Variable '{elem.id}' used in array initializer before definition", elem
                                        )
                                else:
                                    # Complex expression
                                    expr = self.visit(elem)
                                    values.append(expr)
                            
                            self.graph.add_operation('array_init', [target, glsl_type] + array_dims + values,
                                                    in_loop=bool(self.graph.current_scope))
                    else:
                        # Other initializer expression
                        init_expr = self.visit(node.value)
                        self.graph.add_operation('array_init_expr', [target, glsl_type] + array_dims + [init_expr],
                                                in_loop=bool(self.graph.current_scope))
                else:
                    # No initializer - just declare
                    self.graph.add_operation('array_decl', [target, glsl_type] + array_dims, 
                                            in_loop=bool(self.graph.current_scope))
                
                self.var_map[target] = target
            else:
                # Regular type declaration
                self.explicit_types[target] = glsl_type
                self.var_types[target] = glsl_type  # ADD THIS LINE
                
                # Special handling for built-in variables like 'gid' and 'idx'
                # These map to gl_GlobalInvocationID.x which is already defined
                if target in ['gid', 'idx'] and node.value:
                    if isinstance(node.value, ast.Constant) and node.value.value == 0:
                        # This is a marker for "use the built-in gid"
                        # Don't create a new variable, just map it
                        self.var_map[target] = 'gid'  # Always use 'gid' as it's already defined in shader
                        # Mark as int for Python semantics (GLSL will handle the uint conversion)
                        self.explicit_types[target] = 'int'
                        self.var_types[target] = 'int'  # ADD THIS LINE
                        return 'gid'
                
                if node.value:
                    result_var = self.visit(node.value)
                    self.var_map[target] = target
                    
                    # Track the type of the result variable if it's a temporary
                    if result_var.startswith('_t'):
                        # We'll infer the type later in ShaderCompiler
                        pass
                    
                    # Check if we can eliminate the temporary variable
                    if result_var.startswith('_t') and result_var in self.temp_var_usage_count:
                        # Check if this temp is only used here
                        if self.temp_var_usage_count.get(result_var, 0) <= 1:
                            # Replace the temp with target directly in future operations
                            self.var_map[result_var] = target
                            # Store type mapping for the temporary
                            self.var_types[result_var] = glsl_type
                            # We'll handle this during compilation, just store the mapping
                            self.graph.add_operation('direct_assign', [result_var, target, glsl_type], 
                                                    in_loop=bool(self.graph.current_scope))
                            return target
                    
                    self.graph.add_operation('assign', [target, result_var], 
                                            in_loop=bool(self.graph.current_scope))
                else:
                    self.var_map[target] = target
            
            return target
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in annotated assignment: {e}", node)


    def visit_While(self, node):
        try:
            test = self.visit(node.test)
            # Only track usage if not a built-in
            if test not in BUILTIN_VARIABLES:
                self._record_variable_use(test)
            
            loop_info = {'type': 'while', 'test': test}
            
            self.graph.start_loop(loop_info)
            
            for stmt in node.body:
                self.visit(stmt)
            
            self.graph.end_loop()
            
            return None
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in while loop: {e}", node)
    
    def visit_Try(self, node):
        raise self._create_error(
            "Try/except blocks are not supported in GPU kernels", node
        )
    
    def visit_ExceptHandler(self, node):
        raise self._create_error(
            "Exception handling is not supported in GPU kernels", node
        )
    
    def visit_Break(self, node):
        raise self._create_error(
            "Break statements are not supported in GPU kernels", node
        )
    
    def visit_Continue(self, node):
        raise self._create_error(
            "Continue statements are not supported in GPU kernels", node
        )
    
    def visit_Raise(self, node):
        raise self._create_error(
            "Raise statements are not supported in GPU kernels", node
        )
    
    def visit_With(self, node):
        raise self._create_error(
            "With statements are not supported in GPU kernels", node
        )
    
    def visit_Return(self, node):
        try:
            if node.value is None:
                # Void return - add operation if inside control flow
                if self.graph.current_scope:
                    self.graph.add_operation('return', [], output_var='_void_',
                                            in_loop=bool(self.graph.current_scope))
                else:
                    self.graph.set_void_return()
                return None
            
            result_var = self.visit(node.value)
            # Only track usage if not a built-in
            if result_var not in BUILTIN_VARIABLES:
                self._record_variable_use(result_var)
            
            # If inside control flow, add return operation
            if self.graph.current_scope:
                self.graph.add_operation('return', [result_var], output_var='_void_',
                                        in_loop=bool(self.graph.current_scope))
            else:
                self.graph.set_output(result_var)
            
            return result_var
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in return statement: {e}", node)
    
    def visit_FunctionDef(self, node):
        try:
            param_names = [arg.arg for arg in node.args.args]
            
            has_inline_always = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name) and decorator.id == 'inline_always':
                    has_inline_always = True
                    self.inline_always.add(node.name)
                    break
            
            self.local_functions[node.name] = (node, param_names)
            
            if not has_inline_always:
                saved_var_map = self.var_map.copy()
                saved_explicit_types = self.explicit_types.copy()
                saved_operations_count = len(self.graph.operations)
                saved_steps_count = len(self.graph.steps)
                
                for param_name in param_names:
                    self.var_map[param_name] = param_name
                    self.explicit_types[param_name] = 'float'
                
                body_ops = []
                return_type = 'float'
                return_var = None
                
                for stmt in node.body:
                    if isinstance(stmt, ast.Return):
                        if stmt.value is None:
                            return_type = 'void'
                        else:
                            return_var = self.visit(stmt.value)
                    else:
                        self.visit(stmt)
                
                new_operations = self.graph.operations[saved_operations_count:]
                body_ops.extend(new_operations)
                
                if return_var is not None:
                    body_ops.append(('return', [return_var], None))
                elif return_type == 'void':
                    body_ops.append(('return', [], None))
                
                param_types = []
                for param_name in param_names:
                    param_type = self.explicit_types.get(param_name, 'float')
                    param_types.append((param_name, param_type))
                
                self.graph.add_glsl_function(node.name, param_types, return_type, body_ops)
                
                self.var_map = saved_var_map
                self.explicit_types = saved_explicit_types
                self.graph.operations = self.graph.operations[:saved_operations_count]
                self.graph.steps = self.graph.steps[:saved_steps_count]
            
            return None
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in function definition: {e}", node)
    
    def visit_Expr(self, node):
        try:
            return self.visit(node.value)
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in expression: {e}", node)
    
    def generic_visit(self, node):
        """Handle nodes that don't have a specific visit method"""
        node_type = node.__class__.__name__
        
        # Provide more specific error messages for known unsupported constructs
        if node_type in ['Try', 'ExceptHandler', 'Break', 'Continue', 'Raise', 'With']:
            # These should have been caught by specific methods, but just in case
            raise self._create_error(
                f"{node_type} statements are not supported in GPU kernels", node
            )
        elif node_type in ['AsyncFunctionDef', 'Await', 'AsyncFor', 'AsyncWith']:
            raise self._create_error(
                f"Async {node_type.replace('Async', '')} is not supported in GPU kernels", node
            )
        elif node_type in ['ListComp', 'SetComp', 'DictComp', 'GeneratorExp']:
            raise self._create_error(
                f"Comprehensions ({node_type}) are not supported in GPU kernels", node
            )
        elif node_type in ['Yield', 'YieldFrom']:
            raise self._create_error(
                f"Generators ({node_type}) are not supported in GPU kernels", node
            )
        elif node_type in ['Match', 'MatchValue', 'MatchSingleton', 'MatchSequence', 
                          'MatchMapping', 'MatchClass', 'MatchStar', 'MatchAs', 'MatchOr']:
            raise self._create_error(
                f"Match statements ({node_type}) are not supported in GPU kernels", node
            )
        else:
            raise self._create_error(
                f"Unsupported syntax: {node_type}", node
            )


def _extract_hints(hints: Optional[Dict[str, tuple]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if hints is None:
        return {}, {}
    
    type_hints = {}
    storage_hints = {}
    
    for key, value in hints.items():
        if isinstance(value, tuple) and len(value) == 2:
            type_tuple, storage = value
            if isinstance(type_tuple, tuple) and len(type_tuple) == 2:
                # Check if it's an NP_GLTypes or Vec_GLTypes
                # NP_GLTypes: (numpy_type, glsl_type)
                # Vec_GLTypes: (panda3d_type, glsl_type)
                # We only care about the GLSL type for code generation
                glsl_type = type_tuple[1]
                type_hints[key] = glsl_type
            else:
                raise ValueError(f"Invalid type for '{key}': expected tuple like NP_GLTypes.float or Vec_GLTypes.vec3")
            storage_hints[key] = storage
        else:
            raise ValueError(f"Invalid hint for '{key}': expected (Type, storage) tuple")
    
    return type_hints, storage_hints



def static_constant(name: str, glsl_type: str, size: int, values: list):
    """Decorator to mark a variable as a static constant array"""
    def decorator(func):
        if not hasattr(func, '_static_constants'):
            func._static_constants = []
        func._static_constants.append((name, glsl_type, size, values))
        return func
    return decorator

def inline_always(func):
    func._inline_always = True
    return func




def _transpile_kernel(func):
    """Transpile a kernel function - called lazily on first compile"""
    import inspect
    import ast
    
    # Handle bound methods by getting the underlying function
    is_method = inspect.ismethod(func)
    actual_func = func.__func__ if is_method else func
    
    if not getattr(actual_func, '_needs_transpilation', False):
        return  # Already transpiled
    
    hints = getattr(actual_func, '_gpu_kernel_hints', None)
    layout = getattr(actual_func, '_gpu_kernel_layout', (64, 1, 1))
    
    # Get source file path
    try:
        file_path = inspect.getfile(actual_func)
    except:
        file_path = "<unknown>"
    
    source = inspect.getsource(actual_func)
    
    lines = source.split('\n')
    func_start = 0
    # Skip all decorator lines (starting with @) 
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('def '):
            func_start = i
            break
        # Skip decorators and empty lines
        if stripped.startswith('@') or not stripped:
            continue
    
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
    
    # For methods, skip 'self' parameter
    if is_method and arg_names and arg_names[0] == 'self':
        arg_names = arg_names[1:]
    
    # Extract static constants from function decorators (NOW they're applied!)
    static_constants = getattr(actual_func, '_static_constants', None)
    vectorized = getattr(actual_func, '_gpu_kernel_vectorized', None)
    
    transpiler = PythonToGLSLTranspiler(
        arg_names, hints, source, actual_func.__name__, file_path, static_constants, layout, vectorized
    )
    
    try:
        for stmt in func_def.body:
            transpiler.visit(stmt)
    except CompilationError:
        raise
    except Exception as e:
        # Create generic error info
        error_info = CompilationErrorInfo(
            file_path=file_path,
            function_name=actual_func.__name__,
            error_message=str(e)
        )
        raise CompilationError(f"Transpilation error: {e}", error_info) from e
    
    graph = transpiler.graph
    type_hints, storage_hints = _extract_hints(hints)
    
    actual_func._compute_graph = graph
    actual_func._arg_names = arg_names
    actual_func._type_hints = type_hints
    actual_func._storage_hints = storage_hints
    actual_func._explicit_types = transpiler.explicit_types
    actual_func._needs_transpilation = False

def gpu_kernel(hints: Optional[Dict[str, tuple]] = None, layout: Tuple[int, int, int] = (64, 1, 1), vectorized: Optional[bool] = None):
    def decorator(func):
        # Store hints for later transpilation
        func._gpu_kernel_hints = hints
        func._gpu_kernel_layout = layout
        func._gpu_kernel_vectorized = vectorized  # None means auto-detect based on IOTypes
        func._needs_transpilation = True
        return func
    
    if callable(hints):
        func = hints
        hints = None
        return decorator(func)
    
    return decorator