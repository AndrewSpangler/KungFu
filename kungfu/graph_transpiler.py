import ast
from typing import Dict, List, Tuple, Optional, Set, Any
from .ast_utils import ASTVisitorBase
from .compute_graph import ComputeGraph
from .shader_inputs import ShaderInputManager
from .gl_typing import (
    BUILTIN_VARIABLES, KUNGFU_BUILTINS, NP_GLTypes, GLSL_TYPE_MAP,
    ALL_GLSL_FUNCTIONS, GLSL_TYPE_CONSTRUCTORS,
    ShaderType, IOTypes
)
from .helpers import (
    CompilationError,
    is_kungfu_builtin,
    get_builtin_variables,
    get_kungfu_builtin_glsl
)

class GraphTranspiler(ASTVisitorBase, ast.NodeVisitor):
    def __init__(self, arg_names: List[str], shader_type: ShaderType = ShaderType.COMPUTE,
                type_hints: Dict[str, str] = None, storage_hints: Dict[str, str] = None,
                source_code: str = "", function_name: str = "", file_path: str = "",
                static_constants: List[Tuple[str, str, int, list]] = None,
                layout: Tuple[int, int, int] = (64, 1, 1), vectorized: Optional[bool] = None,
                is_shader_function: bool = False, custom_uniforms: Dict = None, 
                custom_textures: Dict = None):
        super().__init__(source_code, function_name, file_path)
        
        self.graph = ComputeGraph()
        self.arg_names = arg_names
        self.shader_type = shader_type
        self.type_hints = type_hints or {}
        self.storage_hints = storage_hints or {}
        self.static_constants = static_constants or []
        self.layout = layout
        self.is_shader_function = is_shader_function
        self.custom_uniforms = custom_uniforms or {}
        self.custom_textures = custom_textures or {}
        self.called_functions = set()
        self.used_builtins = set()
        
        self.graph.layout = layout
        self.graph.vectorized = vectorized
        
        self._initialize_builtins()
        self._initialize_from_hints()
        self._initialize_custom_uniforms()

    def _initialize_builtins(self):
        """Initialize built-in variables for shader type"""
        self.builtin_variables.update(BUILTIN_VARIABLES)
        
        if self.shader_type == ShaderType.COMPUTE:
            self.builtin_variables.update(KUNGFU_BUILTINS)
        else:
            self.builtin_variables.update(get_builtin_variables(self.shader_type))

    def _initialize_from_hints(self):
        """Initialize variables from hints and static constants"""
        if self.static_constants:
            for name, glsl_type, size, values in self.static_constants:
                self.graph.add_static_constant(name, glsl_type, size, values)
                self.var_map[name] = name
                self.explicit_types[name] = f"{glsl_type}[{size}]"
        
        for arg in self.arg_names:
            glsl_type = self.type_hints.get(arg, 'float')
            storage = self.storage_hints.get(arg, IOTypes.buffer)
            
            self.graph.add_input(arg, storage)
            self.var_map[arg] = arg
            # For array storage, store type with array notation for proper type inference
            if storage == IOTypes.array:
                self.explicit_types[arg] = f"{glsl_type}[]"
            else:
                self.explicit_types[arg] = glsl_type
            
            # Store the storage type in a separate map for later use
            if not hasattr(self, 'input_storage'):
                self.input_storage = {}
            self.input_storage[arg] = storage

    def _initialize_custom_uniforms(self):
        """Initialize custom uniforms and textures from decorator"""
        # Add custom uniforms to the graph inputs
        for name, info in self.custom_uniforms.items():
            if isinstance(info, tuple):
                glsl_type = info[0]
            else:
                glsl_type = info
            
            # Add as uniform input
            self.graph.add_input(name, IOTypes.uniform)
            self.var_map[name] = name
            self.explicit_types[name] = glsl_type
            # Also add to storage hints
            self.graph.storage_hints[name] = IOTypes.uniform
        
        # Add custom textures
        for name, info in self.custom_textures.items():
            if isinstance(info, tuple):
                glsl_type = info[0]
            else:
                glsl_type = info
            
            # Add as uniform input (textures are uniforms in GLSL)
            self.graph.add_input(name, IOTypes.uniform)
            self.var_map[name] = name
            self.explicit_types[name] = glsl_type
            # Also add to storage hints
            self.graph.storage_hints[name] = IOTypes.uniform

    def _handle_builtin_variable(self, var_name: str) -> Optional[str]:
        """Handle built-in variables"""
        if self.shader_type == ShaderType.COMPUTE and is_kungfu_builtin(var_name):
            if var_name == 'gid' and getattr(self, 'has_vectorized_inputs', False):
                return 'gl_GlobalInvocationID.x'
            elif var_name == 'n_items' and getattr(self, 'has_vectorized_inputs', False):
                return 'nItems'
            return get_kungfu_builtin_glsl(var_name)
        
        is_builtin, _ = ShaderInputManager.is_builtin_variable(var_name)
        if is_builtin:
            self.used_builtins.add(var_name)
            return var_name
        
        return None

    def _add_operation(self, op_name: str, inputs: list) -> str:
        """Add operation to compute graph"""
        output_var = f"_t{self.graph.var_counter}"
        self.graph.var_counter += 1
        self.graph.add_operation(op_name, inputs, output_var, 
                                in_loop=bool(self.graph.current_scope))
        return output_var

    def _handle_chained_subscript(self, node, is_store=False):
        """Handle multi-dimensional array subscript"""
        indices = []
        current = node
        while isinstance(current, ast.Subscript):
            indices.insert(0, self.visit(current.slice))
            current = current.value
        
        array_var = self.visit(current)
        
        if is_store:
            return array_var, indices
        else:
            if len(indices) == 1:
                result = self.graph.add_operation('subscript', [array_var, indices[0]], 
                                            in_loop=bool(self.graph.current_scope))
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
        return self._visit_constant(node)

    def visit_Name(self, node):
        return self._visit_name(node, self._handle_builtin_variable)

    def visit_Subscript(self, node):
        try:
            if isinstance(node.ctx, ast.Store):
                return node
            else:
                return self._handle_chained_subscript(node, is_store=False)
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in array subscript: {e}", node)

    def visit_Attribute(self, node):
        try:
            value = self.visit(node.value)
            
            if value in ['gl_GlobalInvocationID', 'gl_WorkGroupID', 'gl_LocalInvocationID',
                        'gl_WorkGroupSize', 'gl_NumWorkGroups', 'gid_xyz', 'wgid', 'lid',
                        'wg_size', 'num_wg'] or value in BUILTIN_VARIABLES:
                attr = node.attr
                if attr in ['x', 'y', 'z', 'r', 'g', 'b']:
                    result = self._add_operation('swizzle', [value, attr])
                    self._increment_temp_usage(result)
                    return result
            
            # Check if it's a valid variable or temporary variable
            is_temp_var = value.startswith('_t') if isinstance(value, str) else False
            if value in self.var_map or value in self.builtin_variables or is_temp_var:
                attr = node.attr
                if attr in ['x', 'y', 'z', 'w', 'r', 'g', 'b', 'a'] or attr in ['xy', 'xyz', 'xyzw', 'rgb', 'rgba']:
                    result = self._add_operation('swizzle', [value, attr])
                    self._increment_temp_usage(result)
                    return result
            
            raise self._create_error(
                f"Unsupported attribute access: {value}.{node.attr}", node
            )
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in attribute access: {e}", node)

    def visit_BinOp(self, node):
        return self._visit_binop(node, self._add_operation)

    def visit_Compare(self, node):
        return self._visit_compare(node, self._add_operation)

    def visit_Call(self, node):
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
        
        # Check for local inline functions first
        if func_name in self.local_functions:
            result = self.graph.add_operation('function_call', [func_name] + args, 
                                            in_loop=bool(self.graph.current_scope))
            self._increment_temp_usage(result)
            return result
        
        from .function_registry import FunctionRegistry
        shader_func = FunctionRegistry.get(func_name)
        if shader_func:
            self.called_functions.add(func_name)
            result = self.graph.add_operation('function_call', [func_name] + args, 
                                            in_loop=bool(self.graph.current_scope))
            self._increment_temp_usage(result)
            # Store the return type of the function
            return_type = shader_func.get('return_type', 'float')
            self.var_types[result] = return_type
            self.graph.var_types[result] = return_type
            return result
        
        # Use centralized function sets from gl_typing
        if func_name in GLSL_TYPE_CONSTRUCTORS:
            result = self._add_operation(func_name, args)
            self._increment_temp_usage(result)
            return result
        
        if func_name in ALL_GLSL_FUNCTIONS:
            result = self._add_operation(func_name, args)
            self._increment_temp_usage(result)
            return result
        
        raise self._create_error(
            f"Function '{func_name}' not recognized. Available: GLSL functions, type constructors, or @function decorated functions", 
            node
        )

    def visit_UnaryOp(self, node):
        try:
            operand = self.visit(node.operand)
            if operand not in self.builtin_variables:
                self._record_variable_use(operand)
            
            op_map = {
                ast.USub: 'neg', ast.Not: 'bool_not', ast.Invert: 'bitwise_not',
            }
            
            op_type = type(node.op)
            if op_type in op_map:
                result = self._add_operation(op_map[op_type], [operand])
                self._increment_temp_usage(result)
                return result
            else:
                raise self._create_error(
                    f"Unsupported unary operation: {op_type.__name__}", node.op
                )
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in unary operation: {e}", node)

    def visit_BoolOp(self, node):
        try:
            op_map = {ast.And: 'logical_and', ast.Or: 'logical_or'}
            op_type = type(node.op)
            
            if op_type not in op_map:
                raise self._create_error(
                    f"Unsupported boolean operation: {op_type.__name__}", node.op
                )
            
            values = [self.visit(value) for value in node.values]
            for val in values:
                if val not in self.builtin_variables:
                    self._record_variable_use(val)
            
            if len(values) < 2:
                raise self._create_error(
                    f"Boolean operation requires at least 2 values", node
                )
            
            result = values[0]
            for val in values[1:]:
                result = self._add_operation(op_map[op_type], [result, val])
                self._increment_temp_usage(result)
            
            return result
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in boolean operation: {e}", node)

    def visit_Assign(self, node):
        try:
            if len(node.targets) > 1:
                raise self._create_error(
                    "Multiple assignment targets not supported", node
                )
            
            target = node.targets[0]
            
            if isinstance(target, ast.Subscript):
                array_var, indices = self._handle_chained_subscript(target, is_store=True)
                value_var = self.visit(node.value)
                
                if len(indices) == 1:
                    self.graph.add_operation('subscript_assign', 
                                        [array_var, indices[0], value_var],
                                        in_loop=bool(self.graph.current_scope))
                elif len(indices) == 2:
                    self.graph.add_operation('subscript_assign_2d',
                                        [array_var, indices[0], indices[1], value_var],
                                        in_loop=bool(self.graph.current_scope))
                return
            
            # Handle attribute assignments (e.g., hsv.y = 0.8)
            if isinstance(target, ast.Attribute):
                # Get the base object (e.g., hsv)
                base_var = self.visit(target.value)
                attr_name = target.attr
                value_var = self.visit(node.value)
                
                # Get the base variable name from var_map
                base_var_name = base_var
                if base_var in self.var_map and self.var_map[base_var] != base_var:
                    base_var_name = self.var_map[base_var]
                
                # Create a swizzle operation to rebuild the vector with the new component
                result_var = f"_t{self.graph.var_counter}"
                self.graph.var_counter += 1
                
                # Determine which component is being modified
                component_index = {'x': 0, 'y': 1, 'z': 2, 'w': 3, 
                                'r': 0, 'g': 1, 'b': 2, 'a': 3}.get(attr_name, 0)
                
                # Get the vector type
                vector_type = self._get_var_type(base_var)
                
                # Create a new vector with the modified component
                # This is handled in the compiler by creating a vec constructor
                self.graph.add_operation('set_vector_component', 
                                    [base_var_name, str(component_index), value_var],
                                    output_var=result_var,
                                    in_loop=bool(self.graph.current_scope))
                
                # Update the variable mapping
                # Find which variable name maps to this base_var
                for var_name, mapped_var in self.var_map.items():
                    if mapped_var == base_var_name:
                        self.var_map[var_name] = result_var
                        # Copy type information
                        if base_var_name in self.var_types:
                            self.var_types[result_var] = self.var_types[base_var_name]
                            self.graph.var_types[result_var] = self.var_types[base_var_name]
                        break
                
                return
            
            if not isinstance(target, ast.Name):
                raise self._create_error(
                    f"Unsupported assignment target: {type(target).__name__}", target
                )
            
            var_name = target.id
            value_var = self.visit(node.value)
            
            # Check if this is a built-in output variable
            is_builtin, _ = ShaderInputManager.is_builtin_variable(var_name)
            if is_builtin or var_name in self.builtin_variables:
                self.graph.add_operation('assign_builtin', [value_var, var_name], 
                                        output_var=var_name,
                                        in_loop=bool(self.graph.current_scope))
                self.var_map[var_name] = var_name
                if var_name in self.explicit_types:
                    self.var_types[var_name] = self.explicit_types[var_name]
                    self.graph.var_types[var_name] = self.explicit_types[var_name]
                return
            
            if var_name in self.var_map:
                # This is a reassignment - update the existing variable
                existing_var = self.var_map[var_name]
                
                # Create an assignment operation that updates the variable
                self.graph.add_operation('assign', [value_var, existing_var], 
                                        output_var=existing_var,
                                        in_loop=bool(self.graph.current_scope))
                
                # Update the variable type if we have explicit type info
                if var_name in self.explicit_types:
                    self.var_types[existing_var] = self.explicit_types[var_name]
                    self.graph.var_types[existing_var] = self.explicit_types[var_name]
            else:
                # First assignment - create a temporary variable for it
                temp_var = f"_t{self.graph.var_counter}"
                self.graph.var_counter += 1
                self.var_map[var_name] = temp_var
                
                # Add operation to assign initial value
                self.graph.add_operation('assign', [value_var, temp_var],
                                        output_var=temp_var,
                                        in_loop=bool(self.graph.current_scope))
                
                # Store type information if available
                if var_name in self.explicit_types:
                    value_type = self.explicit_types[var_name]
                    self.var_types[temp_var] = value_type
                    self.graph.var_types[temp_var] = value_type
                else:
                    # Try to infer type from value
                    value_type = self._get_var_type(value_var)
                    self.var_types[temp_var] = value_type
                    self.graph.var_types[temp_var] = value_type
            
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in assignment: {e}", node)

    def visit_AnnAssign(self, node):
        try:
            if not isinstance(node.target, ast.Name):
                raise self._create_error(
                    f"Unsupported annotated assignment target: {type(node.target).__name__}", 
                    node.target
                )
            
            var_name = node.target.id
            
            if isinstance(node.annotation, ast.Name):
                type_name = node.annotation.id
                if type_name in GLSL_TYPE_MAP:
                    glsl_type = GLSL_TYPE_MAP[type_name]
                    self.explicit_types[var_name] = glsl_type
            elif isinstance(node.annotation, ast.Subscript):
                # Handle array types like float[10] or float[8, 8]
                if isinstance(node.annotation.value, ast.Name):
                    base_type = node.annotation.value.id
                    if base_type in GLSL_TYPE_MAP:
                        glsl_base = GLSL_TYPE_MAP[base_type]
                        
                        # Handle multi-dimensional arrays
                        if isinstance(node.annotation.slice, ast.Tuple):
                            # Multi-dimensional: float[8, 8]
                            dims = [str(elt.value) for elt in node.annotation.slice.elts 
                                if isinstance(elt, ast.Constant)]
                            array_type = glsl_base
                            for dim in reversed(dims):
                                array_type = f"{array_type}[{dim}]"
                            self.explicit_types[var_name] = array_type
                        elif isinstance(node.annotation.slice, ast.Constant):
                            # Single dimension: float[10]
                            size = node.annotation.slice.value
                            self.explicit_types[var_name] = f"{glsl_base}[{size}]"
            
            if node.value:
                value_var = self.visit(node.value)
                
                # Check if this is a built-in output variable (like gl_Position, p3d_FragColor)
                is_builtin, _ = ShaderInputManager.is_builtin_variable(var_name)
                if is_builtin or var_name in self.builtin_variables:
                    # For built-in variables, create an assignment operation
                    self.graph.add_operation('assign_builtin', [value_var, var_name], 
                                            output_var=var_name,
                                            in_loop=bool(self.graph.current_scope))
                    # Also map the variable for potential future references
                    self.var_map[var_name] = var_name
                    if var_name in self.explicit_types:
                        self.var_types[var_name] = self.explicit_types[var_name]
                        self.graph.var_types[var_name] = self.explicit_types[var_name]
                else:
                    # Check if variable was pre-declared (e.g., from if-statement pre-scanning)
                    if var_name in self.var_map:
                        # Variable already exists, just assign to it
                        existing_var = self.var_map[var_name]
                        self.graph.add_operation('assign', [value_var, existing_var], 
                                                output_var=existing_var,
                                                in_loop=bool(self.graph.current_scope))
                        
                        # Update type information
                        if var_name in self.explicit_types:
                            value_type = self.explicit_types[var_name]
                            self.var_types[existing_var] = value_type
                            self.graph.var_types[existing_var] = value_type
                    # For regular variables, check if value is already a temp variable
                    elif value_var.startswith('_t') and value_var in self.var_types:
                        # Value is already a temporary, just map directly to it
                        self.var_map[var_name] = value_var
                        
                        if var_name in self.explicit_types:
                            value_type = self.explicit_types[var_name]
                            self.var_types[value_var] = value_type
                            self.graph.var_types[value_var] = value_type
                    else:
                        # Value is not a temp (it's a literal or needs conversion), create a temp variable and assign to it
                        temp_var = f"_t{self.graph.var_counter}"
                        self.graph.var_counter += 1
                        self.var_map[var_name] = temp_var
                        
                        # Add assignment operation
                        self.graph.add_operation('assign', [value_var, temp_var],
                                                output_var=temp_var,
                                                in_loop=bool(self.graph.current_scope))
                        
                        if var_name in self.explicit_types:
                            value_type = self.explicit_types[var_name]
                            self.var_types[temp_var] = value_type
                            self.graph.var_types[temp_var] = value_type
            else:
                # Declaration without initialization
                # For array types, we need to declare them properly
                if var_name in self.explicit_types:
                    var_type = self.explicit_types[var_name]
                    
                    # Check if this is an array type
                    if '[' in var_type and ']' in var_type:
                        # Array type - create a proper variable name and mark it for declaration
                        temp_var = f"_t{self.graph.var_counter}"
                        self.graph.var_counter += 1
                        self.var_map[var_name] = temp_var
                        self.var_types[temp_var] = var_type
                        self.graph.var_types[temp_var] = var_type
                        
                        # Add a special operation to declare the array
                        self.graph.add_operation('declare_array', [var_type],
                                                output_var=temp_var,
                                                in_loop=bool(self.graph.current_scope))
                    else:
                        # Regular variable without initialization
                        temp_var = f"_t{self.graph.var_counter}"
                        self.graph.var_counter += 1
                        self.var_map[var_name] = temp_var
                        self.var_types[temp_var] = var_type
                        self.graph.var_types[temp_var] = var_type
                else:
                    # No type annotation - default
                    temp_var = f"_t{self.graph.var_counter}"
                    self.graph.var_counter += 1
                    self.var_map[var_name] = temp_var
        
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in annotated assignment: {e}", node)

    def visit_Return(self, node):
        try:
            if node.value is None:
                self.graph.set_void_return()
                # Add a return operation
                self.graph.add_operation('return', [], 
                                        in_loop=bool(self.graph.current_scope))
            else:
                result = self.visit(node.value)
                self.graph.set_output(result)
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in return statement: {e}", node)

    def visit_If(self, node):
        try:
            # Pre-scan to find variables assigned in any branch
            assigned_vars = set()
            var_annotations = {}
            
            def find_assignments(stmts):
                for stmt in stmts:
                    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                        if isinstance(stmt, ast.Assign):
                            target = stmt.targets[0]
                        else:
                            target = stmt.target
                            # Extract type annotation
                            if isinstance(stmt.annotation, ast.Name):
                                type_name = stmt.annotation.id
                                if type_name in GLSL_TYPE_MAP:
                                    var_annotations[target.id] = GLSL_TYPE_MAP[type_name]
                        if isinstance(target, ast.Name):
                            assigned_vars.add(target.id)
                    elif isinstance(stmt, ast.If):
                        find_assignments(stmt.body)
                        find_assignments(stmt.orelse)
            
            find_assignments(node.body)
            find_assignments(node.orelse)
            
            # Pre-declare variables that will be assigned in branches
            for var_name in assigned_vars:
                if var_name not in self.var_map:
                    temp_var = f"_t{self.graph.var_counter}"
                    self.graph.var_counter += 1
                    self.var_map[var_name] = temp_var
                    # Store type from annotation if available
                    if var_name in var_annotations:
                        var_type = var_annotations[var_name]
                        self.explicit_types[var_name] = var_type
                        self.var_types[temp_var] = var_type
                        self.graph.var_types[temp_var] = var_type
                    else:
                        # Store a placeholder type (will be updated on actual assignment)
                        self.var_types[temp_var] = 'float'
                        self.graph.var_types[temp_var] = 'float'
            
            test_var = self.visit(node.test)
            
            if_step = {
                'type': 'if',
                'condition': test_var,
                'then_body': [],
                'else_body': [],
                'has_return': False
            }
            
            if self.graph.current_scope:
                parent_scope = self.graph.current_scope[-1]
                if 'body' not in parent_scope:
                    parent_scope['body'] = []
                parent_scope['body'].append(if_step)
            else:
                self.graph.steps.append(if_step)
            
            self.graph.current_scope.append(if_step)
            
            for stmt in node.body:
                if isinstance(stmt, ast.Return):
                    # Add explicit return operation
                    if_step['then_body'].append({
                        'type': 'operation',
                        'op_name': 'return',
                        'inputs': [],
                        'output_var': None
                    })
                    if_step['has_return'] = True
                    break  # Stop processing after return
                else:
                    self.visit(stmt)
            
            self.graph.current_scope.pop()
            
            if node.orelse:
                else_step = {
                    'type': 'else',
                    'body': []
                }
                if_step['else_body'] = []
                
                self.graph.current_scope.append(else_step)
                
                for stmt in node.orelse:
                    self.visit(stmt)
                
                self.graph.current_scope.pop()
                if_step['else_body'] = else_step['body']
        
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in if statement: {e}", node)

    def visit_For(self, node):
        try:
            if not isinstance(node.iter, ast.Call):
                raise self._create_error(
                    "Only range() loops are supported", node.iter
                )
            
            if not isinstance(node.iter.func, ast.Name) or node.iter.func.id != 'range':
                raise self._create_error(
                    "Only range() loops are supported", node.iter
                )
            
            args = node.iter.args
            if len(args) == 1:
                start = "0"
                end = self.visit(args[0])
                step = "1"
            elif len(args) == 2:
                start = self.visit(args[0])
                end = self.visit(args[1])
                step = "1"
            elif len(args) == 3:
                start = self.visit(args[0])
                end = self.visit(args[1])
                step = self.visit(args[2])
            else:
                raise self._create_error(
                    "range() requires 1-3 arguments", node.iter
                )
            
            if not isinstance(node.target, ast.Name):
                raise self._create_error(
                    "Loop target must be a simple variable", node.target
                )
            
            loop_var = node.target.id
            
            is_dynamic = not (self._is_literal(start) and self._is_literal(end))
            
            loop_info = {
                'type': 'for',
                'var': loop_var,
                'start': start,
                'end': end,
                'step': step,
                'dynamic': is_dynamic
            }
            
            loop_step = self.graph.start_loop(loop_info)
            self.var_map[loop_var] = loop_var
            self.var_types[loop_var] = 'int'
            
            for stmt in node.body:
                self.visit(stmt)
            
            self.graph.end_loop()
        
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in for loop: {e}", node)

    def visit_While(self, node):
        try:
            test_var = self.visit(node.test)
            
            loop_info = {
                'type': 'while',
                'test': test_var
            }
            
            loop_step = self.graph.start_loop(loop_info)
            
            for stmt in node.body:
                self.visit(stmt)
            
            self.graph.end_loop()
        
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in while loop: {e}", node)

    def visit_Expr(self, node):
        try:
            # Skip string constants (docstrings)
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return
            
            result = self.visit(node.value)
            
            if self.graph.current_scope:
                self.graph.add_operation('expression_result', [result],
                                    in_loop=True)
            else:
                self.graph.add_operation('expression_result', [result])
        except CompilationError:
            raise
        except Exception as e:
            raise self._create_error(f"Error in expression statement: {e}", node)

    def visit_FunctionDef(self, node):
        """Handle nested function definitions (inline functions)"""
        try:
            func_name = node.name
            param_names = [arg.arg for arg in node.args.args]
            
            # Store as local function that can be inlined
            self.local_functions[func_name] = {
                'name': func_name,
                'params': param_names,
                'ast_node': node
            }
            
            # Mark for inlining if decorated with @inline_always
            if hasattr(node, 'decorator_list'):
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name) and dec.id == 'inline_always':
                        self.inline_always.add(func_name)
            
            return None
        except Exception as e:
            raise self._create_error(f"Error in function definition: {e}", node)

    def generic_visit(self, node):
        node_type = node.__class__.__name__
        raise self._create_error(
            f"Unsupported syntax: {node_type}", node
        )