import ast
from typing import Dict, List, Tuple
from .composition import get_standard_heading, _buff_line
from .compute_graph import ComputeGraph
from .shader_inputs import ShaderInputManager
from .function_registry import FunctionRegistry
from .shader_functions import ShaderFunctionTranspiler
from .helpers import get_op_glsl, get_shader_version
from .gl_typing import (
    GLTypes, NP_GLTypes, GLSL_TO_NP, TypeRules,
    BUILTIN_VARIABLES, GLSL_CONSTRUCTORS, ShaderType,
    SHADER_TEMPLATES, IOTypes
)

class UnifiedCompiler:
    def __init__(self, graph: ComputeGraph, shader_type: ShaderType, 
                input_types: Dict[str, str], explicit_types: Dict[str, str] = None,
                used_builtins: set = None, custom_uniforms: Dict = None,
                custom_textures: Dict = None, local_functions: Dict = None):
        self.graph = graph
        self.shader_type = shader_type
        self.input_types = input_types
        self.var_types = input_types.copy()
        self.var_types.update(graph.var_types)  # Merge graph variable types
        self.explicit_types = explicit_types or {}
        self.used_builtins = used_builtins or set()
        self.custom_uniforms = custom_uniforms or {}
        self.custom_textures = custom_textures or {}
        self.local_functions = local_functions or {}
        self.temporary_types = {}
        self.declared_vars = set()
        
        self.declared_vars.update(graph.input_vars)
        self.declared_vars.update(graph.uniform_vars)
        
        # Add compute shader builtins
        for builtin_name in BUILTIN_VARIABLES:
            self.declared_vars.add(builtin_name)
            self.var_types[builtin_name] = BUILTIN_VARIABLES[builtin_name]
        
        # Add shader-specific builtins (Panda3D and GLSL)
        from .gl_typing import PANDA3D_BUILTINS, GLSL_BUILTINS
        for builtin_name, builtin_type in PANDA3D_BUILTINS.items():
            self.var_types[builtin_name] = builtin_type
        
        shader_builtins = {}
        if shader_type == ShaderType.VERTEX:
            shader_builtins = GLSL_BUILTINS.get('vertex', {})
        elif shader_type == ShaderType.FRAGMENT:
            shader_builtins = GLSL_BUILTINS.get('fragment', {})
        elif shader_type == ShaderType.GEOMETRY:
            shader_builtins = GLSL_BUILTINS.get('geometry', {})
        
        for builtin_name, builtin_type in shader_builtins.items():
            self.var_types[builtin_name] = builtin_type
        
        self.function_dependencies = set()
        self.temp_to_final_map = {}
        self.operation_expressions = {}
        
        self.inputs = []
        self.outputs = []
        self.uniforms = []
        self.textures = []
        
        self.var_declaration_scope = {}
        self.var_usage_scope = {}
        self.current_scope_depth = 0
        self.vars_to_hoist = set()

    def _collect_function_dependencies(self, steps):
        """Collect all function dependencies from the compute graph"""
        def collect_from_steps(steps_list):
            for step in steps_list:
                if step['type'] == 'operation' and step['op_name'] == 'function_call':
                    if step['inputs']:
                        func_name = step['inputs'][0]
                        self.function_dependencies.add(func_name)
                
                elif step['type'] == 'loop' and 'body' in step:
                    collect_from_steps(step['body'])
                elif step['type'] == 'if':
                    if 'then_body' in step:
                        collect_from_steps(step['then_body'])
                    if 'else_body' in step and step['else_body']:
                        collect_from_steps(step['else_body'])
        
        collect_from_steps(steps)

    def _analyze_variable_scopes(self, steps, depth=0):
        """Analyze which variables are declared and used at which scope levels"""
        for step in steps:
            if step['type'] == 'operation':
                output_var = step.get('output_var')
                if output_var and output_var.startswith('_t'):
                    if output_var not in self.var_declaration_scope:
                        self.var_declaration_scope[output_var] = depth
                    
                # Track inputs - they are being used at this depth
                for input_var in step.get('inputs', []):
                    if input_var and input_var.startswith('_t'):
                        if input_var not in self.var_usage_scope:
                            self.var_usage_scope[input_var] = depth
                        else:
                            # Track minimum usage depth (outermost scope where used)
                            self.var_usage_scope[input_var] = min(
                                self.var_usage_scope[input_var], depth
                            )
            
            elif step['type'] in ('loop', 'if'):
                if step['type'] == 'loop' and 'body' in step:
                    self._analyze_variable_scopes(step['body'], depth + 1)
                elif step['type'] == 'if':
                    if 'then_body' in step:
                        self._analyze_variable_scopes(step['then_body'], depth + 1)
                    if 'else_body' in step and step['else_body']:
                        self._analyze_variable_scopes(step['else_body'], depth + 1)
        
        # After analyzing all steps at this depth, propagate usage information up
        # Check for variables that are used later at this depth
        if depth == 0:
            # Mark the output variable as used at depth 0
            if self.graph.output_var and self.graph.output_var.startswith('_t'):
                self.var_usage_scope[self.graph.output_var] = 0

    def _compute_hoist_requirements(self):
        """Determine which variables need to be hoisted to outer scopes"""
        for var_name in self.var_usage_scope:
            if var_name in self.var_declaration_scope:
                decl_depth = self.var_declaration_scope[var_name]
                usage_depth = self.var_usage_scope[var_name]
                
                if usage_depth < decl_depth:
                    self.vars_to_hoist.add(var_name)

    def _transpile_local_function(self, func_info: Dict) -> str:
        """Transpile a local inline function to GLSL"""
        func_name = func_info["name"]
        params = func_info["params"]
        func_node = func_info["ast_node"]
        
        param_types = {}
        return_type = 'float'
        
        # Extract type annotations
        for arg in func_node.args.args:
            if arg.annotation and isinstance(arg.annotation, ast.Name):
                param_types[arg.arg] = arg.annotation.id
            else:
                param_types[arg.arg] = 'float'
        
        if func_node.returns and isinstance(func_node.returns, ast.Name):
            return_type = func_node.returns.id
        
        param_str = ", ".join(f"{param_types.get(p, 'float')} {p}" for p in params)
        
        # Transpile body using base transpiler
        from .base_transpiler import BaseTranspiler
        transpiler = BaseTranspiler()
        
        # Set up variable mapping for parameters
        for param in params:
            transpiler.var_map[param] = param
            transpiler.var_types[param] = param_types.get(param, 'float')
        
        body_lines = []
        for stmt in func_node.body:
            if isinstance(stmt, ast.Return):
                if stmt.value:
                    expr = transpiler.visit(stmt.value)
                    body_lines.append(f"\treturn {expr};")
                else:
                    body_lines.append(f"\treturn;")
            elif isinstance(stmt, ast.Assign):
                # Handle variable assignments
                target = stmt.targets[0]
                if isinstance(target, ast.Name):
                    value = transpiler.visit(stmt.value)
                    var_name = target.id
                    if var_name not in transpiler.var_map:
                        transpiler.var_map[var_name] = var_name
                        # Infer type from annotation if available
                        var_type = 'float'
                        if hasattr(stmt, 'annotation') and isinstance(stmt.annotation, ast.Name):
                            var_type = stmt.annotation.id
                        body_lines.append(f"\t{var_type} {var_name} = {value};")
                    else:
                        body_lines.append(f"\t{var_name} = {value};")
            elif isinstance(stmt, ast.AnnAssign):
                # Handle annotated assignments
                target = stmt.target
                if isinstance(target, ast.Name):
                    var_name = target.id
                    var_type = transpiler.visit(stmt.annotation) if stmt.annotation else 'float'
                    if stmt.value:
                        value = transpiler.visit(stmt.value)
                        body_lines.append(f"\t{var_type} {var_name} = {value};")
                    else:
                        body_lines.append(f"\t{var_type} {var_name};")
                    transpiler.var_map[var_name] = var_name
                    transpiler.var_types[var_name] = var_type
            elif isinstance(stmt, ast.Expr):
                expr = transpiler.visit(stmt.value)
                if expr:
                    body_lines.append(f"\t{expr};")
        
        body = "\n".join(body_lines) if body_lines else "\treturn 0.0;"
        
        return f"{return_type} {func_name}({param_str}) {{\n{body}\n}}"

    def _get_function_declarations(self) -> List[str]:
        """Get GLSL declarations for all required functions"""
        declarations = []
        
        if not self.function_dependencies:
            return declarations
        
        # Handle local inline functions first (they may be dependencies)
        for func_name in list(self.function_dependencies):
            if func_name in self.local_functions:
                local_func = self.local_functions[func_name]
                func_decl = self._transpile_local_function(local_func)
                if func_decl:
                    declarations.append(func_decl)
        
        # Handle registered shader functions
        all_deps = set()
        for func_name in self.function_dependencies:
            if func_name not in self.local_functions:
                all_deps.update(ShaderFunctionTranspiler.get_all_dependencies(func_name))
        all_deps.update(self.function_dependencies)
        
        for func_name in all_deps:
            if func_name in self.local_functions:
                continue
            func_metadata = FunctionRegistry.get(func_name)
            if func_metadata:
                glsl_code = ShaderFunctionTranspiler.transpile_function(
                    func_metadata, self.shader_type
                )
                declarations.append(glsl_code)
        
        return declarations

    def _is_literal(self, val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return val.lower() in ['true', 'false']

    def _get_var_type(self, var_name: str) -> str:
        if var_name in self.explicit_types:
            return self.explicit_types[var_name]
        if var_name in self.var_types:
            return self.var_types[var_name]
        if var_name in self.temporary_types:
            return self.temporary_types[var_name]
        if self._is_literal(var_name):
            if '.' in var_name or 'e' in var_name.lower():
                return 'float'
            elif var_name.lower() in ['true', 'false']:
                return 'bool'
            else:
                try:
                    int(var_name)
                    return 'int'
                except ValueError:
                    return 'float'
        return 'float'

    def promote_types(self, type1: str, type2: str) -> str:
        return TypeRules.promote_types(type1, type2)

    def infer_type(self, op_name: str, input_vars: List[str]) -> str:
        if op_name in GLSL_CONSTRUCTORS:
            return op_name
        
        # Handle function calls - get return type from function registry
        if op_name == 'function_call' and input_vars:
            func_name = input_vars[0]
            # Check local functions first
            if func_name in self.local_functions:
                local_func = self.local_functions[func_name]
                func_node = local_func['ast_node']
                if func_node.returns and isinstance(func_node.returns, ast.Name):
                    return func_node.returns.id
                return 'float'
            
            # Check registered functions
            func_metadata = FunctionRegistry.get(func_name)
            if func_metadata:
                return func_metadata.get('return_type', 'float')
        
        # Handle swizzle to get correct component type
        if op_name == 'swizzle' and input_vars:
            base_type = self._get_var_type(input_vars[0])
            if base_type.startswith('uvec'):
                return 'uint'
            elif base_type.startswith('ivec'):
                return 'int'
            elif base_type.startswith('vec'):
                return 'float'
            # For single component swizzle
            return base_type
        
        input_types = [self._get_var_type(v) for v in input_vars]
        inferred = TypeRules.infer_operator_type(op_name, input_types)
        
        if inferred is None:
            return input_types[0] if input_types else 'float'
        
        return inferred

    def compile(self) -> str:
        """Compile the compute graph to shader code"""
        if self.shader_type == ShaderType.COMPUTE:
            return self._compile_compute()
        else:
            return self._compile_graphics()

    def _compile_compute(self) -> str:
        """Compile compute shader"""
        layout = getattr(self.graph, 'layout', (64, 1, 1))
        
        buffers = []
        assignments = []
        uniforms = []
        buffer_count = 0
        
        # Analyze variable scopes first
        self._analyze_variable_scopes(self.graph.steps, 0)
        self._compute_hoist_requirements()
        
        # Collect function dependencies
        self._collect_function_dependencies(self.graph.steps)
        
        # Generate function declarations
        function_declarations = self._get_function_declarations()
        
        # Process all variables in the graph based on their storage hints
        for var_name in sorted(self.graph.input_vars.union(self.graph.uniform_vars)):
            storage = self.graph.storage_hints.get(var_name, IOTypes.buffer)
            var_type = self.input_types.get(var_name, 'float')
            
            if storage == IOTypes.uniform:
                # Uniform storage - simple uniform variable
                uniforms.append(f"uniform {var_type} {var_name};")
                self.declared_vars.add(var_name)
                # Store type for uniforms
                self.var_types[var_name] = var_type
            elif storage == IOTypes.array:
                # Array storage - declare as buffer, accessible directly as array
                buffers.append(f"layout(std430, binding = {buffer_count}) buffer D{buffer_count} {{ {var_type} {var_name}[]; }};")
                # For array storage, we don't create a per-thread variable
                # The array is accessed directly with indices
                self.declared_vars.add(var_name)
                # Store type WITH array notation so type inference works correctly
                self.var_types[var_name] = f"{var_type}[]"
                buffer_count += 1
            elif storage == IOTypes.buffer:
                # Buffer storage - element-wise access with _val suffix
                buffers.append(f"layout(std430, binding = {buffer_count}) buffer D{buffer_count} {{ {var_type} {var_name}_val[]; }};")
                # Create a per-thread variable for buffer access
                assignments.append(f"\t{var_type} {var_name} = {var_name}_val[gid];")
                self.declared_vars.add(var_name)
                # Store element type (not array) for buffer storage
                self.var_types[var_name] = var_type
                buffer_count += 1
            else:
                # Default to buffer
                buffers.append(f"layout(std430, binding = {buffer_count}) buffer D{buffer_count} {{ {var_type} {var_name}_val[]; }};")
                assignments.append(f"\t{var_type} {var_name} = {var_name}_val[gid];")
                self.declared_vars.add(var_name)
                self.var_types[var_name] = var_type
                buffer_count += 1
        
        # Add custom uniforms from function attributes
        for name, info in self.custom_uniforms.items():
            if isinstance(info, tuple):
                glsl_type, access = info
            else:
                glsl_type, access = info, 'readonly'
            uniforms.append(f"uniform {glsl_type} {name};")
        
        # Add custom textures
        for name, info in self.custom_textures.items():
            if isinstance(info, tuple):
                glsl_type, access = info
            else:
                glsl_type, access = info, 'readonly'
            uniforms.append(f"uniform {glsl_type} {name};")
        
        # Add static constants
        for name, glsl_type, size, values in self.graph.static_constants:
            # Format values based on type
            if glsl_type == 'vec2':
                # vec2 needs vec2(x, y) format
                formatted_values = [f"vec2{v}" for v in values]
                value_str = ', '.join(formatted_values)
            else:
                value_str = ', '.join(str(v) for v in values)
            uniforms.append(f"const {glsl_type} {name}[{size}] = {glsl_type}[]({value_str});")
            
        has_void_return = self.graph.has_void_return
        output_var = self.graph.output_var
        
        # Add output buffer if needed
        if not has_void_return and output_var:
            res_type = self._get_var_type(output_var)
            buffers.append(f"layout(std430, binding = {buffer_count}) buffer DR {{ {res_type} results[]; }};")
        
        # Track all variables declared in function scope
        function_scope_vars = set()
        
        # Declare hoisted variables at function scope
        for var_name in sorted(self.vars_to_hoist):
            var_type = self._get_var_type(var_name)
            default_init = {
                'int': '0',
                'uint': '0u',
                'float': '0.0',
                'bool': 'false',
                'vec2': 'vec2(0.0)',
                'vec3': 'vec3(0.0)',
                'vec4': 'vec4(0.0)',
                'ivec2': 'ivec2(0)',
                'ivec3': 'ivec3(0)',
                'ivec4': 'ivec4(0)',
                'uvec2': 'uvec2(0u)',
                'uvec3': 'uvec3(0u)',
                'uvec4': 'uvec4(0u)',
            }
            init_value = default_init.get(var_type, '0.0')
            
            assignments.append(f"\t{var_type} {var_name} = {init_value};")
            self.var_types[var_name] = var_type
            self.declared_vars.add(var_name)
            function_scope_vars.add(var_name)
        
        # Now process the graph steps to generate code
        for step in self.graph.steps:
            if step['type'] == 'operation':
                self._process_operation(step['op_name'], step['inputs'], 
                                    step.get('output_var'), assignments, 1, function_scope_vars)
            elif step['type'] == 'loop':
                self._process_loop_step(step, assignments, 1, function_scope_vars)
            elif step['type'] == 'if':
                self._process_loop_step(step, assignments, 1, function_scope_vars)
        
        # Add output assignment if needed
        if not has_void_return and output_var:
            assignments.append(f"\tresults[gid] = {output_var};")
        
        # Build the shader
        shader_parts = [get_standard_heading(layout)]
        
        # Check if nItems is already in uniforms from the graph
        has_nitems_uniform = False
        for var_name in sorted(self.graph.input_vars.union(self.graph.uniform_vars)):
            if var_name == 'nItems':
                has_nitems_uniform = True
                break

        # If nItems is already in the standard heading and also a kernel uniform,
        # we need to adjust the heading to not include it
        if has_nitems_uniform:
            # Re-generate heading without nItems
            shader_parts = [get_standard_heading(layout, include_n_items=False)]

        if function_declarations:
            shader_parts.append("\n// Function declarations")
            shader_parts.extend(function_declarations)
            shader_parts.append("")
        
        if uniforms:
            shader_parts.extend(uniforms)
            shader_parts.append("")
        
        if buffers:
            shader_parts.extend(buffers)
            shader_parts.append("")
        
        main_body = "\n".join(assignments) if assignments else "\t// No operations"
        
        # Only add the gid >= nItems check for vectorized kernels
        # If vectorized is None (default/legacy) or True, add the check
        # If vectorized is False, skip the check (user manages thread IDs manually)
        if self.graph.vectorized is False:
            # No vectorization check - user manages thread execution
            shader_parts.append(f"""
    void main() {{
    {main_body}
    }}""")
        else:
            # Vectorized execution - add automatic bounds checking
            shader_parts.append(f"""
    void main() {{
    \tuint gid = gl_GlobalInvocationID.x;
    \tif(gid >= nItems) return;
    {main_body}
    }}""")
        
        return "\n".join(shader_parts)


    def _compile_graphics(self) -> str:
        """Compile fragment, vertex, or geometry shader"""
        version = get_shader_version(self.shader_type)
        
        self._collect_function_dependencies(self.graph.steps)
        function_declarations = self._get_function_declarations()
        
        self._collect_builtin_declarations()
        self._add_custom_declarations()
        
        inputs_code = [f"in {t} {n};" for t, n, a in self.inputs]
        outputs_code = [f"out {t} {n};" for t, n, a in self.outputs]
        uniforms_code = [f"uniform {t} {n};" for t, n, a in self.uniforms]
        textures_code = [f"uniform {t} {n};" for t, n, a in self.textures]
        
        if not outputs_code and self.shader_type == ShaderType.FRAGMENT:
            outputs_code = ShaderInputManager.get_default_outputs(self.shader_type)
        
        shader_body = []
        for step in self.graph.steps:
            assignments = []
            if step['type'] == 'operation':
                self._process_operation(step['op_name'], step['inputs'],
                                    step.get('output_var'), assignments, 1, set())
            elif step['type'] == 'loop':
                self._process_loop_step(step, assignments, 1, set())
            elif step['type'] == 'if':
                self._process_loop_step(step, assignments, 1, set())
            shader_body.extend(assignments)
        
        all_declarations = []
        
        if function_declarations:
            all_declarations.append("// Function declarations")
            all_declarations.extend(function_declarations)
            all_declarations.append("")
        
        if inputs_code:
            all_declarations.append("// Inputs")
            all_declarations.extend(inputs_code)
        if uniforms_code:
            all_declarations.append("// Uniforms")
            all_declarations.extend(uniforms_code)
        if textures_code:
            all_declarations.append("// Textures")
            all_declarations.extend(textures_code)
        if outputs_code:
            all_declarations.append("// Outputs")
            all_declarations.extend(outputs_code)
        
        declarations = '\n'.join(all_declarations) if all_declarations else "// No declarations"
        
        template = SHADER_TEMPLATES.get(self.shader_type, 
                                    SHADER_TEMPLATES[ShaderType.FRAGMENT])
        
        code = template.format(
            version=version,
            declarations=declarations,
            body='\n'.join(shader_body) if shader_body else "\t// Shader body"
        )
        
        return code

    def _collect_builtin_declarations(self):
        """Collect and auto-declare used built-in variables"""
        for var_name in self.used_builtins:
            var_info = ShaderInputManager.get_builtin_info(var_name, self.shader_type)
            if var_info:
                if var_info['storage'] == 'in':
                    self.inputs.append((var_info['glsl_type'], var_info['name'], 
                                    var_info['access']))
                elif var_info['storage'] == 'out':
                    self.outputs.append((var_info['glsl_type'], var_info['name'], 
                                    var_info['access']))
                elif var_info['storage'] == 'uniform':
                    if var_info['glsl_type'].startswith('sampler'):
                        self.textures.append((var_info['glsl_type'], var_info['name'], 
                                            var_info['access']))
                    else:
                        self.uniforms.append((var_info['glsl_type'], var_info['name'], 
                                            var_info['access']))

    def _add_custom_declarations(self):
        """Add custom uniforms and textures from decorator"""
        for name, info in self.custom_uniforms.items():
            if isinstance(info, tuple):
                glsl_type, access = info
            else:
                glsl_type, access = info, 'readonly'
            self.uniforms.append((glsl_type, name, access))
        
        for name, info in self.custom_textures.items():
            if isinstance(info, tuple):
                glsl_type, access = info
            else:
                glsl_type, access = info, 'readonly'
            self.textures.append((glsl_type, name, access))

    def _process_loop_step(self, step, assignments, indent_level, function_scope_vars):
        """Process a loop step"""
        indent = "\t" * indent_level
        loop_info = step.get('loop_info', step)
        
        if loop_info.get('type') == 'if':
            condition = loop_info['condition']
            assignments.append(f"{indent}if({condition}) {{")
            
            body_indent = indent_level + 1
            then_body = step.get('then_body', [])
            if not then_body and 'then_body' in loop_info:
                then_body = loop_info['then_body']
            
            for body_step in then_body:
                if isinstance(body_step, dict):
                    if body_step['type'] == 'operation':
                        self._process_operation(body_step['op_name'], body_step['inputs'],
                                            body_step.get('output_var'), assignments, body_indent, function_scope_vars)
                    elif body_step['type'] in ('loop', 'if'):
                        self._process_loop_step(body_step, assignments, body_indent, function_scope_vars)
            
            assignments.append(f"{indent}}}")
            
            else_body = step.get('else_body', [])
            if not else_body and 'else_body' in loop_info and loop_info['else_body']:
                else_body = loop_info['else_body']
            
            if else_body:
                assignments.append(f"{indent}else {{")
                
                for body_step in else_body:
                    if isinstance(body_step, dict):
                        if body_step['type'] == 'operation':
                            self._process_operation(body_step['op_name'], body_step['inputs'],
                                                body_step.get('output_var'), assignments, body_indent, function_scope_vars)
                        elif body_step['type'] in ('loop', 'if'):
                            self._process_loop_step(body_step, assignments, body_indent, function_scope_vars)
                
                assignments.append(f"{indent}}}")
            
            return
        
        elif loop_info.get('type') == 'for':
            loop_var = loop_info['var']
            start = loop_info['start']
            end = loop_info['end']
            step_val = loop_info.get('step', '1')
            is_dynamic = loop_info.get('dynamic', False)
            
            if is_dynamic:
                start_expr = f"int({start})" if not self._is_literal(start) and not start.endswith('u') else start
                end_expr = f"int({end})" if not self._is_literal(end) and not end.endswith('u') else end
                step_expr = f"int({step_val})" if not self._is_literal(step_val) and not step_val.endswith('u') else step_val
            else:
                start_expr = start
                end_expr = end
                step_expr = step_val
            
            assignments.append(f"{indent}for(int {loop_var} = {start_expr}; {loop_var} < {end_expr}; {loop_var} += {step_expr}) {{")
            
            self.var_types[loop_var] = 'int'
            self.declared_vars.add(loop_var)
            
            body_indent = indent_level + 1
            for body_step in step.get('body', []):
                if isinstance(body_step, dict):
                    if body_step['type'] == 'operation':
                        self._process_operation(body_step['op_name'], body_step['inputs'],
                                            body_step.get('output_var'), assignments, body_indent, function_scope_vars)
                    elif body_step['type'] in ('loop', 'if'):
                        self._process_loop_step(body_step, assignments, body_indent, function_scope_vars)
            
            assignments.append(f"{indent}}}")
        
        elif loop_info.get('type') == 'while':
            test = loop_info['test']
            assignments.append(f"{indent}while({test}) {{")
            
            body_indent = indent_level + 1
            for body_step in step.get('body', []):
                if isinstance(body_step, dict):
                    if body_step['type'] == 'operation':
                        self._process_operation(body_step['op_name'], body_step['inputs'],
                                            body_step.get('output_var'), assignments, body_indent, function_scope_vars)
            
            assignments.append(f"{indent}}}")

    def _process_operation(self, op_name: str, inputs: List[str], output_var: str, 
                        assignments: List[str], indent_level: int, function_scope_vars: set):
        """Process a single operation"""
        indent = "\t" * indent_level
        
        # Handle return statements first
        if op_name == 'return':
            assignments.append(f"{indent}return;")
            return
        
        # Handle assignment to built-in variables
        if op_name == 'assign_builtin':
            if len(inputs) >= 2:
                value_var = inputs[0]
                builtin_var = inputs[1]
                assignments.append(f"{indent}{builtin_var} = {value_var};")
                self.declared_vars.add(builtin_var)
            return
        
        # Handle array assignments
        if op_name in ['subscript_assign', 'subscript_assign_2d']:
            if op_name == 'subscript_assign':
                array_var, index_var, value_var = inputs
                storage = self.graph.storage_hints.get(array_var, IOTypes.buffer)
                if storage == IOTypes.array:
                    index_type = self._get_var_type(index_var)
                    if index_type != 'int':
                        index_var = f"int({index_var})"
                    assignments.append(f"{indent}{array_var}[{index_var}] = {value_var};")
                else:
                    index_type = self._get_var_type(index_var)
                    if index_type != 'int':
                        index_var = f"int({index_var})"
                    assignments.append(f"{indent}{array_var}[{index_var}] = {value_var};")
            else:
                array_var, index1_var, index2_var, value_var = inputs
                storage = self.graph.storage_hints.get(array_var, IOTypes.buffer)
                if storage == IOTypes.array:
                    index1_type = self._get_var_type(index1_var)
                    index2_type = self._get_var_type(index2_var)
                    if index1_type != 'int':
                        index1_var = f"int({index1_var})"
                    if index2_type != 'int':
                        index2_var = f"int({index2_var})"
                    assignments.append(f"{indent}{array_var}[{index1_var}][{index2_var}] = {value_var};")
                else:
                    index1_type = self._get_var_type(index1_var)
                    index2_type = self._get_var_type(index2_var)
                    if index1_type != 'int':
                        index1_var = f"int({index1_var})"
                    if index2_type != 'int':
                        index2_var = f"int({index2_var})"
                    assignments.append(f"{indent}{array_var}[{index1_var}][{index2_var}] = {value_var};")
            return
        
        # Handle expression results (standalone expressions)
        if op_name == 'expression_result':
            if inputs:
                assignments.append(f"{indent}{inputs[0]};")
            return
        
        # Handle array declarations
        if op_name == 'declare_array':
            if output_var and len(inputs) >= 1:
                array_type = inputs[0]
                # Declare the array variable
                assignments.append(f"{indent}{array_type} {output_var};")
                self.declared_vars.add(output_var)
            return
        
        # Handle variable assignments
        if op_name == 'assign':
            if len(inputs) >= 1 and output_var:
                value_var = inputs[0]
                target_var = output_var
                
                # Make sure the target variable is properly handled
                if target_var not in self.declared_vars and not self._is_literal(target_var):
                    # Check if it's in function scope vars
                    if target_var in function_scope_vars:
                        # Already declared at function scope, just assign
                        assignments.append(f"{indent}{target_var} = {value_var};")
                    else:
                        # Declare the variable with its type
                        target_type = self._get_var_type(target_var)
                        if target_type:
                            assignments.append(f"{indent}{target_type} {target_var} = {value_var};")
                        else:
                            assignments.append(f"{indent}float {target_var} = {value_var};")
                        self.declared_vars.add(target_var)
                else:
                    assignments.append(f"{indent}{target_var} = {value_var};")
            return
        
        # Handle array indexing
        if op_name == 'subscript':
            if len(inputs) >= 2:
                array_var = inputs[0]
                index_var = inputs[1]
                storage = self.graph.storage_hints.get(array_var, IOTypes.buffer)
                
                # Always cast index to int for array access
                index_type = self._get_var_type(index_var)
                if index_type != 'int':
                    index_var = f"int({index_var})"
                expr = f"{array_var}[{index_var}]"
            else:
                expr = get_op_glsl(op_name, inputs)
        
        elif op_name == 'subscript_2d':
            if len(inputs) >= 3:
                array_var = inputs[0]
                index1_var = inputs[1]
                index2_var = inputs[2]
                storage = self.graph.storage_hints.get(array_var, IOTypes.buffer)
                
                if storage == IOTypes.array:
                    index1_type = self._get_var_type(index1_var)
                    index2_type = self._get_var_type(index2_var)
                    if index1_type != 'int':
                        index1_var = f"int({index1_var})"
                    if index2_type != 'int':
                        index2_var = f"int({index2_var})"
                    expr = f"{array_var}[{index1_var}][{index2_var}]"
                else:
                    index1_type = self._get_var_type(index1_var)
                    index2_type = self._get_var_type(index2_var)
                    if index1_type != 'int':
                        index1_var = f"int({index1_var})"
                    if index2_type != 'int':
                        index2_var = f"int({index2_var})"
                    expr = f"{array_var}[{index1_var}][{index2_var}]"
            else:
                expr = get_op_glsl(op_name, inputs)
        
        else:
            # Generate the expression for other operations
            if op_name in TypeRules.BITWISE_OPS:
                # For bitwise operations, all operands must be the same integer type
                # Determine the target type (prefer uint for mixed int/uint)
                input_types = [self._get_var_type(inp) for inp in inputs]
                has_uint = any(t in ['uint', 'uvec2', 'uvec3', 'uvec4'] for t in input_types)
                target_type = 'uint' if has_uint else 'int'
                
                casted_inputs = []
                for inp in inputs:
                    inp_type = self._get_var_type(inp)
                    if inp_type not in ['int', 'uint', 'ivec2', 'ivec3', 'ivec4', 'uvec2', 'uvec3', 'uvec4']:
                        # Cast non-integer types to target type
                        casted_inputs.append(f"{target_type}({inp})")
                    elif inp_type in ['int', 'ivec2', 'ivec3', 'ivec4'] and target_type == 'uint':
                        # Cast int to uint
                        casted_inputs.append(f"uint({inp})")
                    elif inp_type in ['uint', 'uvec2', 'uvec3', 'uvec4'] and target_type == 'int':
                        # Cast uint to int
                        casted_inputs.append(f"int({inp})")
                    else:
                        casted_inputs.append(inp)
                expr = get_op_glsl(op_name, casted_inputs)
            else:
                expr = get_op_glsl(op_name, inputs)
        
        # Handle special cases for output_var
        if output_var == GLTypes.void:
            assignments.append(f"{indent}{expr};")
        elif output_var and output_var.startswith('_t'):
            # Temporary variable
            out_type = self.infer_type(op_name, inputs)
            self.var_types[output_var] = out_type
            
            # Check if this variable needs to be hoisted
            if output_var in self.vars_to_hoist:
                assignments.append(f"{indent}{output_var} = {expr};")
            elif output_var not in self.declared_vars:
                assignments.append(f"{indent}{out_type} {output_var} = {expr};")
                self.declared_vars.add(output_var)
            else:
                assignments.append(f"{indent}{output_var} = {expr};")
        elif output_var and (output_var in self.var_map or output_var in self.declared_vars):
            # Existing variable assignment
            assignments.append(f"{indent}{output_var} = {expr};")
        elif output_var:
            # New variable declaration
            out_type = self.infer_type(op_name, inputs)
            self.var_types[output_var] = out_type
            assignments.append(f"{indent}{out_type} {output_var} = {expr};")
            self.declared_vars.add(output_var)