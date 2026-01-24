from typing import Dict, List
from .composition import get_standard_heading, _buff_line
from .gl_typing import GLTypes, NP_GLTypes, GLSL_TO_NP, TypeRules, BUILTIN_VARIABLES
from .compute_graph import ComputeGraph

class ShaderCompiler:
    
    OP_TO_GLSL = {
        'add': lambda inputs: f"({' + '.join(inputs)})",
        'sub': lambda inputs: f"({inputs[0]} - {inputs[1]})",
        'mult': lambda inputs: f"({' * '.join(inputs)})",  # Removed self reference
        'div': lambda inputs: f"({inputs[0]} / {inputs[1]})",
        'floordiv': lambda inputs: f"({inputs[0]} / {inputs[1]})",
        'neg': lambda inputs: f"(-{inputs[0]})",
        'square': lambda inputs: f"({inputs[0]} * {inputs[0]})",
        'gt': lambda inputs: f"({inputs[0]} > {inputs[1]})",
        'lt': lambda inputs: f"({inputs[0]} < {inputs[1]})",
        'eq': lambda inputs: f"({inputs[0]} == {inputs[1]})",
        'gte': lambda inputs: f"({inputs[0]} >= {inputs[1]})",
        'lte': lambda inputs: f"({inputs[0]} <= {inputs[1]})",
        'neq': lambda inputs: f"({inputs[0]} != {inputs[1]})",
        'and': lambda inputs: f"({inputs[0]} & {inputs[1]})",
        'or': lambda inputs: f"({inputs[0]} | {inputs[1]})",
        'xor': lambda inputs: f"({inputs[0]} ^ {inputs[1]})",
        'mod': lambda inputs: f"({inputs[0]} % {inputs[1]})",
        'clamp': lambda inputs: f"clamp({inputs[0]}, {inputs[1]}, {inputs[2]})",
        'avg': lambda inputs: f"(({' + '.join(inputs)}) / {len(inputs)}.0)",
        'bool': lambda inputs: f"({inputs[0]} != 0)",
        'is_zero': lambda inputs: f"({inputs[0]} == 0)",
        'lsh': lambda inputs: f"({inputs[0]} << {inputs[1]})",
        'rsh': lambda inputs: f"({inputs[0]} >> {inputs[1]})",
        'abs': lambda inputs: f"abs({inputs[0]})",
        'sqrt': lambda inputs: f"sqrt({inputs[0]})",
        'sin': lambda inputs: f"sin({inputs[0]})",
        'cos': lambda inputs: f"cos({inputs[0]})",
        'tan': lambda inputs: f"tan({inputs[0]})",
        'asin': lambda inputs: f"asin({inputs[0]})",
        'acos': lambda inputs: f"acos({inputs[0]})",
        'atan': lambda inputs: f"atan({', '.join(inputs)})",
        'exp': lambda inputs: f"exp({inputs[0]})",
        'log': lambda inputs: f"log({inputs[0]})",
        'pow': lambda inputs: f"pow({inputs[0]}, {inputs[1]})",
        'min': lambda inputs: f"min({inputs[0]}, {inputs[1]})",
        'max': lambda inputs: f"max({inputs[0]}, {inputs[1]})",
        'mix': lambda inputs: f"mix({inputs[0]}, {inputs[1]}, {inputs[2]})",
        'step': lambda inputs: f"step({inputs[0]}, {inputs[1]})",
        'smoothstep': lambda inputs: f"smoothstep({inputs[0]}, {inputs[1]}, {inputs[2]})",
        'cast': lambda inputs: f"{inputs[1]}({inputs[0]})",
        'bool_not': lambda inputs: f"(!{inputs[0]})",
        'bitwise_not': lambda inputs: f"(~{inputs[0]})",
        'bitwise_and': lambda inputs: f"({inputs[0]} & {inputs[1]})",
        'bitwise_or': lambda inputs: f"({inputs[0]} | {inputs[1]})",
        'bitwise_xor': lambda inputs: f"({inputs[0]} ^ {inputs[1]})",
        'bitshift_left': lambda inputs: f"({inputs[0]} << {inputs[1]})",
        'bitshift_right': lambda inputs: f"({inputs[0]} >> {inputs[1]})",
        'sign': lambda inputs: f"sign({inputs[0]})",
        'floor': lambda inputs: f"floor({inputs[0]})",
        'ceil': lambda inputs: f"ceil({inputs[0]})",
        'fract': lambda inputs: f"fract({inputs[0]})",
        'round': lambda inputs: f"round({inputs[0]})",
        'subscript': lambda inputs: f"{inputs[0]}[int({inputs[1]})]",  # Fixed
        'subscript_2d': lambda inputs: f"{inputs[0]}[int({inputs[1]})][int({inputs[2]})]",  # Fixed
        'swizzle': lambda inputs: f"{inputs[0]}.{inputs[1]}",
        'subscript_assign': lambda inputs: f"{inputs[0]}[int({inputs[1]})] = {inputs[2]}",
        'subscript_assign_2d': lambda inputs: f"{inputs[0]}[int({inputs[1]})][int({inputs[2]})] = {inputs[3]}",
        'array_decl': lambda inputs: None,  # Just declaration, no expression
        'array_init': lambda inputs: None,  # Initialization handled separately
        'array_init_expr': lambda inputs: None,  # Array fill with expression
        'cmul': lambda inputs: f"vec2({inputs[0]}.x * {inputs[1]}.x - {inputs[0]}.y * {inputs[1]}.y, {inputs[0]}.x * {inputs[1]}.y + {inputs[0]}.y * {inputs[1]}.x)",
        'vec2': lambda inputs: f"vec2({', '.join(inputs)})" if len(inputs) == 2 else f"vec2({inputs[0]})",
        'vec3': lambda inputs: f"vec3({', '.join(inputs)})" if len(inputs) == 3 else f"vec3({inputs[0]})",
        'vec4': lambda inputs: f"vec4({', '.join(inputs)})" if len(inputs) == 4 else f"vec4({inputs[0]})",
        'uvec2': lambda inputs: f"uvec2({', '.join(inputs)})" if len(inputs) == 2 else f"uvec2({inputs[0]})",
        'uvec3': lambda inputs: f"uvec3({', '.join(inputs)})" if len(inputs) == 3 else f"uvec3({inputs[0]})",
        'uvec4': lambda inputs: f"uvec4({', '.join(inputs)})" if len(inputs) == 4 else f"uvec4({inputs[0]})",
        'ivec2': lambda inputs: f"ivec2({', '.join(inputs)})" if len(inputs) == 2 else f"ivec2({inputs[0]})",
        'ivec3': lambda inputs: f"ivec3({', '.join(inputs)})" if len(inputs) == 3 else f"ivec3({inputs[0]})",
        'ivec4': lambda inputs: f"ivec4({', '.join(inputs)})" if len(inputs) == 4 else f"ivec4({inputs[0]})",
    }
    
    def __init__(self, graph: ComputeGraph, input_types: Dict[str, str], explicit_types: Dict[str, str] = None):
        self.graph = graph
        self.input_types = input_types
        self.var_types = input_types.copy()
        self.explicit_types = explicit_types or {}
        self.temporary_types = {}
        self.declared_vars = set()
        self.declared_vars.update(graph.input_vars)
        self.declared_vars.update(graph.uniform_vars)
        # Mark built-in variables as pre-declared
        for builtin_name in BUILTIN_VARIABLES:
            self.declared_vars.add(builtin_name)
            self.var_types[builtin_name] = BUILTIN_VARIABLES[builtin_name]
        
        self.temp_to_final_map = {}
        self.operation_expressions = {}
        
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
                    val = int(var_name)
                    # Default to int for integer literals, not uint
                    return 'int'
                except ValueError:
                    return 'float'
        return 'float'

    def promote_types(self, type1: str, type2: str) -> str:
        return TypeRules.promote_types(type1, type2)

    def infer_type(self, op_name: str, input_vars: List[str]) -> str:
        input_types = [self._get_var_type(v) for v in input_vars]
        inferred = TypeRules.infer_operator_type(op_name, input_types)
        
        # Fix for vector array subscripts - when subscripting a vec2/vec3/vec4 array,
        # the result should be the vector type, not float
        if op_name in ['subscript', 'subscript_2d']:
            if input_vars and not TypeRules.is_array_type(input_types[0]):
                base_type = input_types[0]
                if base_type.startswith(('vec', 'uvec', 'ivec')):
                    return base_type
        
        return inferred
    
    def _optimize_direct_assignments(self, steps):
        """Optimize steps by eliminating unnecessary temporary variables"""
        var_replacements = {}
        
        # Pass 1: Collect replacements recursively from all scopes
        def collect_replacements(steps_list):
            for step in steps_list:
                if step['type'] == 'operation' and step['op_name'] == 'direct_assign':
                    # ONLY optimize direct_assign, NEVER subscript_assign or subscript_assign_2d
                    temp_var = step['inputs'][0]
                    final_var = step['inputs'][1]
                    var_replacements[temp_var] = final_var
                    
                    if self.graph.output_var == temp_var:
                        self.graph.output_var = final_var
                
                elif step['type'] == 'loop' and 'body' in step:
                    collect_replacements(step['body'])
        
        collect_replacements(steps)
        
        # Pass 2: Filter out direct_assign ops and apply replacements recursively
        def filter_and_replace(steps_list):
            new_steps = []
            for step in steps_list:
                if step['type'] == 'operation':
                    if step['op_name'] == 'direct_assign':
                        continue  # Remove this operation entirely
                    
                    # Create new step with replaced variables
                    new_step = step.copy()
                    
                    # Replace inputs
                    new_step['inputs'] = [var_replacements.get(v, v) for v in step['inputs']]
                    
                    # Replace output variable
                    if new_step['output_var'] in var_replacements:
                        new_step['output_var'] = var_replacements[new_step['output_var']]
                    
                    new_steps.append(new_step)
                    
                elif step['type'] == 'loop':
                    # Handle loop/if/while structures
                    new_step = step.copy()
                    
                    # Recursively process the body
                    if 'body' in new_step:
                        new_step['body'] = filter_and_replace(new_step['body'])
                    
                    # Also apply replacements to loop control variables/conditions
                    if 'loop_info' in new_step:
                        new_info = new_step['loop_info'].copy()
                        keys_to_check = ['start', 'end', 'step', 'test', 'condition']
                        for key in keys_to_check:
                            if key in new_info:
                                val = new_info[key]
                                if isinstance(val, str) and val in var_replacements:
                                    new_info[key] = var_replacements[val]
                        new_step['loop_info'] = new_info
                        
                    new_steps.append(new_step)
            
            return new_steps
            
        return filter_and_replace(steps)

    def compile(self):
        # First, optimize the graph to eliminate unnecessary temporaries
        self.graph.steps = self._optimize_direct_assignments(self.graph.steps)
        
        # Generate static constant declarations
        static_constant_defs = []
        for name, glsl_type, size, values in self.graph.static_constants:
            # Format values
            if glsl_type == 'vec2':
                formatted_values = [f"vec2({v[0]}, {v[1]})" for v in values]
            elif glsl_type == 'vec3':
                formatted_values = [f"vec3({v[0]}, {v[1]}, {v[2]})" for v in values]
            elif glsl_type == 'vec4':
                formatted_values = [f"vec4({v[0]}, {v[1]}, {v[2]}, {v[3]})" for v in values]
            else:
                formatted_values = [str(v) for v in values]
            
            values_str = ", ".join(formatted_values)
            const_def = f"const {glsl_type} {name}[{size}] = {glsl_type}[{size}]({values_str});"
            static_constant_defs.append(const_def)
        
        glsl_function_defs = []
        for func_info in self.graph.glsl_functions:
            name = func_info['name']
            params = func_info['params']
            return_type = func_info['return_type']
            body_ops = func_info['body_ops']
            
            param_str = ', '.join([f"{ptype} {pname}" for pname, ptype in params])
            func_header = f"{return_type} {name}({param_str}) {{"
            
            func_assignments = []
            for op_name, inputs, output_var in body_ops:
                if op_name == 'return':
                    if inputs:
                        func_assignments.append(f"\n\treturn {inputs[0]};")
                    else:
                        func_assignments.append(f"\n\treturn;")
                elif op_name == 'assign':
                    target_var = inputs[0]
                    value_var = inputs[1]
                    value_type = self._get_var_type(value_var)
                    
                    if target_var not in self.declared_vars:
                        func_assignments.append(f"\n\t{value_type} {target_var} = {value_var};")
                        self.declared_vars.add(target_var)
                    else:
                        func_assignments.append(f"\n\t{target_var} = {value_var};")
                else:
                    out_type = self.infer_type(op_name, inputs)
                    self.var_types[output_var] = out_type
                    expr = self.OP_TO_GLSL.get(op_name, lambda x: f"{op_name}({', '.join(x)})")(inputs)
                    
                    if output_var not in self.declared_vars:
                        func_assignments.append(f"\n\t{out_type} {output_var} = {expr};")
                        self.declared_vars.add(output_var)
                    else:
                        func_assignments.append(f"\n\t{output_var} = {expr};")
            
            func_body = ''.join(func_assignments)
            func_def = f"{func_header}{func_body}\n}}"
            glsl_function_defs.append(func_def)
        
        buffer_inputs = []
        uniforms = []
        buffers = []
        assignments = []
        
        # Determine if this is a vectorized kernel
        has_vectorized_buffers = False
        
        buffer_count = 0
        for var in sorted(self.graph.input_vars):
            var_type = self.input_types.get(var, 'float')
            storage = self.graph.storage_hints.get(var, 'buffer')
            self.var_types[var] = var_type
            
            if storage == 'array':
                # Non-indexed buffer - no automatic [gid] access
                buffer_inputs.append((buffer_count, var, var_type))
                buffers.append(_buff_line(buffer_count, f"D{buffer_count}", var, var_type))
                self.declared_vars.add(var)  # Mark as declared (it's the buffer name itself)
                buffer_count += 1
            else:
                # Vectorized buffer - automatic [gid] access
                has_vectorized_buffers = True
                buffer_inputs.append((buffer_count, var, var_type))
                buffers.append(_buff_line(buffer_count, f"D{buffer_count}", f"data_{buffer_count}", var_type))
                assignments.append(f"\n\t{var_type} {var} = data_{buffer_count}[gid];")
                buffer_count += 1
        
        # Check if nItems is already in uniforms
        has_n_items_uniform = 'nItems' in self.graph.uniform_vars
        
        for var in sorted(self.graph.uniform_vars):
            var_type = self.input_types.get(var, 'float')
            self.var_types[var] = var_type
            uniforms.append(f"uniform {var_type} {var};")
        
        current_assignments = []
        indent_level = 1
        
        for step in self.graph.steps:
            if step['type'] == 'loop':
                self._process_loop_step(step, current_assignments, indent_level)
            elif step['type'] == 'if':
                self._process_if_step(step, current_assignments, indent_level)
            else:
                op_name = step['op_name']
                inputs = step['inputs']
                output_var = step['output_var']
                self._process_operation(op_name, inputs, output_var, current_assignments, indent_level)
        
        assignments.extend(current_assignments)
        
        if not self.graph.has_void_return and self.graph.output_var:
            result_type = self.var_types.get(self.graph.output_var, 'float')
            result_binding = len(buffer_inputs)
            buffers.append("\n" + _buff_line(result_binding, "DR", "results", result_type))
            assignments.append(f"\n\tresults[gid] = {self.graph.output_var};")
        
        functions_section = '\n'.join(glsl_function_defs) if glsl_function_defs else ''
        
        # Get layout from graph or use default
        layout = getattr(self.graph, 'layout', (64, 1, 1))
        
        # Only include nItems in header if:
        # 1. This is a vectorized kernel (has buffer inputs), AND
        # 2. nItems is not already declared as a uniform parameter
        include_n_items = has_vectorized_buffers and not has_n_items_uniform
        
        code_parts = [get_standard_heading(layout, include_n_items=include_n_items)]
        
        if static_constant_defs:
            code_parts.extend(static_constant_defs)
            code_parts.append("")
        
        if uniforms:
            code_parts.extend(uniforms)
        
        if buffers:
            code_parts.extend(buffers)
        
        if functions_section:
            code_parts.append(functions_section)
        
        code_parts.append(f"""
void main() {{
\tint gid = int(gl_GlobalInvocationID.x);
\tif(gid >= nItems) return;
\t{''.join(assignments)}
}}""")
        
        code = "\n".join(code_parts)
        
        if self.graph.has_void_return:
            result_type = 'void'
        elif self.graph.output_var:
            result_type = self.var_types.get(self.graph.output_var, 'float')
        else:
            result_type = 'void'
        
        return code, result_type
        
    def _process_if_step(self, step: Dict, assignments: List[str], indent_level: int):
        """Process an If statement step"""
        indent = "\t" * indent_level
        condition = step['condition']
        
        assignments.append(f"\n{indent}if({condition}) {{")
        
        # Process then body
        body_indent = indent_level + 1
        then_body = step.get('then_body', [])
        
        for body_step in then_body:
            if isinstance(body_step, dict):
                if body_step['type'] == 'operation':
                    op_name = body_step['op_name']
                    inputs = body_step['inputs']
                    output_var = body_step.get('output_var')
                    
                    # Special handling for return statements
                    if op_name == 'return':
                        if inputs:
                            assignments.append(f"\n{indent}\treturn {inputs[0]};")
                        else:
                            assignments.append(f"\n{indent}\treturn;")
                    elif op_name == 'assign':
                        target_var = inputs[0]
                        value_var = inputs[1]
                        assignments.append(f"\n{indent}\t{target_var} = {value_var};")
                    elif op_name == 'subscript_assign':
                        array_var = inputs[0]
                        index_var = inputs[1]
                        value_var = inputs[2]
                        assignments.append(f"\n{indent}\t{array_var}[{index_var}] = {value_var};")
                    elif op_name == 'subscript_assign_2d':
                        array_var = inputs[0]
                        index1_var = inputs[1]
                        index2_var = inputs[2]
                        value_var = inputs[3]
                        assignments.append(f"\n{indent}\t{array_var}[{index1_var}][{index2_var}] = {value_var};")
                    else:
                        self._process_operation(op_name, inputs, output_var, assignments, body_indent)
                elif body_step['type'] == 'loop':
                    self._process_loop_step(body_step, assignments, body_indent)
                elif body_step['type'] == 'if':
                    self._process_if_step(body_step, assignments, body_indent)
        
        assignments.append(f"\n{indent}}}")
        
        # Process else body if exists
        else_body = step.get('else_body')
        if else_body:
            assignments.append(f"\n{indent}else {{")
            
            for body_step in else_body:
                if isinstance(body_step, dict):
                    if body_step['type'] == 'operation':
                        op_name = body_step['op_name']
                        inputs = body_step['inputs']
                        output_var = body_step.get('output_var')
                        
                        if op_name == 'return':
                            if inputs:
                                assignments.append(f"\n{indent}\treturn {inputs[0]};")
                            else:
                                assignments.append(f"\n{indent}\treturn;")
                        elif op_name == 'assign':
                            target_var = inputs[0]
                            value_var = inputs[1]
                            assignments.append(f"\n{indent}\t{target_var} = {value_var};")
                        elif op_name == 'subscript_assign':
                            array_var = inputs[0]
                            index_var = inputs[1]
                            value_var = inputs[2]
                            assignments.append(f"\n{indent}\t{array_var}[{index_var}] = {value_var};")
                        elif op_name == 'subscript_assign_2d':
                            array_var = inputs[0]
                            index1_var = inputs[1]
                            index2_var = inputs[2]
                            value_var = inputs[3]
                            assignments.append(f"\n{indent}\t{array_var}[{index1_var}][{index2_var}] = {value_var};")
                        else:
                            self._process_operation(op_name, inputs, output_var, assignments, body_indent)
                    elif body_step['type'] == 'loop':
                        self._process_loop_step(body_step, assignments, body_indent)
                    elif body_step['type'] == 'if':
                        self._process_if_step(body_step, assignments, body_indent)
            
            assignments.append(f"\n{indent}}}")
    
    def _process_loop_step(self, step: Dict, assignments: List[str], indent_level: int):
        indent = "\t" * indent_level
        loop_info = step.get('loop_info', {})
        
        if loop_info.get('type') == 'if':
            condition = loop_info['condition']
            assignments.append(f"\n{indent}if({condition}) {{")
            
            # Process then body
            body_indent = indent_level + 1
            then_body = step.get('then_body', [])
            if not then_body and 'then_body' in loop_info:
                then_body = loop_info['then_body']
            
            for body_step in then_body:
                if isinstance(body_step, dict):
                    if body_step['type'] == 'operation':
                        op_name = body_step['op_name']
                        inputs = body_step['inputs']
                        output_var = body_step.get('output_var')
                        
                        # Check if this is an assignment operation within if block
                        if op_name == 'assign':
                            target_var = inputs[0]
                            value_var = inputs[1]
                            assignments.append(f"\n{indent}\t{target_var} = {value_var};")
                        elif op_name == 'subscript_assign':
                            array_var = inputs[0]
                            index_var = inputs[1]
                            value_var = inputs[2]
                            assignments.append(f"\n{indent}\t{array_var}[{index_var}] = {value_var};")
                        elif op_name == 'subscript_assign_2d':
                            array_var = inputs[0]
                            index1_var = inputs[1]
                            index2_var = inputs[2]
                            value_var = inputs[3]
                            assignments.append(f"\n{indent}\t{array_var}[{index1_var}][{index2_var}] = {value_var};")
                        elif op_name == 'expression_result':
                            # Just output the expression as a statement
                            if inputs:
                                assignments.append(f"\n{indent}\t{inputs[0]};")
                        else:
                            self._process_operation(op_name, inputs, output_var, assignments, body_indent)
                    elif body_step['type'] == 'loop':
                        self._process_loop_step(body_step, assignments, body_indent)
            
            assignments.append(f"\n{indent}}}")
            
            # Process else body if exists
            else_body = step.get('else_body', [])
            if not else_body and 'else_body' in loop_info and loop_info['else_body']:
                else_body = loop_info['else_body']
            
            if else_body:
                assignments.append(f"\n{indent}else {{")
                
                for body_step in else_body:
                    if isinstance(body_step, dict):
                        if body_step['type'] == 'operation':
                            op_name = body_step['op_name']
                            inputs = body_step['inputs']
                            output_var = body_step.get('output_var')
                            
                            if op_name == 'assign':
                                target_var = inputs[0]
                                value_var = inputs[1]
                                assignments.append(f"\n{indent}\t{target_var} = {value_var};")
                            elif op_name == 'subscript_assign':
                                array_var = inputs[0]
                                index_var = inputs[1]
                                value_var = inputs[2]
                                assignments.append(f"\n{indent}\t{array_var}[{index_var}] = {value_var};")
                            elif op_name == 'subscript_assign_2d':
                                array_var = inputs[0]
                                index1_var = inputs[1]
                                index2_var = inputs[2]
                                value_var = inputs[3]
                                assignments.append(f"\n{indent}\t{array_var}[{index1_var}][{index2_var}] = {value_var};")
                            elif op_name == 'expression_result':
                                if inputs:
                                    assignments.append(f"\n{indent}\t{inputs[0]};")
                            else:
                                self._process_operation(op_name, inputs, output_var, assignments, body_indent)
                        elif body_step['type'] == 'loop':
                            self._process_loop_step(body_step, assignments, body_indent)
                
                assignments.append(f"\n{indent}}}")
            
            return
        
        elif loop_info.get('type') == 'for':
            loop_var = loop_info['var']
            start = loop_info['start']
            end = loop_info['end']
            step_val = loop_info.get('step', '1')
            is_dynamic = loop_info.get('dynamic', False)
            
            # Handle dynamic bounds by ensuring proper type casting
            if is_dynamic:
                # For dynamic bounds, we need to ensure they're integers
                start_expr = f"int({start})" if not self._is_literal(start) and not start.endswith('u') else start
                end_expr = f"int({end})" if not self._is_literal(end) and not end.endswith('u') else end
                step_expr = f"int({step_val})" if not self._is_literal(step_val) and not step_val.endswith('u') else step_val
            else:
                start_expr = start
                end_expr = end
                step_expr = step_val
            
            assignments.append(f"\n{indent}for(int {loop_var} = {start_expr}; {loop_var} < {end_expr}; {loop_var} += {step_expr}) {{")
            
            # Track the loop variable type
            self.var_types[loop_var] = 'int'
            self.declared_vars.add(loop_var)
            
            body_indent = indent_level + 1
            for body_step in step.get('body', []):
                if body_step['type'] == 'operation':
                    op_name = body_step['op_name']
                    inputs = body_step['inputs']
                    output_var = body_step['output_var']
                    self._process_operation(op_name, inputs, output_var, assignments, body_indent)
                elif body_step['type'] == 'loop':
                    self._process_loop_step(body_step, assignments, body_indent)
            
            assignments.append(f"\n{indent}}}")
        elif loop_info.get('type') == 'while':
            test = loop_info['test']
            assignments.append(f"\n{indent}while({test}) {{")
            
            body_indent = indent_level + 1
            for body_step in step.get('body', []):
                if body_step['type'] == 'operation':
                    op_name = body_step['op_name']
                    inputs = body_step['inputs']
                    output_var = body_step['output_var']
                    self._process_operation(op_name, inputs, output_var, assignments, body_indent)
            
            assignments.append(f"\n{indent}}}")
    
    def _process_operation(
        self, op_name: str, inputs: List[str], output_var: str, 
        assignments: List[str], indent_level: int
    ):
        indent = "\t" * indent_level
        
        if op_name == 'array_decl':
            array_name = inputs[0]
            element_type = inputs[1]
            array_dims = inputs[2:]
            
            # Build type string with dimensions
            full_type = element_type
            for dim in array_dims:
                full_type += f"[{dim}]"
            
            self.var_types[array_name] = full_type
            self.explicit_types[array_name] = full_type
            
            if array_name not in self.declared_vars:
                # Generate declaration like: float temp_array[16][18];
                dim_str = ''.join(f"[{dim}]" for dim in array_dims)
                assignments.append(f"\n{indent}{element_type} {array_name}{dim_str};")
                self.declared_vars.add(array_name)
        
        elif op_name == 'array_init':
            array_name = inputs[0]
            element_type = inputs[1]
            array_dims = inputs[2:-len(inputs[2:])]  # All but the last elements are dimensions
            values = inputs[2 + len(array_dims):]  # Values come after dimensions
            
            full_type = element_type
            for dim in array_dims:
                full_type += f"[{dim}]"
            
            self.var_types[array_name] = full_type
            self.explicit_types[array_name] = full_type
            
            if array_name not in self.declared_vars:
                # For simplicity, we'll initialize with zeros for now
                # For complex initialization, we'd need to generate nested loops
                total_elements = 1
                for dim in array_dims:
                    if self._is_literal(dim):
                        total_elements *= int(float(dim))
                    else:
                        # Dynamic size - can't precompute
                        total_elements = -1
                        break
                
                if total_elements > 0 and total_elements == len(values):
                    # Static size with explicit values
                    values_str = ', '.join(values)
                    dim_str = ''.join(f"[{dim}]" for dim in array_dims)
                    assignments.append(f"\n{indent}{element_type} {array_name}{dim_str} = {element_type}{dim_str}({values_str});")
                else:
                    # Dynamic size or mismatched values - use loops
                    dim_str = ''.join(f"[{dim}]" for dim in array_dims)
                    assignments.append(f"\n{indent}{element_type} {array_name}{dim_str};")
                    # TODO: Add initialization loops
                
                self.declared_vars.add(array_name)
        
        elif op_name == 'array_init_expr':
            array_name = inputs[0]
            element_type = inputs[1]
            array_dims = inputs[2:-1]  # All but last two
            init_expr = inputs[-1]
            
            full_type = element_type
            for dim in array_dims:
                full_type += f"[{dim}]"
            
            self.var_types[array_name] = full_type
            self.explicit_types[array_name] = full_type
            
            if array_name not in self.declared_vars:
                # For filling array with expression, we need loops
                dim_str = ''.join(f"[{dim}]" for dim in array_dims)
                assignments.append(f"\n{indent}{element_type} {array_name}{dim_str};")
                
                # Generate initialization loops
                if len(array_dims) == 1:
                    # 1D array
                    loop_var = "i"
                    assignments.append(f"\n{indent}for(int {loop_var} = 0; {loop_var} < {array_dims[0]}; {loop_var}++) {{")
                    assignments.append(f"\n{indent}\t{array_name}[{loop_var}] = {init_expr};")
                    assignments.append(f"\n{indent}}}")
                elif len(array_dims) == 2:
                    # 2D array
                    assignments.append(f"\n{indent}for(int i = 0; i < {array_dims[0]}; i++) {{")
                    assignments.append(f"\n{indent}\tfor(int j = 0; j < {array_dims[1]}; j++) {{")
                    assignments.append(f"\n{indent}\t\t{array_name}[i][j] = {init_expr};")
                    assignments.append(f"\n{indent}\t}}")
                    assignments.append(f"\n{indent}}}")
                
                self.declared_vars.add(array_name)
        
        elif op_name == 'subscript_assign':
            array_var = inputs[0]
            index_var = inputs[1]
            value_var = inputs[2]
            
            # Ensure index is int for GLSL
            if self._is_literal(index_var) and index_var.endswith('u'):
                index_var = f"int({index_var[:-1]})"
            
            assignments.append(f"\n{indent}{array_var}[{index_var}] = {value_var};")
            return  # Return early to avoid processing as regular operation
        
        elif op_name == 'subscript_assign_2d':
            array_var = inputs[0]
            index1_var = inputs[1]
            index2_var = inputs[2]
            value_var = inputs[3]
            
            # Ensure indices are int for GLSL
            if self._is_literal(index1_var) and index1_var.endswith('u'):
                index1_var = f"int({index1_var[:-1]})"
            if self._is_literal(index2_var) and index2_var.endswith('u'):
                index2_var = f"int({index2_var[:-1]})"
            
            assignments.append(f"\n{indent}{array_var}[{index1_var}][{index2_var}] = {value_var};")
            return  # Return early to avoid processing as regular operation
        
        elif op_name == 'cast':
            target_type = inputs[1]
            source_var = inputs[0]
            source_type = self._get_var_type(source_var)
            self.var_types[output_var] = target_type
            
            if target_type != source_type:
                expr = f"{target_type}({source_var})"
            else:
                expr = source_var
            
            if output_var == "_void_":
                # Just output the expression as a statement
                assignments.append(f"\n{indent}{expr};")
                return

            # Define out_type here
            out_type = target_type
            
            if output_var not in self.declared_vars:
                assignments.append(f"\n{indent}{out_type} {output_var} = {expr};")
                self.declared_vars.add(output_var)
            else:
                assignments.append(f"\n{indent}{output_var} = {expr};")
                
        elif op_name == 'assign':
            target_var = inputs[0]
            value_var = inputs[1]
            value_type = self._get_var_type(value_var)
            target_type = self._get_var_type(target_var)
            
            if target_var in self.explicit_types and value_type != target_type:
                expr = f"{target_type}({value_var})"
            else:
                expr = value_var
                if target_var not in self.explicit_types:
                    self.var_types[target_var] = value_type
                    target_type = value_type
            
            if target_var not in self.declared_vars:
                assignments.append(f"\n{indent}{target_type} {target_var} = {expr};")
                self.declared_vars.add(target_var)
            else:
                assignments.append(f"\n{indent}{target_var} = {expr};")
                    
            # Store the output_var type if different from target_var
            if output_var and output_var != target_var:
                self.var_types[output_var] = target_type

        elif op_name.startswith(('vec', 'uvec', 'ivec')):
            # Determine vector type and dimension
            if op_name.startswith('vec'):
                base_type = 'float'
            elif op_name.startswith('uvec'):
                base_type = 'uint'
            elif op_name.startswith('ivec'):
                base_type = 'int'
            
            # Get dimension from name (e.g., 'vec2' -> 2)
            try:
                dim = int(op_name[-1])
                if dim not in [2, 3, 4]:
                    raise ValueError(f"Invalid vector dimension: {dim}")
            except ValueError:
                # Default to 2 if can't parse
                dim = 2
            
            out_type = op_name
            self.var_types[output_var] = out_type
            
            # Process inputs based on dimension
            processed_inputs = []
            for inp in inputs:
                inp_type = self._get_var_type(inp)
                
                # For integer vectors, ensure inputs are appropriate type
                if base_type == 'int' and inp_type in ['uint', 'float']:
                    if self._is_literal(inp):
                        # Convert literal to int
                        try:
                            val = int(float(inp))
                            processed_inputs.append(str(val))
                        except:
                            processed_inputs.append(f"int({inp})")
                    else:
                        processed_inputs.append(f"int({inp})")
                elif base_type == 'uint' and inp_type in ['int', 'float']:
                    if self._is_literal(inp):
                        # Convert literal to uint
                        try:
                            val = int(float(inp))
                            if val < 0:
                                val = 0
                            processed_inputs.append(f"{val}u")
                        except:
                            processed_inputs.append(f"uint({inp})")
                    else:
                        processed_inputs.append(f"uint({inp})")
                elif base_type == 'float' and inp_type in ['int', 'uint']:
                    if self._is_literal(inp):
                        # Add .0 to integer literals for float vectors
                        if '.' not in inp and 'e' not in inp.lower():
                            processed_inputs.append(f"{inp}.0")
                        else:
                            processed_inputs.append(inp)
                    else:
                        processed_inputs.append(f"float({inp})")
                else:
                    processed_inputs.append(inp)
            
            # Build the constructor expression
            if len(processed_inputs) == 1 and dim > 1:
                # Single argument constructor (e.g., vec3(1.0))
                expr = f"{op_name}({processed_inputs[0]})"
            elif len(processed_inputs) == dim:
                # Full constructor (e.g., vec2(x, y))
                expr = f"{op_name}({', '.join(processed_inputs)})"
            else:
                # Handle partial construction or mixed arguments
                # For simplicity, we'll use the first argument repeated
                if processed_inputs:
                    expr = f"{op_name}({', '.join([processed_inputs[0]] * dim)})"
                else:
                    expr = f"{op_name}(0.0)"
            
            if output_var == "_void_":
                # Just output the expression as a statement
                assignments.append(f"\n{indent}{expr};")
                return
            
            if output_var not in self.declared_vars:
                assignments.append(f"\n{indent}{out_type} {output_var} = {expr};")
                self.declared_vars.add(output_var)
            else:
                assignments.append(f"\n{indent}{output_var} = {expr};")
            return
            
        elif any(f['name'] == op_name for f in self.graph.glsl_functions):
            out_type = 'float'
            self.var_types[output_var] = out_type
            
            if output_var not in self.declared_vars:
                assignments.append(f"\n{indent}{out_type} {output_var} = {op_name}({', '.join(inputs)});")
                self.declared_vars.add(output_var)
            else:
                assignments.append(f"\n{indent}{output_var} = {op_name}({', '.join(inputs)});")
        else:
            out_type = self.infer_type(op_name, inputs)
            self.var_types[output_var] = out_type
            
            processed_inputs = []
            for inp in inputs:
                inp_type = self._get_var_type(inp)
                if self._is_literal(inp):
                    # For array subscript operations, ensure indices are int not uint
                    if op_name in ['subscript', 'subscript_2d'] and inp in inputs[1:]:  # indices are after the array
                        # Force integer literals for array indices
                        if inp.endswith('u'):
                            processed_inputs.append(f"int({inp[:-1]})")
                        else:
                            processed_inputs.append(f"int({inp})")
                    elif inp_type == 'float' and '.' not in inp and 'e' not in inp.lower():
                        processed_inputs.append(f"{inp}.0")
                    elif inp_type == 'uint' and not inp.startswith('-'):
                        # Handle uint literals - only add 'u' suffix if needed
                        # For operations that mix int and uint, prefer int
                        other_inputs_have_int = any(
                            self._get_var_type(other) == 'int' 
                            for other in inputs 
                            if other != inp
                        )
                        if other_inputs_have_int or op_name in ['add', 'sub', 'mult', 'div', 'mod', 'floordiv']:
                            # Convert to int for mixed operations
                            processed_inputs.append(f"int({inp})")
                        else:
                            processed_inputs.append(f"{inp}u")
                    elif inp_type == 'int' and inp.startswith('-'):
                        processed_inputs.append(f"{inp}")
                    else:
                        processed_inputs.append(inp)
                else:
                    # Handle non-literal variables
                    if inp_type == 'uint':
                        # Check if any other input is int
                        other_inputs_have_int = any(
                            self._get_var_type(other) == 'int' 
                            for other in inputs 
                            if other != inp
                        )
                        if other_inputs_have_int or op_name in ['add', 'sub', 'mult', 'div', 'mod', 'floordiv', 'and', 'or', 'xor']:
                            # Cast uint to int for mixed or bitwise operations
                            processed_inputs.append(f"int({inp})")
                        else:
                            processed_inputs.append(inp)
                    else:
                        processed_inputs.append(inp)
            
            expr = self.OP_TO_GLSL.get(op_name, lambda x: f"{op_name}({', '.join(x)})")(processed_inputs)
            
            if output_var not in self.declared_vars:
                assignments.append(f"\n{indent}{out_type} {output_var} = {expr};")
                self.declared_vars.add(output_var)
            else:
                assignments.append(f"\n{indent}{output_var} = {expr};")