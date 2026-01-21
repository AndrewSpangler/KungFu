from typing import Dict, List
from .composition import create_shader, STANDARD_HEADING, _buff_line
from .gl_typing import GLTypes, NP_GLTypes, GLSL_TO_NP
from .compute_graph import ComputeGraph

class ShaderCompiler:
    TYPE_HIERARCHY = {
        'bool': 0, 'int': 1, 'uint': 2, 'float': 3, 'double': 4
    }
    
    TYPE_PROMOTION_MATRIX = {
        ('bool', 'bool'): 'bool', ('bool', 'int'): 'int',
        ('bool', 'uint'): 'uint', ('bool', 'float'): 'float',
        ('bool', 'double'): 'double', ('int', 'int'): 'int',
        ('int', 'uint'): 'int', ('int', 'float'): 'float',
        ('int', 'double'): 'double', ('uint', 'uint'): 'uint',
        ('uint', 'float'): 'float', ('uint', 'double'): 'double',
        ('float', 'float'): 'float', ('float', 'double'): 'double',
        ('double', 'double'): 'double',
    }
    
    OP_TO_GLSL = {
        'add': lambda inputs: f"({' + '.join(inputs)})",
        'sub': lambda inputs: f"({inputs[0]} - {inputs[1]})",
        'mult': lambda inputs: f"({' * '.join(inputs)})",
        'div': lambda inputs: f"({inputs[0]} / {inputs[1]})",
        'neg': lambda inputs: f"(-{inputs[0]})",
        'square': lambda inputs: f"({inputs[0]} * {inputs[0]})",
        'gt': lambda inputs: f"({inputs[0]} > {inputs[1]})",
        'lt': lambda inputs: f"({inputs[0]} < {inputs[1]})",
        'eq': lambda inputs: f"({inputs[0]} == {inputs[1]})",
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
        'cast': lambda inputs: f"{inputs[1]}({inputs[0]})",
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
        'step': lambda inputs: f"step({inputs[0]}, {inputs[1]})",
        'smoothstep': lambda inputs: f"smoothstep({inputs[0]}, {inputs[1]}, {inputs[2]})",
        'assign': lambda inputs: f"{inputs[0]} = {inputs[1]}",
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
                    return 'int' if val < 0 else 'uint'
                except ValueError:
                    return 'float'
        return 'float'

    def promote_types(self, type1: str, type2: str) -> str:
        type1 = type1.lower()
        type2 = type2.lower()
        
        if type1 == type2:
            return type1
        
        key = tuple(sorted([type1, type2]))
        if key in self.TYPE_PROMOTION_MATRIX:
            return self.TYPE_PROMOTION_MATRIX[key]
        
        rank1 = self.TYPE_HIERARCHY.get(type1, 3)
        rank2 = self.TYPE_HIERARCHY.get(type2, 3)
        return type1 if rank1 > rank2 else type2

    def infer_type(self, op_name: str, input_vars: List[str]) -> str:
        input_types = [self._get_var_type(v) for v in input_vars]
        
        if op_name in ['gt', 'lt', 'eq', 'gte', 'lte', 'is_zero']:
            return 'bool'
        
        if op_name in ['and', 'or', 'xor', 'lsh', 'rsh']:
            if input_types and input_types[0] in ['int', 'uint']:
                return 'uint' if input_types[0] == 'uint' else 'int'
            return 'bool'
        
        if op_name in ['bool']:
            return 'bool'
        
        if op_name in ['div', 'avg', 'sqrt', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                       'exp', 'log', 'pow', 'floor', 'ceil', 'fract', 'round',
                       'mix', 'step', 'smoothstep']:
            return 'float'
        
        if op_name == 'mod':
            if all(t in ['int', 'uint'] for t in input_types):
                return input_types[0]
            return 'int'
        
        result_type = input_types[0] if input_types else 'float'
        for t in input_types[1:]:
            result_type = self.promote_types(result_type, t)
        
        return result_type

    def compile(self):
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
        
        buffer_count = 0
        for var in sorted(self.graph.input_vars):
            var_type = self.input_types.get(var, 'float')
            self.var_types[var] = var_type
            buffer_inputs.append((buffer_count, var, var_type))
            buffers.append(_buff_line(buffer_count, f"D{buffer_count}", f"data_{buffer_count}", var_type))
            assignments.append(f"\n\t{var_type} {var} = data_{buffer_count}[gid];")
            buffer_count += 1
        
        for var in sorted(self.graph.uniform_vars):
            var_type = self.input_types.get(var, 'float')
            self.var_types[var] = var_type
            uniforms.append(f"uniform {var_type} {var};")
        
        current_assignments = []
        indent_level = 1
        
        for step in self.graph.steps:
            if step['type'] == 'loop':
                self._process_loop_step(step, current_assignments, indent_level)
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
        
        code_parts = [STANDARD_HEADING]
        
        if uniforms:
            code_parts.extend(uniforms)
        
        if buffers:
            code_parts.extend(buffers)
        
        if functions_section:
            code_parts.append(functions_section)
        
        code_parts.append(f"""
void main() {{
\tuint gid = gl_GlobalInvocationID.x;
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
    
    def _process_loop_step(self, step: Dict, assignments: List[str], indent_level: int):
        indent = "\t" * indent_level
        loop_info = step['loop_info']
        
        if loop_info['type'] == 'for':
            loop_var = loop_info['var']
            start = loop_info['start']
            end = loop_info['end']
            step_val = loop_info.get('step', '1')
            
            assignments.append(f"\n{indent}for(int {loop_var} = {start}; {loop_var} < {end}; {loop_var} += {step_val}) {{")
            
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
    
    def _process_operation(self, op_name: str, inputs: List[str], output_var: str, 
                      assignments: List[str], indent_level: int):
        indent = "\t" * indent_level
        
        if op_name == 'cast':
            target_type = inputs[1]
            source_var = inputs[0]
            source_type = self._get_var_type(source_var)
            self.var_types[output_var] = target_type
            
            if target_type != source_type:
                expr = f"{target_type}({source_var})"
            else:
                expr = source_var
            
            if output_var not in self.declared_vars:
                assignments.append(f"\n{indent}{target_type} {output_var} = {expr};")
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
                
            if output_var and output_var != target_var:
                self.var_types[output_var] = target_type
                
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
                    if inp_type == 'float' and '.' not in inp and 'e' not in inp.lower():
                        processed_inputs.append(f"{inp}.0")
                    elif inp_type == 'uint' and not inp.startswith('-'):
                        processed_inputs.append(f"{inp}u")
                    elif inp_type == 'int' and inp.startswith('-'):
                        processed_inputs.append(f"{inp}")
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