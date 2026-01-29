from typing import Dict, List, Tuple

class ComputeGraph:
    def __init__(self):
        self.operations = [] 
        self.input_vars = set()
        self.uniform_vars = set()
        self.var_counter = 0
        self.output_var = None
        self.glsl_functions = []
        self.storage_hints = {}
        self.has_void_return = False
        self.steps = []
        self.current_scope = []
        self.static_constants = []  # List of (name, type, size, values) tuples
        self.var_types = {}
        self.vectorized = None  # Whether this kernel uses vectorized execution

    def add_operation(self, op_name: str, inputs: List[str], output_var: str = None, in_loop: bool = False):
        if output_var is None:
            output_var = f"_t{self.var_counter}"
            self.var_counter += 1
        
        self.operations.append((op_name, inputs, output_var))
        
        if in_loop and self.current_scope:
            current_scope = self.current_scope[-1]
            
            # Determine the correct body list based on scope type
            if current_scope['type'] == 'if':
                body_list = current_scope['then_body']
            elif current_scope['type'] == 'else':
                body_list = current_scope['body']
            else:
                # For loops and other scopes
                if 'body' not in current_scope:
                    current_scope['body'] = []
                body_list = current_scope['body']
            
            body_list.append({
                'type': 'operation',
                'op_name': op_name,
                'inputs': inputs,
                'output_var': output_var
            })
        else:
            self.steps.append({
                'type': 'operation',
                'op_name': op_name,
                'inputs': inputs,
                'output_var': output_var
            })
        
        return output_var
    
    
    def start_loop(self, loop_info: Dict):
        loop_step = {
            'type': 'loop',
            'loop_info': loop_info,
            'body': []
        }
        loop_info['body'] = []
        
        if self.current_scope:
            parent_scope = self.current_scope[-1]
            if 'body' not in parent_scope:
                parent_scope['body'] = []
            parent_scope['body'].append(loop_step)
        else:
            self.steps.append(loop_step)
        
        self.current_scope.append(loop_step)
        return loop_step
    
    def end_loop(self):
        if self.current_scope:
            self.current_scope.pop()
    
    def add_glsl_function(self, name: str, params: List[Tuple[str, str]], return_type: str, body_ops: List):
        self.glsl_functions.append({
            'name': name,
            'params': params,
            'return_type': return_type,
            'body_ops': body_ops
        })
    
    def set_output(self, var_name: str):
        self.output_var = var_name
    
    def set_void_return(self):
        self.has_void_return = True
        self.output_var = None

    def add_input(self, var_name: str, storage: str = 'buffer'):
        if storage == 'uniform':
            self.uniform_vars.add(var_name)
        else:
            self.input_vars.add(var_name)
        self.storage_hints[var_name] = storage
    
    def add_static_constant(self, name: str, glsl_type: str, size: int, values: List):
        """Add a static constant array"""
        self.static_constants.append((name, glsl_type, size, values))
    def optimize_temporaries(self):
        """Optimize the graph by eliminating unnecessary temporary variables"""
        # Placeholder - actual optimization in ShaderCompiler
        # Can do some basic optimizations here later
        pass