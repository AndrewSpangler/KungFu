import ast
import inspect
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional, Union
from panda3d.core import Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f
from .composition import create_shader, STANDARD_HEADING, _buff_line
from .cast_buffer import CastBuffer
from .gl_typing import GLTypes, NP_GLTypes, Vec_GLTypes, VEC_TO_GLSL
from .compute_graph import ComputeGraph

class PythonToGLSLTranspiler(ast.NodeVisitor):
    def __init__(self, arg_names: List[str], hints: Dict[str, tuple] = None):
        self.graph = ComputeGraph()
        self.arg_names = arg_names
        self.var_map = {}
        self.local_functions = {}
        self.inline_always = set()
        self.explicit_types = {}
        self.hints = hints or {}
        
        for arg in arg_names:
            hint = self.hints.get(arg, (NP_GLTypes.float, "buffer"))
            storage = hint[1] if len(hint) > 1 else "buffer"
            self.graph.add_input(arg, storage)
            self.var_map[arg] = arg
    
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
        else:
            return 'float'

    def _is_literal(self, val: str) -> bool:
        try:
            float(val)
            return True
        except ValueError:
            return val.lower() in ['true', 'false']

    def visit_Constant(self, node):
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

    def visit_Name(self, node):
        if node.id in self.var_map:
            return self.var_map[node.id]
        raise ValueError(f"Variable '{node.id}' used before assignment")

    def visit_AnnAssign(self, node):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple variable assignment supported")
        
        target = node.target.id
        
        type_name = None
        if isinstance(node.annotation, ast.Name):
            type_name = node.annotation.id
        elif isinstance(node.annotation, ast.Attribute):
            type_name = node.annotation.attr
        
        glsl_type_map = {
            'int': 'int', 'uint': 'uint', 'float': 'float',
            'double': 'double', 'bool': 'bool'
        }
        
        glsl_type = glsl_type_map.get(type_name, 'float')
        self.explicit_types[target] = glsl_type
        
        if node.value:
            result_var = self.visit(node.value)
            self.var_map[target] = target
            result_type = self._get_var_type(result_var)
            
            if glsl_type != result_type:
                cast_var = self.graph.add_operation('cast', [result_var, glsl_type], 
                                                in_loop=bool(self.graph.current_scope))
                self.graph.add_operation('assign', [target, cast_var], 
                                        in_loop=bool(self.graph.current_scope))
            else:
                self.graph.add_operation('assign', [target, result_var], 
                                        in_loop=bool(self.graph.current_scope))
        else:
            self.var_map[target] = target
        
        return target

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            raise ValueError("Multiple assignment targets not supported")
        
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            raise ValueError("Only simple variable assignment supported")
        
        target_name = target.id
        result_var = self.visit(node.value)
        self.var_map[target_name] = target_name
        self.graph.add_operation('assign', [target_name, result_var], 
                                in_loop=bool(self.graph.current_scope))
        return target_name

    def visit_AugAssign(self, node):
        target = node.target.id if isinstance(node.target, ast.Name) else None
        if not target:
            raise ValueError("Only simple variable assignment supported")
        
        left = self.visit(node.target)
        right = self.visit(node.value)
        
        op_map = {
            ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mult',
            ast.Div: 'div', ast.Pow: 'pow'
        }
        
        op_name = op_map.get(type(node.op))
        if op_name is None:
            raise ValueError(f"Unsupported augmented assignment: {type(node.op)}")
        
        temp_result = self.graph.add_operation(op_name, [left, right], 
                                              in_loop=bool(self.graph.current_scope))
        self.graph.add_operation('assign', [target, temp_result], 
                                in_loop=bool(self.graph.current_scope))
        self.var_map[target] = target
        return target

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Pow):
            return self.graph.add_operation('pow', [left, right], 
                                           in_loop=bool(self.graph.current_scope))
        
        op_map = {
            ast.Add: 'add', ast.Sub: 'sub', ast.Mult: 'mult', ast.Div: 'div',
            ast.Mod: 'mod', ast.BitAnd: 'and', ast.BitOr: 'or',
            ast.BitXor: 'xor', ast.LShift: 'lsh', ast.RShift: 'rsh',
        }
        
        op_type = type(node.op)
        if op_type not in op_map:
            raise ValueError(f"Unsupported binary operation: {op_type}")
        
        return self.graph.add_operation(op_map[op_type], [left, right], 
                                       in_loop=bool(self.graph.current_scope))
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            raise ValueError(f"Unsupported function call type: {type(node.func)}")
        
        args = [self.visit(arg) for arg in node.args]
        
        if func_name in ['int', 'float', 'bool', 'uint', 'double']:
            target_type = func_name
            return self.graph.add_operation('cast', [args[0], target_type], 
                                           in_loop=bool(self.graph.current_scope))
        
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
        
        return self.graph.add_operation(func_name, args, 
                                       in_loop=bool(self.graph.current_scope))
    
    def visit_Compare(self, node):
        left = self.visit(node.left)
        comparisons = []
        
        for op, right in zip(node.ops, node.comparators):
            right_var = self.visit(right)
            
            op_map = {
                ast.Lt: 'lt', ast.LtE: 'lte', ast.Gt: 'gt',
                ast.GtE: 'gte', ast.Eq: 'eq', ast.NotEq: 'neq',
            }
            
            op_type = type(op)
            if op_type not in op_map:
                raise ValueError(f"Unsupported comparison: {op_type}")
            
            result = self.graph.add_operation(op_map[op_type], [left, right_var], 
                                            in_loop=bool(self.graph.current_scope))
            comparisons.append(result)
            left = right_var
        
        if len(comparisons) == 1:
            return comparisons[0]
        
        result = comparisons[0]
        for comp in comparisons[1:]:
            result = self.graph.add_operation('and', [result, comp], 
                                            in_loop=bool(self.graph.current_scope))
        return result
    
    def visit_If(self, node):
        condition = self.visit(node.test)
        
        if_info = {
            'type': 'if',
            'condition': condition,
            'then_body': [],
            'else_body': [] if node.orelse else None
        }
        
        self.graph.start_loop(if_info)
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.graph.end_loop()
        
        if node.orelse:
            else_info = {'type': 'else', 'body': []}
            self.graph.start_loop(else_info)
            
            for stmt in node.orelse:
                self.visit(stmt)
            
            self.graph.end_loop()
        
        return None
    
    def visit_For(self, node):
        if not isinstance(node.target, ast.Name):
            raise ValueError("Only simple loop variables supported")
        
        loop_var = node.target.id
        
        if not isinstance(node.iter, ast.Call) or node.iter.func.id != 'range':
            raise ValueError("Only range() loops supported")
        
        args = [self.visit(arg) for arg in node.iter.args]
        
        if len(args) == 1:
            start, end, step = '0', args[0], '1'
        elif len(args) == 2:
            start, end, step = args[0], args[1], '1'
        elif len(args) == 3:
            start, end, step = args[0], args[1], args[2]
        else:
            raise ValueError("range() requires 1-3 arguments")
        
        loop_info = {
            'type': 'for',
            'var': loop_var,
            'start': start,
            'end': end,
            'step': step
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
    
    def visit_While(self, node):
        test = self.visit(node.test)
        
        loop_info = {'type': 'while', 'test': test}
        
        self.graph.start_loop(loop_info)
        
        for stmt in node.body:
            self.visit(stmt)
        
        self.graph.end_loop()
        
        return None
    
    def visit_Return(self, node):
        if node.value is None:
            self.graph.set_void_return()
            return None
        
        result_var = self.visit(node.value)
        self.graph.set_output(result_var)
        return result_var
    
    def visit_FunctionDef(self, node):
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
    
    def visit_Expr(self, node):
        return self.visit(node.value)


def _extract_hints(hints: Optional[Dict[str, tuple]]) -> Tuple[Dict[str, str], Dict[str, str]]:
    if hints is None:
        return {}, {}
    
    type_hints = {}
    storage_hints = {}
    
    for key, value in hints.items():
        if isinstance(value, tuple) and len(value) == 2:
            type_tuple, storage = value
            if isinstance(type_tuple, tuple) and len(type_tuple) == 2:
                type_hints[key] = type_tuple[1]
            else:
                raise ValueError(f"Invalid type for '{key}': expected tuple like NP_GLTypes.float or Vec_GLTypes.vec3")
            storage_hints[key] = storage
        else:
            raise ValueError(f"Invalid hint for '{key}': expected (Type, storage) tuple")
    
    return type_hints, storage_hints


def inline_always(func):
    func._inline_always = True
    return func


def gpu_kernel(hints: Optional[Dict[str, tuple]] = None):
    def decorator(func):
        source = inspect.getsource(func)
        
        lines = source.split('\n')
        func_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_start = i
                break
        
        func_lines = lines[func_start:]
        
        if func_lines:
            min_indent = min(len(line) - len(line.lstrip()) 
                            for line in func_lines if line.strip())
            func_lines = [line[min_indent:] if line.strip() else line 
                         for line in func_lines]
        
        source = '\n'.join(func_lines)
        tree = ast.parse(source)
        func_def = tree.body[0]
        
        arg_names = [arg.arg for arg in func_def.args.args]
        
        transpiler = PythonToGLSLTranspiler(arg_names, hints)
        
        for stmt in func_def.body:
            transpiler.visit(stmt)
        
        graph = transpiler.graph
        type_hints, storage_hints = _extract_hints(hints)
        
        func._compute_graph = graph
        func._arg_names = arg_names
        func._type_hints = type_hints
        func._storage_hints = storage_hints
        func._explicit_types = transpiler.explicit_types
        
        return func
    
    if callable(hints):
        func = hints
        hints = None
        return decorator(func)
    
    return decorator