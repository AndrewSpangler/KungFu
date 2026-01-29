import ast
from typing import Dict, List, Tuple, Optional, Set, Any
from .function_registry import FunctionRegistry
from .gl_typing import OP_TO_GLSL, ShaderType
from .helpers import get_op_glsl

class ShaderFunction:
    """Represents a transpiled shader function"""
    
    def __init__(self, name: str, source_code: str = "", ast_node: ast.FunctionDef = None,
                 param_names: List[str] = None, param_types: Dict[str, str] = None,
                 return_type: str = "float"):
        self.name = name
        self.source_code = source_code
        self.ast_node = ast_node
        self.param_names = param_names or []
        self.param_types = param_types or {}
        self.return_type = return_type
    
    @classmethod
    def from_metadata(cls, func_metadata: Dict) -> 'ShaderFunction':
        """Create a ShaderFunction from function registry metadata"""
        return cls(
            name=func_metadata['name'],
            source_code=func_metadata.get('source_code', ''),
            ast_node=func_metadata.get('ast_node'),
            param_names=func_metadata.get('param_names', []),
            param_types=func_metadata.get('param_types', {}),
            return_type=func_metadata.get('return_type', 'float')
        )
class ShaderFunctionTranspiler:
    """Transpiler for shader functions"""
    
    # Use centralized OP_TO_GLSL
    OP_TO_GLSL = OP_TO_GLSL
    
    @classmethod
    def transpile_function(cls, func_metadata: Dict, shader_type: ShaderType) -> str:
        """Transpile a function to GLSL"""
        name = func_metadata['name']
        param_names = func_metadata['param_names']
        param_types = func_metadata['param_types']
        return_type = func_metadata['return_type']
        ast_node = func_metadata['ast_node']
        
        # Build parameter signature
        param_signature = []
        for param in param_names:
            param_type = param_types.get(param, 'float')
            param_signature.append(f"{param_type} {param}")
        
        # Transpile function body
        body_code = cls._transpile_function_body(ast_node, param_names, param_types, return_type)
        
        # Build function declaration
        signature = f"{return_type} {name}({', '.join(param_signature)})"
        glsl_code = f"{signature} {{\n{body_code}\n}}"
        
        return glsl_code
    
    @classmethod
    def _transpile_function_body(cls, func_def: ast.FunctionDef, param_names: List[str],
                                param_types: Dict[str, str], return_type: str) -> str:
        """Transpile function body to GLSL"""
        transpiler = FunctionBodyTranspiler(param_names, param_types)
        
        body_lines = []
        for stmt in func_def.body:
            if isinstance(stmt, ast.Return):
                if stmt.value:
                    result = transpiler.visit(stmt.value)
                    body_lines.append(f"\treturn {result};")
                else:
                    body_lines.append("\treturn;")
            elif isinstance(stmt, ast.Expr):
                result = transpiler.visit(stmt.value)
                body_lines.append(f"\t{result};")
        
        return '\n'.join(body_lines) if body_lines else "\t// Function body"
    
    @classmethod
    def get_all_dependencies(cls, function_name: str) -> Set[str]:
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
                calls = cls._find_function_calls(func_metadata['ast_node'])
                for called_func in calls:
                    if called_func in FunctionRegistry.get_all() and called_func not in visited:
                        to_visit.append(called_func)
        
        return visited
    
    @classmethod
    def _find_function_calls(cls, node: ast.AST) -> List[str]:
        """Find all function calls in an AST node"""
        calls = []
        
        class FunctionCallFinder(ast.NodeVisitor):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                self.generic_visit(node)
        
        finder = FunctionCallFinder()
        finder.visit(node)
        return calls

class FunctionBodyTranspiler(ast.NodeVisitor):
    """Simple transpiler for function bodies"""
    
    def __init__(self, param_names: List[str], param_types: Dict[str, str]):
        self.param_names = param_names
        self.param_types = param_types
        self.temp_counter = 0
    
    def _new_temp(self) -> str:
        """Create a new temporary variable"""
        temp = f"_t{self.temp_counter}"
        self.temp_counter += 1
        return temp
    
    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Visit binary operation"""
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        op_map = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Pow: '**',
        }
        
        op_type = type(node.op)
        if op_type == ast.Pow:
            return f"pow({left}, {right})"
        elif op_type in op_map:
            return f"({left} {op_map[op_type]} {right})"
        else:
            return f"({left} ? {right})"
    
    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Visit unary operation"""
        operand = self.visit(node.operand)
        
        op_map = {
            ast.USub: '-',
            ast.UAdd: '+',
        }
        
        op_type = type(node.op)
        if op_type in op_map:
            return f"{op_map[op_type]}{operand}"
        else:
            return f"?{operand}"
    
    def visit_Call(self, node: ast.Call) -> str:
        """Visit function call"""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        else:
            return f"/* unsupported call */"
        
        args = [self.visit(arg) for arg in node.args]
        
        # Check if it's a type constructor
        from .gl_typing import GLSL_TYPE_CONSTRUCTORS
        if func_name in GLSL_TYPE_CONSTRUCTORS:
            return f"{func_name}({', '.join(args)})"
        
        return f"{func_name}({', '.join(args)})"
    
    def visit_Name(self, node: ast.Name) -> str:
        """Visit variable name"""
        return node.id
    
    def visit_Constant(self, node: ast.Constant) -> str:
        """Visit constant value"""
        if node.value is True:
            return "true"
        elif node.value is False:
            return "false"
        elif node.value is None:
            return "0"
        elif isinstance(node.value, (int, float)):
            if isinstance(node.value, float):
                return str(node.value)
            else:
                return str(node.value)
        else:
            return f'"{node.value}"'
    
    def generic_visit(self, node: ast.AST) -> str:
        """Default visitor"""
        return f"/* {type(node).__name__} */"