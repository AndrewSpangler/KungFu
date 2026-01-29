# [file name]: base_transpiler.py
"""
Base transpiler class with common functionality for all shader types.
"""

import ast
from .ast_utils import ASTVisitorBase
from .gl_typing import (
    GLTypes, TypeRules, BUILTIN_VARIABLES,
    GLSL_MATH_FUNCTIONS, GLSL_TYPE_CONSTRUCTORS,
    ALL_GLSL_FUNCTIONS, ShaderType, AST_BIN_SYMBOL_MAP,
    AST_UNARY_SYMBOL_MAP, AST_COMPARISON_SYMBOL_MAP,
    AST_BOOL_SYMBOL_MAP
)
from .helpers import CompilationError, get_builtin_variables

class BaseTranspiler(ASTVisitorBase, ast.NodeVisitor):
    """Base class for Python to GLSL transpilation"""
    
    def __init__(
        self,
        source_code: str = "",
        function_name: str = "", 
        file_path: str = ""
    ):
        super().__init__(source_code, function_name, file_path)
        self.builtin_variables.update(BUILTIN_VARIABLES)
        
    def visit_Constant(self, node):
        return self._visit_constant(node)
    
    def visit_Name(self, node):
        return self._visit_name(node)
    
    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
                
        op_type = type(node.op)
        if op_type == ast.Pow:
            return f"pow({left}, {right})"
        elif op_type in AST_BIN_SYMBOL_MAP:
            return f"({left} {AST_BIN_SYMBOL_MAP[op_type]} {right})"
        else:
            raise self._create_error(
                f"Unsupported binary operation: {op_type.__name__}", node.op
            )
    
    def visit_UnaryOp(self, node):
        operand = self.visit(node.operand)

        op_type = type(node.op)
        if op_type in AST_UNARY_SYMBOL_MAP:
            return f"{AST_UNARY_SYMBOL_MAP[op_type]}{operand}"
        else:
            raise self._create_error(
                f"Unsupported unary operation: {op_type.__name__}", node.op
            )
    
    def visit_Compare(self, node):
        left = self.visit(node.left)
        comparisons = []
        
        for op, right in zip(node.ops, node.comparators):
            right_var = self.visit(right)
            
            op_type = type(op)
            if op_type in AST_COMPARISON_SYMBOL_MAP:
                comparisons.append(f"({left} {AST_COMPARISON_SYMBOL_MAP[op_type]} {right_var})")
            else:
                raise self._create_error(
                    f"Unsupported comparison: {op_type.__name__}", op
                )
            left = right_var
        
        if len(comparisons) == 1:
            return comparisons[0]
        return '(' + ' && '.join(comparisons) + ')'
    
    def visit_BoolOp(self, node):
        op_type = type(node.op)
        
        if op_type not in AST_BOOL_SYMBOL_MAP:
            raise self._create_error(
                f"Unsupported boolean operation: {op_type.__name__}", node.op
            )
        
        op_symbol = AST_BOOL_SYMBOL_MAP[op_type]
        values = [self.visit(value) for value in node.values]
        
        if len(values) < 2:
            raise self._create_error(
                f"Boolean operation requires at least 2 values", node
            )
        return '(' + f' {op_symbol} '.join(values) + ')'
    
    def visit_Call(self, node):
        # Use centralized GLSL function sets
        return self._visit_call(node, None, ALL_GLSL_FUNCTIONS, GLSL_TYPE_CONSTRUCTORS)
    
    def visit_Subscript(self, node):
        value = self.visit(node.value)
        
        if isinstance(node.slice, ast.Index):
            index = self.visit(node.slice.value)
            return f"{value}[{index}]"
        elif isinstance(node.slice, ast.Slice):
            start = self.visit(node.slice.lower) if node.slice.lower else "0"
            stop = self.visit(node.slice.upper) if node.slice.upper else ""
            step = self.visit(node.slice.step) if node.slice.step else ""
            
            if step:
                return f"{value}[{start}:{stop}:{step}]"
            elif stop:
                return f"{value}[{start}:{stop}]"
            else:
                return f"{value}[{start}:]"
        else:
            raise self._create_error(
                "Complex subscript not supported", node
            )
    
    def visit_Attribute(self, node):
        value = self.visit(node.value)
        attr = node.attr
        
        # Handle vector swizzling
        if attr in SWIZZLES:
            return f"{value}.{attr}"
        
        # Handle matrix access
        if attr.startswith('_m'):
            try:
                idx = int(attr[2:])
                return f"{value}[{idx}]"
            except:
                pass
        
        raise self._create_error(
            f"Unsupported attribute access: .{attr}", node
        )
    
    def generic_visit(self, node):
        """Handle unsupported nodes"""
        node_type = node.__class__.__name__
        raise self._create_error(
            f"Unsupported syntax in shader: {node_type}", node
        )