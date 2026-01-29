from typing import Dict
from .compute_graph import ComputeGraph
from .gl_typing import ShaderType
from .unified_compiler import UnifiedCompiler

class ShaderCompiler:
    """Wrapper for UnifiedCompiler for backward compatibility"""
    
    def __init__(self, graph: ComputeGraph, input_types: Dict[str, str], 
                 explicit_types: Dict[str, str] = None, local_functions: Dict = None):
        self.compiler = UnifiedCompiler(
            graph, ShaderType.COMPUTE, input_types, explicit_types,
            used_builtins=None, custom_uniforms=None, custom_textures=None,
            local_functions=local_functions
        )
        self.graph = graph
        self.input_types = input_types
        self.explicit_types = explicit_types or {}
    
    def compile(self) -> str:
        """Compile the compute graph to GLSL"""
        return self.compiler.compile()