from typing import Dict
import numpy as np
from ..gl_typing import (
    GLTypes,
    ShaderType,
    OP_TO_GLSL,
    SHADER_BUILTINS,
    PANDA3D_BUILTINS,
    KUNGFU_BUILTINS,
    NP_TO_GLSL,
    GLSL_VERSION
)

def get_op_glsl(op_name: str, inputs: list) -> str:
    """Get GLSL expression for an operation"""
    if op_name in OP_TO_GLSL:
        return OP_TO_GLSL[op_name](inputs)
    return f"{op_name}({', '.join(inputs)})"

def is_panda3d_builtin(var_name: str) -> bool:
    """Check if variable is a Panda3D built-in"""
    return var_name in PANDA3D_BUILTINS

def is_kungfu_builtin(var_name: str) -> bool:
    """Check if a variable name is a KungFu built-in"""
    return var_name in KUNGFU_BUILTINS

def numpy_to_glsl_type(dtype):
    if dtype == np.complex64:
        return GLTypes.vec2
    elif dtype == np.complex128:
        return GLTypes.vec2
    else:
        return NP_TO_GLSL.get(
            dtype.type if hasattr(dtype, 'type') else dtype,
            GLTypes.float
        )

def get_shader_version(shader_type: ShaderType) -> int:
    """Get appropriate GLSL version for shader type"""
    return GLSL_VERSION.get(shader_type.value, GLSL_VERSION["default"])

def get_builtin_variables(shader_type: ShaderType) -> Dict[str, str]:
    """Get built-in variables for specific shader type"""
    builtins = {}
    builtins.update(SHADER_BUILTINS.get(shader_type, {}))
    builtins.update(PANDA3D_BUILTINS)
    return builtins

def get_kungfu_builtin_glsl(var_name: str) -> str:
    """Get the GLSL expression for a KungFu built-in variable"""
    mappings = {
        'gid': 'gl_GlobalInvocationID.x',
        'gid_x': 'gl_GlobalInvocationID.x',
        'gid_y': 'gl_GlobalInvocationID.y',
        'gid_z': 'gl_GlobalInvocationID.z',
        'gid_xyz': 'gl_GlobalInvocationID',
        'wgid': 'gl_WorkGroupID',
        'wgid_x': 'gl_WorkGroupID.x',
        'wgid_y': 'gl_WorkGroupID.y',
        'wgid_z': 'gl_WorkGroupID.z',
        'lid': 'gl_LocalInvocationID',
        'lid_x': 'gl_LocalInvocationID.x',
        'lid_y': 'gl_LocalInvocationID.y',
        'lid_z': 'gl_LocalInvocationID.z',
        'lid_idx': 'gl_LocalInvocationIndex',
        'wg_size': 'gl_WorkGroupSize',
        'wg_size_x': 'gl_WorkGroupSize.x',
        'wg_size_y': 'gl_WorkGroupSize.y',
        'wg_size_z': 'gl_WorkGroupSize.z',
        'num_wg': 'gl_NumWorkGroups',
        'num_wg_x': 'gl_NumWorkGroups.x',
        'num_wg_y': 'gl_NumWorkGroups.y',
        'num_wg_z': 'gl_NumWorkGroups.z',
        'n_items': 'nItems',
        'global_idx': 'gid',  # For 1D kernels, global_idx is just gid
    }
    return mappings.get(var_name, var_name)