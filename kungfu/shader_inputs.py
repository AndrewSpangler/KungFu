import re
from typing import Dict, List, Tuple
from .gl_typing import GLTypes, PANDA3D_BUILTINS, GLSL_BUILTINS, ShaderType

PANDA3D_BUILTINS_DETAILED = {
    # Vertex shader inputs (attributes)
    'vertex': {
        'p3d_Vertex': ('vec4', 'in', 'readonly'),
        'p3d_Normal': ('vec3', 'in', 'readonly'),
        'p3d_Color': ('vec4', 'in', 'readonly'),
        'p3d_MultiTexCoord': ('vec2', 'in', 'readonly'),
        'p3d_Binormal': ('vec3', 'in', 'readonly'),
        'p3d_Tangent': ('vec3', 'in', 'readonly'),
    },
    
    # Vertex shader outputs
    'vertex_output': {
        'p3d_Color': ('vec4', 'out', 'writeonly'),
        'p3d_TexCoord': ('vec2', 'out', 'writeonly'),
        'p3d_Normal': ('vec3', 'out', 'writeonly'),
    },
    
    # Fragment shader inputs
    'fragment': {
        'p3d_Color': ('vec4', 'in', 'readonly'),
        'p3d_TexCoord': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord0': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord1': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord2': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord3': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord4': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord5': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord6': ('vec2', 'in', 'readonly'),
        'p3d_TexCoord7': ('vec2', 'in', 'readonly'),
        'p3d_Normal': ('vec3', 'in', 'readonly'),
    },
    
    # Fragment shader outputs
    'fragment_output': {
        'p3d_FragColor': ('vec4', 'out', 'writeonly'),
        'p3d_FragData': ('vec4[]', 'out', 'writeonly'),
    },
    
    # Common uniforms
    'uniforms': {
        'p3d_ModelViewProjectionMatrix': ('mat4', 'uniform', 'readonly'),
        'p3d_ModelViewMatrix': ('mat4', 'uniform', 'readonly'),
        'p3d_ProjectionMatrix': ('mat4', 'uniform', 'readonly'),
        'p3d_ModelMatrix': ('mat4', 'uniform', 'readonly'),
        'p3d_ViewMatrix': ('mat4', 'uniform', 'readonly'),
        'p3d_NormalMatrix': ('mat3', 'uniform', 'readonly'),
        'osg_FrameTime': ('float', 'uniform', 'readonly'),
    },
    
    # Texture samplers
    'textures': {
        'p3d_Texture': ('sampler2D', 'uniform', 'readonly'),
        'p3d_TextureArray': ('sampler2DArray', 'uniform', 'readonly'),
    }
}

class ShaderInputManager:
    """Manages shader input/output declarations and uniforms"""
    
    @staticmethod
    def get_shader_stage_category(shader_type: ShaderType) -> str:
        """Convert ShaderType to category string"""
        if shader_type == ShaderType.VERTEX:
            return 'vertex'
        elif shader_type == ShaderType.FRAGMENT:
            return 'fragment'
        elif shader_type == ShaderType.GEOMETRY:
            return 'geometry'
        elif shader_type == ShaderType.COMPUTE:
            return 'compute'
        return 'unknown'
    
    @staticmethod
    def is_builtin_variable(var_name: str) -> Tuple[bool, str]:
        """Check if a variable is a built-in and return its category"""
        base_name = re.sub(r'(\d+)$', '', var_name)
        
        for category, builtins in PANDA3D_BUILTINS_DETAILED.items():
            if var_name in builtins:
                return True, category
            elif base_name + '0' in builtins:
                return True, category
        
        stage_categories = ['vertex', 'fragment', 'geometry']
        for category in stage_categories:
            if category in GLSL_BUILTINS and var_name in GLSL_BUILTINS[category]:
                return True, f'glsl_{category}'
        
        return False, ''
    
    @staticmethod
    def get_builtin_info(var_name: str, shader_type: ShaderType) -> Dict:
        """Get complete information about a built-in variable"""
        stage_category = ShaderInputManager.get_shader_stage_category(shader_type)
        
        base_name = re.sub(r'(\d+)$', '', var_name)
        number_match = re.search(r'(\d+)$', var_name)
        index = int(number_match.group(1)) if number_match else None
        
        # Try detailed Panda3D built-ins first
        for category, builtins in PANDA3D_BUILTINS_DETAILED.items():
            if var_name in builtins:
                glsl_type, storage, access = builtins[var_name]
                return {
                    'name': var_name,
                    'glsl_type': glsl_type,
                    'storage': storage,
                    'access': access,
                    'category': category,
                    'index': index
                }
        
        # Try GLSL built-ins
        if stage_category in GLSL_BUILTINS and var_name in GLSL_BUILTINS[stage_category]:
            glsl_type = GLSL_BUILTINS[stage_category][var_name]
            # Determine storage based on variable name patterns
            if var_name.startswith('gl_'):
                if 'ID' in var_name or 'Coord' in var_name or 'Facing' in var_name:
                    storage = 'in'
                    access = 'readonly'
                else:
                    storage = 'out'
                    access = 'writeonly'
            else:
                storage = 'in'
                access = 'readonly'
            
            return {
                'name': var_name,
                'glsl_type': glsl_type,
                'storage': storage,
                'access': access,
                'category': 'glsl_' + stage_category,
                'index': index
            }
        
        return None
    
    @staticmethod
    def generate_declaration(var_info: Dict) -> str:
        """Generate GLSL declaration for a variable"""
        if var_info['storage'] == 'uniform':
            return f"uniform {var_info['glsl_type']} {var_info['name']};"
        elif var_info['storage'] == 'in':
            if var_info['access'] == 'readonly':
                return f"layout(location={var_info.get('index', 0)}) in {var_info['glsl_type']} {var_info['name']};"
            else:
                return f"in {var_info['glsl_type']} {var_info['name']};"
        elif var_info['storage'] == 'out':
            if var_info['access'] == 'writeonly':
                return f"layout(location={var_info.get('index', 0)}) out {var_info['glsl_type']} {var_info['name']};"
            else:
                return f"out {var_info['glsl_type']} {var_info['name']};"
        return ''
    
    @staticmethod
    def get_default_outputs(shader_type: ShaderType) -> List[str]:
        """Get default output declarations for shader type"""
        if shader_type == ShaderType.FRAGMENT:
            return ["out vec4 p3d_FragColor;"]
        elif shader_type == ShaderType.VERTEX:
            return []
        return []