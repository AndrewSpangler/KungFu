from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, Shader, TextNode, Vec4, Vec3
import kungfu as kf
from kungfu.gl_typing import TextureFilter, WrapMode
import math

from panda3d.core import loadPrcFileData
loadPrcFileData("", "win-size 600 600")

app = ShowBase()
engine = kf.GPUMath(app)

# Create a full-screen card
cm = CardMaker("card")
cm.setFrame(-1, 1, -1, 1)
node = app.aspect2d.attachNewNode(cm.generate())
node.setPos(0, 0, 0)

# Import function libraries
engine.import_file("./shader_libraries/math.py") 
engine.import_file("./shader_libraries/colors.py") 
engine.import_file("./shader_libraries/strings.py") 

@engine.function({
    'input_string': f'uint[{255}]',
    'color': kf.GLTypes.vec4,
    'uv_x': kf.GLTypes.float,
    'uv_y': kf.GLTypes.float,
    'charmap_texture': kf.GLTypes.sampler2D,
    'char_uvs': 'vec4[48]'
}, return_type=kf.GLTypes.vec4)
def render_text_row(
    input_string: f'uint[{255}]',
    color: kf.GLTypes.vec4,
    uv_x: kf.GLTypes.float,
    uv_y: kf.GLTypes.float,
    charmap_texture: kf.GLTypes.sampler2D,
    char_uvs: 'vec4[48]'
) -> kf.GLTypes.vec4:
    # Define text rendering area
    text_cols: uint = uint(16)
    text_rows: uint = uint(1)
    text_scale: float = 0.05
   
    # Position text in the center of the screen
    text_area_x: float = 0.5 - (text_scale * float(text_cols)) / 2.0
    text_area_y: float = 0.5 - (text_scale * float(text_rows)) / 2.0
    
    # Calculate if we're inside the text area
    in_text_area: bool = bool(False)
    if uv_x >= text_area_x and uv_x <= text_area_x + text_scale * float(text_cols):
        if uv_y >= text_area_y and uv_y <= text_area_y + text_scale * float(text_rows):
            in_text_area = bool(True)
    
    result: vec4 = vec4(0.1, 0.2, 0.3, 1.0)
    
    if in_text_area:
        # Calculate normalized position within text area
        text_x: float = (uv_x - text_area_x) / (text_scale * float(text_cols))
        text_y: float = (uv_y - text_area_y) / (text_scale * float(text_rows))
        
        # Render the text
        text_color: vec4 = vec4(1.0, 1.0, 1.0, 1.0)
        text_result: vec4 = render_text(
            input_string, 
            text_color, 
            text_cols, 
            text_rows, 
            text_x, 
            text_y,
            charmap_texture,
            char_uvs
        )
        
        # Blend text with background
        if text_result.a > 0.001:
            result = text_result

    return result

@engine.shader('vertex')
def vertex_shader():
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex

@engine.shader('fragment', {
    'time'              : (kf.GLTypes.float,     'readonly'),
    'resolution'        : (kf.GLTypes.vec3,      'readonly'),
    'input_string'      : (kf.GLTypes.uint+f"[{engine.STRLIB_MAX_LENGTH}]",'readonly'),
    'charmap_texture'   : (kf.GLTypes.sampler2D, 'readonly'),
    'char_uvs'          : (kf.GLTypes.vec4+f"[{engine.STRLIB_CHAR_UV_LENGTH}]",'readonly'),
})
def fragment_shader():
    uv_x = gl_FragCoord.x / resolution.x
    uv_y = gl_FragCoord.y / resolution.y
    
    if uv_x < 0.5:
        if uv_y < 0.5:
            p3d_FragColor = render_text_row(
                "CUSTOM STRING",
                vec4(0,0,0,0),
                uv_x * 2,
                uv_y * 2,
                charmap_texture,
                char_uvs
            )
        else:
            p3d_FragColor = render_text_row(
                string_to_upper(input_string),
                vec4(0,0,0,0),
                uv_x * 2,
                (uv_y - 0.5) * 2,
                charmap_texture,
                char_uvs
            )
    else:
        if uv_y < 0.5:
            p3d_FragColor = render_text_row(
                string_to_lower(input_string),
                vec4(0,0,0,0),
                (uv_x - 0.5) * 2,
                uv_y * 2,
                charmap_texture,
                char_uvs
            )
        else:
            p3d_FragColor = render_text_row(
                string_replace_char( # o -> 0
                    string_replace_char( # l -> !
                        input_string, uint(79), uint(47)
                    ), uint(76), uint(1)
                ), # o -> 0
                vec4(0,0,0,0),
                (uv_x - 0.5) * 2,
                (uv_y - 0.5) * 2,
                charmap_texture,
                char_uvs
            )

def create_panda_shader(vert, frag) -> Shader:
    vertex, vertex_info = engine.compile_shader(vert, debug=True)
    fragment, fragment_info = engine.compile_shader(frag, debug=True)
    return Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)

shader = create_panda_shader(vertex_shader, fragment_shader)

node.setShader(shader)
# Set initial uniform values
node.setShaderInput("mode", 0)
node.setShaderInput("time", 0.0)
node.setShaderInput("input_string", engine.encode_string("Hello World"))
node.setShaderInput("charmap_texture", engine.charmap_texture)
node.setShaderInput("char_uvs", engine.STRLIB_PACKED_UVS)

text_node = TextNode("mode_text")
text_node.setText("String Test")
text_node.setTextColor(1, 1, 1, 1)
text_node.setShadow(0.05, 0.05)
text_node.setShadowColor(0, 0, 0, 1)
text_np = aspect2d.attachNewNode(text_node)
text_np.setScale(0.07)
text_np.setPos(-1.2, 0, 0.9)

# Task to update time and mode
def update_shader(task):
    global current_mode
    mouse_node = app.mouseWatcherNode
    dt = app.taskMgr.globalClock.getDt()
    w, h = app.win.getXSize(), app.win.getYSize()
    node.setShaderInput("resolution", Vec3(w, h, 1.0))
    node.setShaderInput("time", task.time)

    return task.cont

# Add the update task
app.taskMgr.add(update_shader, "update_shader")
app.run()