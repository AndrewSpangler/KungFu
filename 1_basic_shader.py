from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, Shader, TransparencyAttrib, Vec4
import kungfu as kf

app = ShowBase()
engine = kf.GPUMath(app)
cm = CardMaker("card")
cm.setFrame(-0.5, 0.5, -0.5, 0.5)
node = app.aspect2d.attachNewNode(cm.generate())
node.setPos(0, 0, 0)

@engine.function(
    param_types={'matrix': 'mat4', 'position': 'vec4'},
    return_type='vec4'
)
def custom_position(matrix, position) -> Vec4:
    # Double position
    return vec4(matrix * (2.0 * position)) 

@engine.shader('vertex')
def vertex_shader():
    position: vec4 = p3d_Vertex
    gl_Position : vec4 = custom_position(p3d_ModelViewProjectionMatrix, position)

@engine.shader('fragment')
def fragment_shader():
    p3d_FragColor = vec4(1, 0, 0, 1)

vertex, vertex_info = engine.compile_shader(vertex_shader, debug=True)
fragment, fragment_info = engine.compile_shader(fragment_shader, debug=True)
shader = Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)

node.setShader(shader)
node.setTransparency(TransparencyAttrib.MAlpha)
app.run()