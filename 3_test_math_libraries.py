from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, Shader, TransparencyAttrib, Vec4
import kungfu as kf

app = ShowBase()
engine = kf.GPUMath(app)
cm = CardMaker("card")
cm.setFrame(-1, 1, -1, 1)
node = app.aspect2d.attachNewNode(cm.generate())
node.setPos(0, 0, 0)

# contains dist engine function
engine.import_file("./shader_libraries/math.py") 

@engine.shader('vertex')
def vertex_shader():
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex

@engine.shader('fragment')
def fragment_shader():
    uv_x = gl_FragCoord.x / 600
    uv_y = gl_FragCoord.y / 600

    x = uv_x * 2.0 - 1.0
    y = uv_y * 2.0 - 1.0

    # imported from math library
    distance = dist(x, y)
    distance2 = dist(x, y + 0.3)
    distance3 = dist(x - 0.3, y)

    p3d_FragColor = vec4(
        1 * distance,
        0.7 * distance2,
        distance3 * 4, 
        1.0
    )

vertex, vertex_info = engine.compile_shader(vertex_shader, debug=True)
fragment, fragment_info = engine.compile_shader(fragment_shader, debug=True)
shader = Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)
node.setShader(shader)
app.run()