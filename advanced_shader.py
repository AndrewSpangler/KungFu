from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, Shader, TransparencyAttrib, Vec4
import kungfu as kf

app = ShowBase()
engine = kf.GPUMath(app)
cm = CardMaker("card")
cm.setFrame(-1, 1, -1, 1)
node = app.aspect2d.attachNewNode(cm.generate())
node.setPos(0, 0, 0)

@engine.shader('vertex')
def vertex_shader():
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex

@engine.shader('fragment')
def fragment_shader():
    PI = 3.14159265358979323846
    time = osg_FrameTime
    
    uv_x = gl_FragCoord.x / 600
    uv_y = gl_FragCoord.y / 600

    pos_x = uv_x * 2.0 - 1.0
    pos_y = uv_y * 2.0 - 1.0
    
    dist = sqrt(pos_x * pos_x + pos_y * pos_y)
    angle = atan(pos_y, pos_x)
    
    # Create spiral twist
    twist_strength = 3.0
    twist = (1.0 - dist) * twist_strength + time * 0.5
    new_angle = angle + twist
    
    # Convert back to cartesian
    distorted_x = cos(new_angle) * dist
    distorted_y = sin(new_angle) * dist
    
    # Wave distortion
    wave_intensity = sin(dist * 8.0 - time * 2.0) * 0.15
    distorted_x = distorted_x + sin(time * 1.5 + angle * 3.0) * wave_intensity
    distorted_y = distorted_y + cos(time * 1.5 + angle * 3.0) * wave_intensity
    
    # Plasma effect
    c1 = sin(distorted_x * 10.0 + time)
    c2 = sin(distorted_y * 10.0 + time * 1.3)
    c3 = sin((distorted_x + distorted_y) * 8.0 + time * 0.8)
    
    ddist = sqrt(distorted_x * distorted_x + distorted_y * distorted_y)
    c4 = sin(ddist * 12.0 + time * 1.5)
    
    value = (c1 + c2 + c3 + c4) / 4.0
    
    # Shift colors over time
    r = sin(value * PI + time * 0.5) * 0.5 + 0.5
    g = sin(value * PI + 2.0 + time * 0.7) * 0.5 + 0.5
    b = sin(value * PI + 4.0 + time * 0.3) * 0.5 + 0.5

    # Add brightness pulsing
    brightness = 0.85 + sin(time * 2.0) * 0.15
    
    # Final color
    final_r = r * brightness
    final_g = g * brightness
    final_b = b * brightness

    p3d_FragColor = vec4(final_r, final_g, final_b, 1.0)

vertex, vertex_info = engine.compile_shader(vertex_shader, debug=True)
fragment, fragment_info = engine.compile_shader(fragment_shader, debug=True)
shader = Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)
node.setShader(shader)
app.run()