from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, Shader, TextNode, Vec4, Vec3
import kungfu as kf
import math

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

@engine.shader('vertex')
def vertex_shader():
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex

@engine.shader('fragment', uniforms={'mode': ('uint', 'readonly'), 'time': ('float', 'readonly')})
def fragment_shader():
    uv_x = gl_FragCoord.x / 600.0
    uv_y = gl_FragCoord.y / 600.0
    
    # Animated points
    t = time * 0.5
    point1_x = 0.5 + 0.3 * sin(t)
    point1_y = 0.5 + 0.3 * cos(t)
    
    point2_x = 0.5 + 0.2 * sin(t * 1.5)
    point2_y = 0.5 + 0.2 * cos(t * 1.3)
    
    point3_x = 0.5 + 0.4 * sin(t * 0.7)
    point3_y = 0.5 + 0.4 * cos(t * 0.9)
    
    # Calculate distances to animated points
    dist1 = dist(uv_x - point1_x, uv_y - point1_y)
    dist2 = dist(uv_x - point2_x, uv_y - point2_y)
    dist3 = dist(uv_x - point3_x, uv_y - point3_y)
    
    # Create base color from distances
    base_color : vec3 = vec3(
        0.5 + 0.5 * sin(dist1 * 10.0 - time),
        0.5 + 0.5 * sin(dist2 * 12.0 + time * 0.7),
        0.5 + 0.5 * sin(dist3 * 8.0 - time * 0.5)
    )
    
    if mode == uint(0):
        # Sepia effect
        result : vec3 = sepia(base_color)
        
    elif mode == uint(1):
        # Grayscale
        gray_val = grayscale_vec3(base_color)
        result : vec3 = vec3(gray_val, gray_val, gray_val)
        
    elif mode == uint(2):
        # Invert
        result = invert(base_color)
        
    elif mode == uint(3):
        # Posterize
        result = posterize(base_color, 5.0 + sin(time) * 3.0)
        
    elif mode == uint(4):
        # Brightness adjustment
        brightness_amount = 0.5 + 0.5 * sin(time * 0.5)
        result = brightness(base_color, brightness_amount * 0.5)
        
    elif mode == uint(5):
        # Contrast adjustment
        contrast_amount = 1.0 + sin(time) * 0.5
        result = contrast(base_color, contrast_amount)
        
    elif mode == uint(6):
        # Saturation adjustment
        sat_amount = 0.5 + 0.5 * sin(time * 0.7)
        result = saturation(base_color, sat_amount)
        
    elif mode == uint(7):
        # Hue shift (animated)
        hue_shift_amount = time * 50.0
        result : vec3 = hue_shift(base_color, hue_shift_amount)
        
    elif mode == uint(8):
        # Convert to HSV and back with modified saturation
        hsv : vec3 = rgb_to_hsv_vec3(base_color)
        result : vec3 = hsv_to_rgb_vec3(vec3(hsv.x, 0.8 + 0.2 * sin(time), hsv.z))
        
    elif mode == uint(9):
        # Convert to HSL and back with modified lightness
        hsl : vec3 = rgb_to_hsl_vec3(base_color)
        result : vec3 = hsl_to_rgb_vec3(vec3(hsl.x, hsl.y, 0.5 + 0.3 * sin(time * 0.8)))
        
    elif mode == uint(10):
        # Heat map effect
        t_val = (sin(dist1 * 20.0 - time * 2.0) + 1.0) * 0.5
        result : vec3 = heat_map(t_val)
        
    elif mode == uint(11):
        # Apply tint (cyan tint)
        cyan_tint : vec3 = vec3(0.0, 1.0, 1.0)
        tint_amount = 0.5 + 0.5 * sin(time * 0.6)
        result : vec3 = apply_tint(base_color, cyan_tint, tint_amount)
        
    elif mode == uint(12):
        # Color threshold
        threshold_val = 0.5 + 0.3 * sin(time)
        result : vec3 = threshold_color(base_color, threshold_val)
                
    elif mode == uint(13):
        # Additive blend
        r = sin(time) * 0.3
        g = sin(time + 1.0) * 0.3
        b = sin(time + 2.0) * 0.3
        result : vec3 = blend_add(base_color, vec3(r,g,b), 0.5)
        
    elif mode == uint(14):
        # Difference blend
        blend_color : vec3 = vec3(
            0.7 + 0.3 * sin(time * 0.5),
            0.7 + 0.3 * sin(time * 0.7),
            0.7 + 0.3 * sin(time * 0.9)
        )
        result : vec3 = blend_difference(base_color, blend_color, 0.6)
        
    else:
        # Default: original base color
        result : vec3 = base_color
        
    p3d_FragColor = vec4(result.r, result.g, result.b, 1.0)

# Compile shaders
vertex, vertex_info = engine.compile_shader(vertex_shader)
fragment, fragment_info = engine.compile_shader(fragment_shader, debug=True)
shader = Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)
node.setShader(shader)

# Set initial uniform values
node.setShaderInput("mode", 0)
node.setShaderInput("time", 0.0)

# Create text node to display current mode
mode_names = [
    "Sepia",
    "Grayscale",
    "Invert",
    "Posterize",
    "Brightness",
    "Contrast",
    "Saturation",
    "Hue Shift",
    "HSV Adjust",
    "HSL Adjust",
    "Heat Map",
    "Cyan Tint",
    "Threshold",
    "Additive Blend",
    "Difference Blend",
    "Original"
]

text_node = TextNode("mode_text")
text_node.setText("Mode 0: Rainbow Gradient")
text_node.setTextColor(1, 1, 1, 1)
text_node.setShadow(0.05, 0.05)
text_node.setShadowColor(0, 0, 0, 1)
text_np = aspect2d.attachNewNode(text_node)
text_np.setScale(0.07)
text_np.setPos(-1.2, 0, 0.9)

# Variables for mode switching
current_mode = 0
mode_count = len(mode_names)

# Task to update time and mode
def update_shader(task):
    global current_mode
    
    # Update time uniform
    node.setShaderInput("time", task.time)
    
    # Check for key presses to change mode
    if app.mouseWatcherNode.is_button_down('space'):
        current_mode = (current_mode + 1) % mode_count
        node.setShaderInput("mode", current_mode)
        
        # Update text display
        text_node.setText(f"Mode {current_mode}: {mode_names[current_mode]}")
        
        # Wait a bit before allowing another mode change
        taskMgr.doMethodLater(0.3, lambda task: None, 'mode_delay')
    
    # Also change mode automatically every 5 seconds
    if int(task.time) % 0.5 == 0 and int(task.time) != int(task.time - app.taskMgr.globalClock.getDt()):
        current_mode = (current_mode + 1) % mode_count
        node.setShaderInput("mode", current_mode)
        text_node.setText(f"Mode {current_mode}: {mode_names[current_mode]}")
    
    return task.cont

# Add the update task
app.taskMgr.add(update_shader, "update_shader")

# Instructions text
instructions = TextNode("instructions")
instructions.setText("Press SPACE to cycle through color effects\nAuto-cycles every 5 seconds")
instructions.setTextColor(1, 1, 1, 1)
instructions.setShadow(0.05, 0.05)
instructions.setShadowColor(0, 0, 0, 1)
instructions_np = aspect2d.attachNewNode(instructions)
instructions_np.setScale(0.05)
instructions_np.setPos(-1.2, 0, 0.8)

app.run()