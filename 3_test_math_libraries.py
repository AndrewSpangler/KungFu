from direct.showbase.ShowBase import ShowBase
from panda3d.core import CardMaker, Shader, TextNode, Vec4, Vec3, WindowProperties
import kungfu as kf
import math

app = ShowBase()
# props = WindowProperties()
# props.setCursorHidden(True)
# app.win.requestProperties(props)

engine = kf.GPUMath(app)

# Create a full-screen card
cm = CardMaker("card")
cm.setFrame(-1, 1, -1, 1)
node = app.aspect2d.attachNewNode(cm.generate())
node.setPos(0, 0, 0)

# Import the expanded math library
engine.import_file("./shader_libraries/math.py") 

@engine.shader('vertex')
def vertex_shader():
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex


@engine.shader('fragment', {
    'mode'       : ('uint',  'readonly'),
    'time'       : ('float', 'readonly'),
    'resolution' : ('vec3',  'readonly'),
    'mouse'      : ('vec4',  'readonly')
})
def fragment_shader():
    # Normalize coordinates to [0, 1]
    uv_x = gl_FragCoord.x / resolution.x
    uv_y = gl_FragCoord.y / resolution.y
    
    # Center coordinates to [-1, 1]
    x = uv_x * 2.0 - 1.0
    y = uv_y * 2.0 - 1.0
    
    # Mouse coordinates
    mouse_uv_x = mouse.x / resolution.x
    mouse_uv_y = mouse.y / resolution.y
    mouse_x = mouse_uv_x * 2.0 - 1.0
    mouse_y = mouse_uv_y * 2.0 - 1.0
    
    color : vec3 = vec3(0.0, 0.0, 0.0)
    
    if mode == uint(0):
        # Distance functions comparison
        euclidean_dist : float = dist(x, y)
        manhattan : float = manhattan_dist(x, y)
        chebyshev : float = chebyshev_dist(x, y)
        
        color = vec3(
            euclidean_dist * 0.5,
            manhattan * 0.3,
            chebyshev * 0.5
        )
    
    elif mode == uint(1):        
        # Different easing functions at different Y positions
        eased : float = 0.0
        if uv_y > 0.75:
            eased = ease_in_quad(uv_x)
        elif uv_y > 0.5:
            eased = ease_out_quad(uv_x)
        elif uv_y > 0.25:
            eased = ease_in_out_cubic(uv_x)
        else:
            eased = ease_in_out_sine(uv_x)
        
        # Draw the curve
        curve_y : float = eased
        diff : float = abs((uv_y - 0.5) - (curve_y - 0.5))
        intensity : float = 1.0
        if diff < 0.02:
            intensity = 0.0
        
        color = vec3(intensity, intensity * 0.5, intensity * 0.8)
    
    elif mode == uint(2):
        # Smooth step and smoother step comparison
        edge0 : float = 0.3
        edge1 : float = 0.7
        
        smooth_val : float = smooth_step(uv_x, edge0, edge1)
        smoother_val : float = smoother_step(uv_x, edge0, edge1)
        
        if uv_y > 0.5:
            color = vec3(smooth_val, smooth_val * 0.5, 0.0)
        else:
            color = vec3(0.0, smoother_val * 0.5, smoother_val)
    
    elif mode == uint(3): 
        # Vector rotation
        center : vec2 = vec2(0.0, 0.0)
        point : vec2 = vec2(x, y)
        
        # Rotate point around center
        rotated : vec2 = rotate_vec2(point - center, time) + center
        
        # Create pattern based on rotated coordinates
        pattern : float = sin(rotated.x * 10.0) * cos(rotated.y * 10.0)
        color = vec3(pattern * 0.5 + 0.5, 0.3, 0.6)
    
    elif mode == uint(4):
        # Bezier curves - simplified without loop
        # Define control points
        p0 : vec2 = vec2(-0.8, -0.5)
        p1 : vec2 = vec2(-0.2, 0.8)
        p2 : vec2 = vec2(0.2, -0.8)
        p3 : vec2 = vec2(0.8, 0.5)
        
        # Use the current UV as t parameter for visualization
        t_param : float = uv_x
        curve_point : vec2 = bezier_cubic(p0, p1, p2, p3, t_param)
        
        # Show the curve path
        point_dist : float = dist_vec2(vec2(x, y), curve_point)
        
        # Color based on distance to curve
        curve_intensity : float = 1.0
        if point_dist < 0.02:
            curve_intensity = 0.0
        color = vec3(curve_intensity, curve_intensity * 0.7, 0.0)
        
        # Draw control points
        if dist_vec2(vec2(x, y), p0) < 0.05:
            color = vec3(1.0, 0.0, 0.0)
        elif dist_vec2(vec2(x, y), p1) < 0.05:
            color = vec3(0.0, 1.0, 0.0)
        elif dist_vec2(vec2(x, y), p2) < 0.05:
            color = vec3(0.0, 0.0, 1.0)
        elif dist_vec2(vec2(x, y), p3) < 0.05:
            color = vec3(1.0, 1.0, 0.0)
    
    elif mode == uint(5):
        # SDF shapes
        circle_sdf : float = sdf_circle(vec2(x - 0.5, y), 0.3)
        box_sdf : float = sdf_box(vec2(x + 0.5, y), vec2(0.25, 0.35))
        line_sdf : float = sdf_line(vec2(x, y), vec2(-0.3, -0.5), vec2(0.3, 0.5), 0.02)
        
        # Color based on SDFs (filled shapes)
        circle_col : vec3 = vec3(0.0, 0.0, 0.0)
        if circle_sdf <= 0.0:
            circle_col = vec3(1.0, 0.0, 0.0)
            
        box_col : vec3 = vec3(0.0, 0.0, 0.0)
        if box_sdf <= 0.0:
            box_col = vec3(0.0, 1.0, 0.0)
            
        line_col : vec3 = vec3(0.0, 0.0, 0.0)
        if line_sdf <= 0.0:
            line_col = vec3(0.0, 0.0, 1.0)
        
        color = circle_col + box_col + line_col
    
    elif mode == uint(6):
        # Bounds checking
        bound1_a : vec2 = vec2(-0.8, -0.5)
        bound1_b : vec2 = vec2(-0.2, 0.5)
        
        bound2_center : vec2 = vec2(0.5, 0.0)
        bound2_radius : float = 0.4
        
        check_point : vec2 = vec2(x, y)
        
        if in_bounds(check_point, bound1_a, bound1_b):
            color = vec3(1.0, 0.0, 0.0)
        elif in_circle(check_point, bound2_center, bound2_radius):
            color = vec3(0.0, 1.0, 0.0)
        else:
            color = vec3(0.1, 0.1, 0.2)
    
    elif mode == uint(7):
        # Pseudo random noise
        grid_x : float = floor(x * 10.0)
        grid_y : float = floor(y * 10.0)
        
        noise_val : float = pseudo_random(grid_x + time * 0.1, grid_y)
        color = vec3(noise_val, noise_val * 0.7, noise_val * 0.5)
    
    elif mode == uint(8):
        # Map range example - create gradient with custom mapping
        mapped : float = map_range(uv_x, 0.0, 1.0, -1.0, 1.0)
        wave : float = sin(mapped * 3.14159 * 4.0 + time) * 0.5 + 0.5
        
        color = vec3(wave, wave * uv_y, 1.0 - wave)
    
    elif mode == uint(9):
        # Lerp and interpolation
        c1 : vec3 = vec3(1.0, 0.0, 0.0)
        c2 : vec3 = vec3(0.0, 1.0, 0.0)
        c3 : vec3 = vec3(0.0, 0.0, 1.0)
        
        t : float = 0.0
        if uv_x < 0.5:
            t = uv_x * 2.0
            color = lerp_vec3(c1, c2, t)
        else:
            t = (uv_x - 0.5) * 2.0
            color = lerp_vec3(c2, c3, t)
    
    elif mode == uint(10):
        center_point : vec2 = vec2(0.0, 0.0)
        to_point : vec2 = vec2(x, y) - center_point
        
        # Create perpendicular vector
        perp : vec2 = perpendicular(to_point)
        
        # Use perpendicular for coloring
        angle : float = atan(perp.y, perp.x)
        hue : float = angle / 6.28318 + 0.5
        
        color = vec3(
            sin(hue * 6.28318) * 0.5 + 0.5,
            sin(hue * 6.28318 + 2.094) * 0.5 + 0.5,
            sin(hue * 6.28318 + 4.189) * 0.5 + 0.5
        )
    
    elif mode == uint(11):
        # Wrapping and snapping
        wrapped_x : float = wrap(x + time * 0.5, 2.0) - 1.0
        wrapped_y : float = wrap(y + time * 0.3, 2.0) - 1.0
        
        snapped_x : float = snap_to_grid(wrapped_x, 0.2)
        snapped_y : float = snap_to_grid(wrapped_y, 0.2)
        
        grid_val : float = pseudo_random(snapped_x, snapped_y)
        color = vec3(grid_val, grid_val * 0.8, grid_val * 0.6)
    
    elif mode == uint(12):
        # Angle operations
        to_center : vec2 = vec2(0.0, 0.0) - vec2(x, y)
        to_mouse : vec2 = vec2(mouse_x, mouse_y) - vec2(x, y)
        
        angle_diff : float = angle_between_vec2(to_center, to_mouse)
        normalized_angle : float = normalize_angle(angle_diff)
        
        angle_val : float = (normalized_angle + 3.14159) / 6.28318
        color = vec3(angle_val, 1.0 - angle_val, 0.5)
    
    elif mode == uint(13):
        # Clamp visualization
        raw_value : float = x
        clamped : float = clamp_float(raw_value, -0.5, 0.5)
        val : float = 0.0
        # Show difference
        if uv_y > 0.5:
            val = (raw_value + 1.0) * 0.5
            color = vec3(val, 0.0, 0.0)
        else:
            val = (clamped + 1.0) * 0.5
            color = vec3(0.0, val, 0.0)
    
    elif mode == uint(14):
        # Multiple distance metrics comparison
        point1 : vec2 = vec2(0.5 + 0.3 * cos(time), 0.3 * sin(time))
        point2 : vec2 = vec2(-0.5 + 0.3 * cos(time * 1.3), 0.3 * sin(time * 0.7))
        
        dist_euclidean : float = dist_vec2(vec2(x, y), point1)
        dist_manhattan : float = manhattan_dist_vec2(vec2(x, y), point2)
        
        color = vec3(
            smoothstep(0.0, 0.5, 1.0 - dist_euclidean),
            smoothstep(0.0, 0.5, 1.0 - dist_manhattan),
            0.3
        )
    
    else:
        # Default: original distance demo
        distance1 : float = dist(x, y)
        distance2 : float = dist(x, y + 0.3)
        distance3 : float = dist(x - 0.3, y)
        
        color = vec3(
            distance1,
            0.7 * distance2,
            distance3 * 0.8
        )
    
    p3d_FragColor = vec4(color.r, color.g, color.b, 1.0)


def create_panda_shader(vert, frag) -> Shader:
    vertex, vertex_info = engine.compile_shader(vert, debug=True)
    fragment, fragment_info = engine.compile_shader(frag, debug=True)
    return Shader.make(Shader.SL_GLSL, vertex=vertex, fragment=fragment)

shader = create_panda_shader(vertex_shader, fragment_shader)
node.setShader(shader)

# Set initial uniform values
node.setShaderInput("mode", 0)
node.setShaderInput("time", 0.0)

# Create text node to display current mode
mode_names = [
    "Distance Functions",
    "Easing Functions",
    "Smooth Step vs Smoother Step",
    "Vector Rotation",
    "Bezier Curves",
    "SDF Shapes",
    "Bounds Checking",
    "Pseudo Random",
    "Map Range",
    "Linear Interpolation",
    "Vector Perpendicular",
    "Wrapping & Snapping",
    "Angle Operations (move mouse)",
    "Clamp Visualization",
    "Distance Metrics",
    "Original Demo"
]

text_node = TextNode("mode_text")
text_node.setText(f"Mode 0: {mode_names[0]}")
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
    
    mouse_node = app.mouseWatcherNode
    dt = app.taskMgr.globalClock.getDt()
    w, h = app.win.getXSize(), app.win.getYSize()
    
    node.setShaderInput("resolution", Vec3(w, h, 1.0))
    node.setShaderInput("time", task.time)
    node.setShaderInput("mode", current_mode)
    
    # Update mouse position
    if mouse_node and mouse_node.hasMouse():
        x = mouse_node.getMouseX()
        y = mouse_node.getMouseY()
        px, py = (x + 1) * 0.5 * w, (y + 1) * 0.5 * h
        node.setShaderInput("mouse", Vec4(px, py, 0, 0))
    else:
        node.setShaderInput("mouse", Vec4(w/2, h/2, 0, 0))
    
    # Auto-cycle modes every 5 seconds
    if (
        int(task.time) % 1 == 0
        and int(task.time) != int(task.time - dt)
        and task.time > 1.0  # Skip first second
    ):
        current_mode = (current_mode + 1) % mode_count
        node.setShaderInput("mode", current_mode)
        text_node.setText(f"Mode {current_mode}: {mode_names[current_mode]}")
    
    return task.cont

# Add the update task
app.taskMgr.add(update_shader, "update_shader")

# Instructions text
instructions = TextNode("instructions")
instructions.setText("Press SPACE to manually cycle through demos\nAuto-cycles every 5 seconds")
instructions.setTextColor(1, 1, 1, 1)
instructions.setShadow(0.05, 0.05)
instructions.setShadowColor(0, 0, 0, 1)
instructions_np = aspect2d.attachNewNode(instructions)
instructions_np.setScale(0.05)
instructions_np.setPos(-1.2, 0, 0.8)

# Manual mode switching
def cycle_mode():
    global current_mode
    current_mode = (current_mode + 1) % mode_count
    node.setShaderInput("mode", current_mode)
    text_node.setText(f"Mode {current_mode}: {mode_names[current_mode]}")

app.accept("space", cycle_mode)

app.run()