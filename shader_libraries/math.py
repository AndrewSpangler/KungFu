import kungfu as kf
import engine

# ============================================================================
# DISTANCE & GEOMETRY

@engine.function({
    'a' : kf.GLTypes.float,
    'b' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def dist(
    a   : kf.GLTypes.float,
    b   : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """2D Euclidean distance from origin"""
    return sqrt(a * a + b * b)


@engine.function({
    'p1' : kf.GLTypes.vec2,
    'p2' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.float)
def dist_vec2(
    p1  : kf.GLTypes.vec2,
    p2  : kf.GLTypes.vec2,
) -> kf.GLTypes.float:
    """Distance between two 2D points"""
    dx : float = p2.x - p1.x
    dy : float = p2.y - p1.y
    return sqrt(dx * dx + dy * dy)


@engine.function({
    'p1' : kf.GLTypes.vec3,
    'p2' : kf.GLTypes.vec3,
},  return_type=kf.GLTypes.float)
def dist_vec3(
    p1  : kf.GLTypes.vec3,
    p2  : kf.GLTypes.vec3,
) -> kf.GLTypes.float:
    """Distance between two 3D points"""
    diff : vec3 = p2 - p1
    return sqrt(dot(diff, diff))


@engine.function({
    'x' : kf.GLTypes.float,
    'y' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def manhattan_dist(
    x   : kf.GLTypes.float,
    y   : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Manhattan distance (L1 norm)"""
    return abs(x) + abs(y)


@engine.function({
    'p1' : kf.GLTypes.vec2,
    'p2' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.float)
def manhattan_dist_vec2(
    p1  : kf.GLTypes.vec2,
    p2  : kf.GLTypes.vec2,
) -> kf.GLTypes.float:
    """Manhattan distance between two 2D points"""
    return abs(p2.x - p1.x) + abs(p2.y - p1.y)


@engine.function({
    'x' : kf.GLTypes.float,
    'y' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def chebyshev_dist(
    x   : kf.GLTypes.float,
    y   : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Chebyshev distance (L-infinity norm)"""
    return max(abs(x), abs(y))


# ============================================================================
# BOUNDS CHECKING

@engine.function({
    'position'  : kf.GLTypes.vec2,
    'bound_a'   : kf.GLTypes.vec2,
    'bound_b'   : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.bool)
def in_bounds(
    position    : kf.GLTypes.vec2,
    bound_a     : kf.GLTypes.vec2,
    bound_b     : kf.GLTypes.vec2,
) -> kf.GLTypes.bool:
    """Check if position is within rectangular bounds"""
    return (
        min(bound_a.x, bound_b.x) <= position.x
        and position.x <= max(bound_a.x, bound_b.x)
        and min(bound_a.y, bound_b.y) <= position.y
        and position.y <= max(bound_a.y, bound_b.y)
    )


@engine.function({
    'point'     : kf.GLTypes.vec2,
    'center'    : kf.GLTypes.vec2,
    'radius'    : kf.GLTypes.float,
},  return_type=kf.GLTypes.bool)
def in_circle(
    point   : kf.GLTypes.vec2,
    center  : kf.GLTypes.vec2,
    radius  : kf.GLTypes.float,
) -> kf.GLTypes.bool:
    """Check if point is inside circle"""
    return dist_vec2(point, center) <= radius


@engine.function({
    'point'     : kf.GLTypes.vec3,
    'center'    : kf.GLTypes.vec3,
    'radius'    : kf.GLTypes.float,
},  return_type=kf.GLTypes.bool)
def in_sphere(
    point   : kf.GLTypes.vec3,
    center  : kf.GLTypes.vec3,
    radius  : kf.GLTypes.float,
) -> kf.GLTypes.bool:
    """Check if point is inside sphere"""
    return dist_vec3(point, center) <= radius


# ============================================================================
# CLAMPING & MAPPING

@engine.function({
    'value' : kf.GLTypes.float,
    'min_val' : kf.GLTypes.float,
    'max_val' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def clamp_float(
    value   : kf.GLTypes.float,
    min_val : kf.GLTypes.float,
    max_val : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))


@engine.function({
    'value'     : kf.GLTypes.float,
    'in_min'    : kf.GLTypes.float,
    'in_max'    : kf.GLTypes.float,
    'out_min'   : kf.GLTypes.float,
    'out_max'   : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def map_range(
    value   : kf.GLTypes.float,
    in_min  : kf.GLTypes.float,
    in_max  : kf.GLTypes.float,
    out_min : kf.GLTypes.float,
    out_max : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Map value from one range to another"""
    t : float = (value - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


@engine.function({
    'value' : kf.GLTypes.float,
    'edge0' : kf.GLTypes.float,
    'edge1' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def smooth_step(
    value : kf.GLTypes.float,
    edge0 : kf.GLTypes.float,
    edge1 : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Smooth Hermite interpolation between 0 and 1"""
    t : float = clamp_float((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@engine.function({
    'value' : kf.GLTypes.float,
    'edge0' : kf.GLTypes.float,
    'edge1' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def smoother_step(
    value : kf.GLTypes.float,
    edge0 : kf.GLTypes.float,
    edge1 : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Even smoother Hermite interpolation (6t^5 - 15t^4 + 10t^3)"""
    t : float = clamp_float((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ============================================================================
# INTERPOLATION

@engine.function({
    'a' : kf.GLTypes.float,
    'b' : kf.GLTypes.float,
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def lerp(
    a : kf.GLTypes.float,
    b : kf.GLTypes.float,
    t : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Linear interpolation"""
    return a + t * (b - a)


@engine.function({
    'a' : kf.GLTypes.vec2,
    'b' : kf.GLTypes.vec2,
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.vec2)
def lerp_vec2(
    a : kf.GLTypes.vec2,
    b : kf.GLTypes.vec2,
    t : kf.GLTypes.float,
) -> kf.GLTypes.vec2:
    """Linear interpolation for vec2"""
    return a + t * (b - a)


@engine.function({
    'a' : kf.GLTypes.vec3,
    'b' : kf.GLTypes.vec3,
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.vec3)
def lerp_vec3(
    a : kf.GLTypes.vec3,
    b : kf.GLTypes.vec3,
    t : kf.GLTypes.float,
) -> kf.GLTypes.vec3:
    """Linear interpolation for vec3"""
    return a + t * (b - a)


@engine.function({
    'p0' : kf.GLTypes.vec2,
    'p1' : kf.GLTypes.vec2,
    'p2' : kf.GLTypes.vec2,
    't'  : kf.GLTypes.float,
},  return_type=kf.GLTypes.vec2)
def bezier_quadratic(
    p0 : kf.GLTypes.vec2,
    p1 : kf.GLTypes.vec2,
    p2 : kf.GLTypes.vec2,
    t  : kf.GLTypes.float,
) -> kf.GLTypes.vec2:
    """Quadratic Bezier curve"""
    one_minus_t : float = 1.0 - t
    return one_minus_t * one_minus_t * p0 + 2.0 * one_minus_t * t * p1 + t * t * p2


@engine.function({
    'p0' : kf.GLTypes.vec2,
    'p1' : kf.GLTypes.vec2,
    'p2' : kf.GLTypes.vec2,
    'p3' : kf.GLTypes.vec2,
    't'  : kf.GLTypes.float,
},  return_type=kf.GLTypes.vec2)
def bezier_cubic(
    p0 : kf.GLTypes.vec2,
    p1 : kf.GLTypes.vec2,
    p2 : kf.GLTypes.vec2,
    p3 : kf.GLTypes.vec2,
    t  : kf.GLTypes.float,
) -> kf.GLTypes.vec2:
    """Cubic Bezier curve"""
    one_minus_t : float = 1.0 - t
    a : float = one_minus_t * one_minus_t * one_minus_t
    b : float = 3.0 * one_minus_t * one_minus_t * t
    c : float = 3.0 * one_minus_t * t * t
    d : float = t * t * t
    return a * p0 + b * p1 + c * p2 + d * p3


# ============================================================================
# VECTOR OPERATIONS

@engine.function({
    'v' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.float)
def magnitude_vec2(v : kf.GLTypes.vec2) -> kf.GLTypes.float:
    """Get magnitude/length of 2D vector"""
    return sqrt(dot(v, v))


@engine.function({
    'v' : kf.GLTypes.vec3,
},  return_type=kf.GLTypes.float)
def magnitude_vec3(v : kf.GLTypes.vec3) -> kf.GLTypes.float:
    """Get magnitude/length of 3D vector"""
    return sqrt(dot(v, v))


@engine.function({
    'v' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.vec2)
def normalize_vec2(v : kf.GLTypes.vec2) -> kf.GLTypes.vec2:
    """Normalize 2D vector to unit length"""
    mag : float = magnitude_vec2(v)
    if mag > 0.0:
        return v / mag
    return vec2(0.0, 0.0)


@engine.function({
    'v' : kf.GLTypes.vec3,
},  return_type=kf.GLTypes.vec3)
def normalize_vec3(v : kf.GLTypes.vec3) -> kf.GLTypes.vec3:
    """Normalize 3D vector to unit length"""
    mag : float = magnitude_vec3(v)
    if mag > 0.0:
        return v / mag
    return vec3(0.0, 0.0, 0.0)


@engine.function({
    'v' : kf.GLTypes.vec2,
    'angle' : kf.GLTypes.float,
},  return_type=kf.GLTypes.vec2)
def rotate_vec2(
    v     : kf.GLTypes.vec2,
    angle : kf.GLTypes.float,
) -> kf.GLTypes.vec2:
    """Rotate 2D vector by angle (in radians)"""
    c : float = cos(angle)
    s : float = sin(angle)
    return vec2(
        v.x * c - v.y * s,
        v.x * s + v.y * c
    )


@engine.function({
    'v1' : kf.GLTypes.vec2,
    'v2' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.float)
def angle_between_vec2(
    v1 : kf.GLTypes.vec2,
    v2 : kf.GLTypes.vec2,
) -> kf.GLTypes.float:
    """Get angle between two 2D vectors"""
    return atan(v2.y - v1.y, v2.x - v1.x)


@engine.function({
    'v' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.vec2)
def perpendicular(v : kf.GLTypes.vec2) -> kf.GLTypes.vec2:
    """Get perpendicular vector (rotated 90 degrees)"""
    return vec2(-v.y, v.x)


@engine.function({
    'incident' : kf.GLTypes.vec2,
    'normal'   : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.vec2)
def reflect_vec2(
    incident : kf.GLTypes.vec2,
    normal   : kf.GLTypes.vec2,
) -> kf.GLTypes.vec2:
    """Reflect vector across normal"""
    return incident - 2.0 * dot(incident, normal) * normal


@engine.function({
    'incident' : kf.GLTypes.vec3,
    'normal'   : kf.GLTypes.vec3,
},  return_type=kf.GLTypes.vec3)
def reflect_vec3(
    incident : kf.GLTypes.vec3,
    normal   : kf.GLTypes.vec3,
) -> kf.GLTypes.vec3:
    """Reflect 3D vector across normal"""
    return incident - 2.0 * dot(incident, normal) * normal


# ============================================================================
# TRIGONOMETRY & ANGLES

@engine.function({
    'degrees' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def deg_to_rad(degrees : kf.GLTypes.float) -> kf.GLTypes.float:
    """Convert degrees to radians"""
    return degrees * 0.0174533


@engine.function({
    'radians' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def rad_to_deg(radians : kf.GLTypes.float) -> kf.GLTypes.float:
    """Convert radians to degrees"""
    return radians * 57.2958


@engine.function({
    'angle' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def normalize_angle(angle : kf.GLTypes.float) -> kf.GLTypes.float:
    """Normalize angle to [-PI, PI]"""
    PI : float = 3.14159265359
    TWO_PI : float = 6.28318530718
    result : float = angle
    # Normalize to [0, TWO_PI]
    result = result - floor(result / TWO_PI) * TWO_PI
    # Shift to [-PI, PI]
    if result > PI:
        result = result - TWO_PI
    return result


# ============================================================================
# EASING FUNCTIONS

@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_in_quad(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Quadratic ease-in"""
    return t * t


@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_out_quad(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Quadratic ease-out"""
    return t * (2.0 - t)


@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_in_out_quad(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Quadratic ease-in-out"""
    if t < 0.5:
        return 2.0 * t * t
    else:
        return -1.0 + (4.0 - 2.0 * t) * t


@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_in_cubic(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Cubic ease-in"""
    return t * t * t


@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_out_cubic(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Cubic ease-out"""
    t_minus = t - 1.0
    return t_minus * t_minus * t_minus + 1.0




@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_in_sine(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Sine ease-in"""
    return 1.0 - cos(t * 1.5708)


@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_out_sine(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Sine ease-out"""
    return sin(t * 1.5708)


@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_in_out_sine(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Sine ease-in-out"""
    return 0.5 * (1.0 - cos(3.14159 * t))


@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_in_out_cubic(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Cubic ease-in-out"""
    result : float = 0.0
    if t < 0.5:
        result = 4.0 * t * t * t
    else:
        t_minus : float = (2.0 * t - 2.0)
        result = 0.5 * t_minus * t_minus * t_minus + 1.0
    return result

@engine.function({
    't' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def ease_in_out_sine(t : kf.GLTypes.float) -> kf.GLTypes.float:
    """Sine ease-in-out"""
    return 0.5 * (1.0 - cos(3.14159 * t))


# ============================================================================
# NOISE & RANDOMNESS

@engine.function({
    'x' : kf.GLTypes.float,
    'y' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def pseudo_random(
    x : kf.GLTypes.float,
    y : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Pseudo-random hash function for 2D coordinates"""
    return fract(sin(dot(vec2(x, y), vec2(12.9898, 78.233))) * 43758.5453)


@engine.function({
    'v' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.float)
def pseudo_random_vec2(v : kf.GLTypes.vec2) -> kf.GLTypes.float:
    """Pseudo-random hash function for vec2"""
    return fract(sin(dot(v, vec2(12.9898, 78.233))) * 43758.5453)


@engine.function({
    'v' : kf.GLTypes.vec3,
},  return_type=kf.GLTypes.float)
def pseudo_random_vec3(v : kf.GLTypes.vec3) -> kf.GLTypes.float:
    """Pseudo-random hash function for vec3"""
    return fract(sin(dot(v, vec3(12.9898, 78.233, 45.5432))) * 43758.5453)


# ============================================================================
# SHAPE FUNCTIONS (SDF - Signed Distance Functions)

@engine.function({
    'p'      : kf.GLTypes.vec2,
    'radius' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def sdf_circle(
    p      : kf.GLTypes.vec2,
    radius : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Signed distance to circle"""
    return magnitude_vec2(p) - radius


@engine.function({
    'p'    : kf.GLTypes.vec2,
    'size' : kf.GLTypes.vec2,
},  return_type=kf.GLTypes.float)
def sdf_box(
    p    : kf.GLTypes.vec2,
    size : kf.GLTypes.vec2,
) -> kf.GLTypes.float:
    """Signed distance to box"""
    d : vec2 = abs(p) - size
    return magnitude_vec2(max(d, vec2(0.0, 0.0))) + min(max(d.x, d.y), 0.0)


@engine.function({
    'p'       : kf.GLTypes.vec2,
    'a'       : kf.GLTypes.vec2,
    'b'       : kf.GLTypes.vec2,
    'radius'  : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def sdf_line(
    p      : kf.GLTypes.vec2,
    a      : kf.GLTypes.vec2,
    b      : kf.GLTypes.vec2,
    radius : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Signed distance to line segment"""
    pa : vec2 = p - a
    ba : vec2 = b - a
    h : float = clamp_float(dot(pa, ba) / dot(ba, ba), 0.0, 1.0)
    return magnitude_vec2(pa - ba * h) - radius


# ============================================================================
# UTILITY FUNCTIONS

@engine.function({
    'value' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def sign_non_zero(value : kf.GLTypes.float) -> kf.GLTypes.float:
    """Sign function that returns 1.0 for 0 instead of 0"""
    if value >= 0.0:
        return 1.0
    return -1.0


@engine.function({
    'value' : kf.GLTypes.float,
    'step_size' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def snap_to_grid(
    value     : kf.GLTypes.float,
    step_size : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Snap value to nearest grid point"""
    return floor(value / step_size + 0.5) * step_size


@engine.function({
    'value' : kf.GLTypes.float,
    'period' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def wrap(
    value  : kf.GLTypes.float,
    period : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Wrap value to period [0, period)"""
    return value - floor(value / period) * period


@engine.function({
    'value' : kf.GLTypes.float,
    'threshold' : kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def step_threshold(
    value     : kf.GLTypes.float,
    threshold : kf.GLTypes.float,
) -> kf.GLTypes.float:
    """Step function: 0 if value < threshold, 1 otherwise"""
    result : float = 1.0
    if value < threshold:
        result = 0.0
    return result


@engine.function({
    'a' : kf.GLTypes.float,
    'b' : kf.GLTypes.float,
    'tolerance' : kf.GLTypes.float,
},  return_type=kf.GLTypes.bool)
def approximately_equal(
    a         : kf.GLTypes.float,
    b         : kf.GLTypes.float,
    tolerance : kf.GLTypes.float,
) -> kf.GLTypes.bool:
    """Check if two floats are approximately equal"""
    return abs(a - b) < tolerance