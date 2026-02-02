# KungFU

## Table of Contents
- [Libraries](#libraries)
  - [strings.py](#stringspy)
    - [Char Handling](#char-handling)
    - [String Handling](#string-handling)
    - [Conversion](#conversion)
    - [Fragment Shader Helper](#fragment-shader-helper)
    - [Engine Helpers](#engine-helpers)
  - [math.py](#mathpy)
    - [Distance and Geometry](#distance-and-geometry)
    - [Bounds Checking](#bounds-checking)
    - [Clamping and Mapping](#clamping-and-mapping)
    - [Interpolation](#interpolation)
    - [Vector Operations](#vector-operations)
    - [Easing Functions](#easing-functions)
    - [Noise and RNG](#noise-and-rng)
    - [SDF Shapes](#sdf-shapes)
    - [Utility Functions](#utility-functions)











## Libraries
Below are the included libraries, and their signatures.
These signatures are in a pseudo-code format for easy reference.
See the library files for full decorators / typehinting. 

### strings.py
#### Char handling
```py
def is_whitespace(char_code: uint) -> bool:
    """Check if character is whitespace (space, tab, newline, etc.)"""

def is_digit(char_code: uint) -> bool:
    """Check if character is a digit (0-9)"""

def is_alpha(char_code: uint) -> bool:
    """Check if character is alphabetic (A-Z or a-z)"""

def is_alnum(char_code: uint) -> bool:
    """Check if character is alphanumeric (A-Z, a-z, or 0-9)"""

def is_upper(char_code: uint) -> bool:
    """Check if character is uppercase (A-Z)"""

def is_lower(char_code: uint) -> bool:
    """Check if character is lowercase (a-z)"""

def to_upper(char_code: uint) -> uint:
    """Convert character to uppercase"""

def to_lower(char_code: uint) -> uint:
    """Convert character to lowercase"""
```

#### String Handling
```py
def string_length(str_array: uint[255]) -> uint:
    """Get the length of a string (up to first terminator or array length)"""

def is_empty_string(str_array: uint[255]) -> bool:
    """Check if string is empty (first character is terminator)"""

def string_equal(str1: uint[255], str2: uint[255]) -> bool:
    """Compare two strings for equality"""

def string_compare(str1: uint[255], str2: uint[255]) -> int:
    """
    Compare two strings lexicographically.
    Returns:
        0 if equal
        <0 if str1 < str2
        >0 if str1 > str2
    """

def string_to_upper(str_array: uint[255]) -> uint[255]:
    """Convert string to uppercase"""

def string_to_lower(str_array: uint[255]) -> uint[255]:
    """Convert string to lowercase"""

def string_reverse(str_array: uint[255]) -> uint[255]:
    """Reverse a string"""

def string_find(str_array: uint[255], char_code: uint) -> uint:
    """Find first occurrence of character. Returns index or max uint if not found"""

def string_rfind(str_array: uint[255], char_code: uint) -> uint:
    """Find last occurrence of character. Returns index or max uint if not found"""

def string_count(str_array: uint[255], char_code: uint) -> uint:
    """Count occurrences of character"""

def string_replace_char(str_array: uint[255], old_char_code: uint, new_char_code: uint) -> uint[255]:
    """Replace all occurrences of old_char_code with new_char_code"""

def string_concat(str1: uint[255], str2: uint[255]) -> uint[255]:
    """Concatenate two strings"""

def string_substring(str_array: uint[255], start: uint, length: uint) -> uint[255]:
    """Extract substring by index and length"""
```

#### Conversion
```py
def string_to_int(str_array: uint[255]) -> int:
    """Convert string to integer"""

def string_to_float(string_to_float: uint[255]) -> float:
    """Convert string to float"""

def string_is_numeric(str_array: uint[255]) -> bool:
    """Check if string contains only digits and optionally one dot and minus sign"""

def string_hash(str_array: uint[255]) -> uint:
    """Simple string hash function (djb2 algorithm)"""

def int_to_string(value: int) -> uint[255]:
    """Convert integer to string"""

def float_to_string(value: float, precision: uint) -> uint[255]:
    """Convert float to string with given precision"""
```

#### Fragment Shader Helper
```py
def render_text(
    text_array:      uint[{255}],
    color:           vec4,
    cols:            uint,
    rows:            uint,
    x:               float,
    y:               float,
    charmap_texture: sampler2D,
    char_uvs:        vec4[48]
) -> vec4:
    """Render text using character map texture"""
```

#### Engine Helpers    
```py
def encode_string(val: str) -> list:
    """
    Helper function to encode strings to a list of integers for Panda3D
    Pads with -1 (which is 0xFFFFFFFF in two's complement / unsigned interpretation)
    This works with Panda3D's C++ backend which uses signed int conversion
    """

def encode_string_glsl(val: str) -> str:
    """Helper function to encode strings to GLSL format"""


"""Engine helpers available with:
- engine.encode_string_numpy
- engine.encode_string_glsl
"""
```

### math.py

#### Distance and Geometry
```py
def dist(a:float, b:float) -> float:
    """2D Euclidean distance from origin"""

def dist_vec2(p1: vec2, p2: vec2) -> float:
    """Distance between two 2D points"""

def dist_vec3(p1: vec3, p2: vec3) -> float:
    """Distance between two 3D points"""

def manhattan_dist_vec2(p1: vec2, p2: vec2) ->float:
    """Manhattan distance between two 2D points"""

def chebyshev_dist(x: float, y: float) -> float:
    """Chebyshev distance (L-infinity norm)"""
```

#### Bounds Checking
```py
def in_bounds(position:vec2, bound_a:vec2, bound_b:vec2) -> bool:
    """Check if position is within rectangular bounds"""

def in_circle(point: vec2, center: vec2, radius: float) -> bool:
    """Check if point is inside circle"""

def in_sphere( point: vec3, center: vec3, radius: float) -> bool:
    """Check if point is inside sphere"""
```

#### Clamping and Mapping
```py
def clamp_float(value: float, min_val: float, max_val: float) -> float:
    """Clamp floating value between min and max"""

def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    """Map value from one range to another"""

def smooth_step(value: float, edge0: float, edge1: float) -> float:
    """Hermite interpolation between 0 and 1"""

def smoother_step(value: float, edge0: float, edge1: float) -> float:
    """Smoother Hermite interpolation"""
```

#### Interpolation
```py
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation"""

def lerp_vec2(a: vec2, b: vec2, t: float) -> vec2:
    """Linear interpolation with vec2s"""

def lerp_vec3(a: vec3, b: vec3, t: float) -> vec3:
    """Linear interpolation for vec3s"""

def bezier_quadratic(p0: vec2, p1: vec2, p2: vec2, t: float) -> vec2:
    """Quadratic Bezier curve"""

def bezier_cubic(p0: vec2, p1: vec2, p2: vec2, p3: vec2, t: float) -> vec2:
    """Cubic Bezier curve"""
```

#### Vector Operations
```py
def magnitude_vec2(v: vec2) -> float:
    """Get magnitude/length of 2D vector"""

def magnitude_vec3(v: vec3) -> float:
    """Get magnitude/length of 3D vector"""

def normalize_vec2(v: vec2) -> vec2:
    """Normalize 2D vector to unit length"""

def normalize_vec3(v: vec3) -> vec3:
    """Normalize 3D vector to unit length"""

def rotate_vec2(v: vec2, angle: float) -> vec2:
    """Rotate 2D vector by angle (in radians)"""

def angle_between_vec2(v: vec2, v2: vec2) -> float:
    """Get angle between two 2D vectors"""

def perpendicular(v: vec2) -> vec2:
    """Get perpendicular vector (rotated 90 degrees)"""

def reflect_vec2(incident: vec2, normal: vec2) -> vec2:
    """Reflect vector across normal"""

def reflect_vec3(incident: vec3, normal: vec3) -> vec3:
    """Reflect 3D vector across normal"""
```

#### Easing Functions
```py
def ease_in_quad(t: float) -> float:
    """Quadratic ease-in"""

def ease_out_quad(t: float) -> float:
    """Quadratic ease-out"""

def ease_in_out_quad(t: float) -> float:
    """Quadratic ease-in-out"""

def ease_in_cubic(t: float) -> float:
    """Cubic ease-in"""

def ease_out_cubic(t: float) -> float:
    """Cubic ease-out"""

def ease_in_sine(t: float) -> float:
    """Sine ease-in"""

def ease_out_sine(t: float) -> float:
    """Sine ease-out"""

def ease_in_out_sine(t: float) -> float:
    """Sine ease-in-out"""

def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out"""

def ease_in_out_sine(t: float) -> float:
    """Sine ease-in-out"""
```

#### Noise and RNG
```py
def pseudo_random(x: float, y: float) -> float:
    """Pseudo-random hash function for 2D coordinates"""

def pseudo_random_vec2(v: vec2) -> float:
    """Pseudo-random hash function for vec2"""

def pseudo_random_vec3(v: vec3) -> float:
    """Pseudo-random hash function for vec3"""
```

#### SDF Shapes
```py
def sdf_circle(p: vec2, radius: float) -> float:
    """Signed distance to circle"""

def sdf_box(p: vec2, size: vec2) -> float:
    """Signed distance to box"""

def sdf_line(p: vec2, a: vec2, b: vec2, radius: float) -> float:
    """Signed distance to line segment"""
```

#### Utility Functions
```py
def sign_non_zero(value : float) -> float:
    """Sign function that returns 1.0 for 0 instead of 0"""

def snap_to_grid(value: float, step_size: float) -> float:
    """Snap value to nearest grid point"""

def wrap(value: float, period: float) -> float:
    """Wrap value to period [0, period)"""

def step_threshold(value: float, threshold: float) -> float:
    """Step function: 0 if value < threshold, 1 otherwise"""

def approximately_equal(a: float, b: float, tolerance: float) -> bool:
    """Check if two floats are approximately equal"""
```