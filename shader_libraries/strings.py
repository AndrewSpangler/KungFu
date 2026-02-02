import os
import kungfu as kf
import engine
from panda3d.core import Vec4

CHARMAP_PATH = os.path.join("shader_libraries/assets/charmap.png")

CHARS = [
    [" ", "!", '"', "#", "$", "%", "&", "'", "<", ">", "*", "+", ",", "-", ".", "/" ],
    ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?" ],
    ["@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O" ],
    ["P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_"],
    ["'", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o" ],
    ["p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "␡"]
]
CHARS_FLAT = []
for row in CHARS: CHARS_FLAT.extend(row)

CHAR_TO_INDEX = {char: idx for idx, char in enumerate(CHARS_FLAT)}
CHAR_COUNT = 96
CHARS_PER_ROW = 16
CHAR_UV_LOOKUP = []
for i in range(CHAR_COUNT):
    col = i % int(CHARS_PER_ROW)
    row = i // int(CHARS_PER_ROW)
    u = col / float(CHARS_PER_ROW)
    v = 1.0 - (row + 1) / 6.0
    CHAR_UV_LOOKUP.extend([u, v])

PACKED_UVS = []
for i in range(0, len(CHAR_UV_LOOKUP), 4):
    u0 = CHAR_UV_LOOKUP[i]
    v0 = CHAR_UV_LOOKUP[i+1]
    u1 = CHAR_UV_LOOKUP[i+2] if (i+2) < len(CHAR_UV_LOOKUP) else 0.0
    v1 = CHAR_UV_LOOKUP[i+3] if (i+3) < len(CHAR_UV_LOOKUP) else 0.0
    PACKED_UVS.append(Vec4(u0, v0, u1, v1))

MAX_STRING_LENGTH = 255
STRING_TERMINATOR = 0xFFFFFFFF  # uint32 max value

# ============================================================================
# STRING LENGTH & VALIDATION

@engine.function({
    'str_array': f'uint[{255}]'
}, return_type=kf.GLTypes.uint)
def string_length(str_array: f'uint[{255}]') -> kf.GLTypes.uint:
    """Get the length of a string (up to first terminator or array length)"""
    length : uint = uint(0)
    found = False
    for i in range(255):
        if str_array[i] == uint(0xFFFFFFFF):  # 0xFFFFFFFF
            found = True
        if not found:
            length = length + uint(1)
    return length


@engine.function({
    'str_array': f'uint[{255}]'
}, return_type=kf.GLTypes.bool)
def is_empty_string(str_array: f'uint[{255}]') -> kf.GLTypes.bool:
    """Check if string is empty (first character is terminator)"""
    return str_array[0] == uint(0xFFFFFFFF)


# ============================================================================
# CHARACTER MANIPULATION

@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.bool)
def is_whitespace(char_code: kf.GLTypes.uint) -> kf.GLTypes.bool:
    """Check if character is whitespace (space, tab, newline, etc.)"""
    return char_code == uint(0) or char_code == uint(9) or char_code == uint(10) or char_code == uint(13)


@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.bool)
def is_digit(char_code: kf.GLTypes.uint) -> kf.GLTypes.bool:
    """Check if character is a digit (0-9)"""
    return char_code >= uint(16) and char_code <= uint(25)


@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.bool)
def is_alpha(char_code: kf.GLTypes.uint) -> kf.GLTypes.bool:
    """Check if character is alphabetic (A-Z or a-z)"""
    return (char_code >= uint(33) and char_code <= uint(58)) or (char_code >= uint(65) and char_code <= uint(90))


@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.bool)
def is_alnum(char_code: kf.GLTypes.uint) -> kf.GLTypes.bool:
    """Check if character is alphanumeric (A-Z, a-z, or 0-9)"""
    return is_alpha(char_code) or is_digit(char_code)


@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.bool)
def is_upper(char_code: kf.GLTypes.uint) -> kf.GLTypes.bool:
    """Check if character is uppercase (A-Z)"""
    return char_code >= uint(33) and char_code <= uint(58)


@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.bool)
def is_lower(char_code: kf.GLTypes.uint) -> kf.GLTypes.bool:
    """Check if character is lowercase (a-z)"""
    return char_code >= uint(65) and char_code <= uint(90)


@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.uint)
def to_upper(char_code: kf.GLTypes.uint) -> kf.GLTypes.uint:
    """Convert character to uppercase"""
    if char_code >= uint(65) and char_code <= uint(90):
        return char_code - uint(32)
    return char_code


@engine.function({
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.uint)
def to_lower(char_code: kf.GLTypes.uint) -> kf.GLTypes.uint:
    """Convert character to lowercase"""
    if char_code >= uint(33) and char_code <= uint(58):
        return char_code + uint(32)
    return char_code


# ============================================================================
# STRING COMPARISON

@engine.function({
    'str1': f'uint[{255}]',
    'str2': f'uint[{255}]'
}, return_type=kf.GLTypes.bool)
def string_equal(str1: f'uint[{255}]', str2: f'uint[{255}]') -> kf.GLTypes.bool:
    """Compare two strings for equality"""
    length1 = string_length(str1)
    length2 = string_length(str2)
    
    if length1 != length2:
        return false
    
    for i in range(255):
        if i >= length1:
            break
        if str1[i] != str2[i]:
            return false
    
    return true


@engine.function({
    'str1': f'uint[{255}]',
    'str2': f'uint[{255}]'
}, return_type=kf.GLTypes.int)
def string_compare(str1: f'uint[{255}]', str2: f'uint[{255}]') -> kf.GLTypes.int:
    """Compare two strings lexicographically. Returns:
       0 if equal
       <0 if str1 < str2
       >0 if str1 > str2
    """
    length1 = string_length(str1)
    length2 = string_length(str2)
    min_length : uint = uint(0)
    if length1 < length2:
        min_length = length1
    else:
        min_length = length2
    
    for i in range(255):
        if i >= min_length:
            break
        if str1[i] != str2[i]:
            return int(str1[i]) - int(str2[i])
    
    return int(length1) - int(length2)


# ============================================================================
# STRING TRANSFORMATION

@engine.function({
    'str_array': f'uint[{255}]',
}, return_type=f'uint[{255}]')
def string_to_upper(str_array: f'uint[{255}]') -> f'uint[{255}]':
    """Convert string to uppercase"""
    length = string_length(str_array)
    out : uint[255]
    for i in range(255):
        if i >= length:
            out[i] = uint(0xFFFFFFFF)
        else:
            out[i] = to_upper(str_array[i])
    return out


@engine.function({
    'str_array': f'uint[{255}]',
}, return_type=f'uint[{255}]')
def string_to_lower(str_array: f'uint[{255}]') -> 'void':
    """Convert string to lowercase"""
    length = string_length(str_array)
    out : uint[255]
    for i in range(255):
        if i >= length:
            out[i] = uint(0xFFFFFFFF)
        else:
            out[i] = to_lower(str_array[i])
    return out


@engine.function({
    'str_array': f'uint[{255}]',
}, return_type=f'uint[{255}]')
def string_reverse(str_array: f'uint[{255}]') -> f'uint[{255}]':
    """Reverse a string"""
    length = string_length(str_array)
    out : uint[255]
    for i in range(255):
        if i < length:
            out[i] = str_array[length - uint(1) - i]
        else:
            out[i] = uint(0xFFFFFFFF)
    return out


# ============================================================================
# STRING SEARCH

@engine.function({
    'str_array': f'uint[{255}]',
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.uint)
def string_find(str_array: f'uint[{255}]', char_code: kf.GLTypes.uint) -> kf.GLTypes.uint:
    """Find first occurrence of character. Returns index or max uint if not found"""
    length = string_length(str_array)
    for i in range(255):
        if i >= length:
            return uint(0xFFFFFFFF)
        if str_array[i] == char_code:
            return i
    return uint(0xFFFFFFFF)


@engine.function({
    'str_array': f'uint[{255}]',
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.uint)
def string_rfind(str_array: f'uint[{255}]', char_code: kf.GLTypes.uint) -> kf.GLTypes.uint:
    """Find last occurrence of character. Returns index or max uint if not found"""
    length = string_length(str_array)
    result : uint = uint(0xFFFFFFFF)
    
    for i in range(255):
        if i >= length:
            return uint(0xFFFFFFFF)
        if str_array[i] == chaar:
            result = i
    
    return result


@engine.function({
    'str_array': f'uint[{255}]',
    'char_code': kf.GLTypes.uint
}, return_type=kf.GLTypes.uint)
def string_count(str_array: f'uint[{255}]', char_code: kf.GLTypes.uint) -> kf.GLTypes.uint:
    """Count occurrences of char_codeacter"""
    length = string_length(str_array)
    count : uint = uint(0)
    
    for i in range(255):
        if i >= length:
            return count
        if str_array[i] == char_code:
            count = count + uint(1)
    
    return count


# ============================================================================
# STRING MODIFICATION

@engine.function({
    'str_array': f'uint[{255}]',
    'old_char_code': kf.GLTypes.uint,
    'new_char_code': kf.GLTypes.uint
}, return_type=f'uint[{255}]')
def string_replace_char(str_array: f'uint[{255}]', old_char_code: kf.GLTypes.uint, new_char_code: kf.GLTypes.uint) -> 'void':
    """Replace all occurrences of old_char_code with new_char_code"""
    length = string_length(str_array)
    out : uint[255]
    
    for i in range(255):
        if i >= length:
            out[i] = uint(0xFFFFFFFF)
        else:
            if str_array[i] == old_char_code:
                out[i] = new_char_code
            else:
                out[i] = str_array[i]
    
    return out


@engine.function({
    'str1': f'uint[{255}]',
    'str2': f'uint[{255}]'
}, return_type=f'uint[{255}]')
def string_concat(str1: f'uint[{255}]', str2: f'uint[{255}]') -> f'uint[{255}]':
    """Concatenate two strings"""
    len1 = string_length(str1)
    len2 = string_length(str2)
    
    idx : uint = uint(0)
    out : uint[255]
    for i in range(255):
        if i < len1 and idx < 255:
            out[idx] = str1[i]
            idx = idx + uint(1)
    
    for i in range(255):
        if i < len2 and idx < 255:
            out[idx] = str2[i]
            idx = idx + uint(1)
    
    if idx < 255:
        out[idx] = uint(0xFFFFFFFF)
    
    for i in range(idx + 1, 255):
        out[i] = uint(0xFFFFFFFF)

    return out


@engine.function({
    'str_array': f'uint[{255}]',
    'start': kf.GLTypes.uint,
    'length': kf.GLTypes.uint
}, return_type=f'uint[{255}]')
def string_substring(str_array: f'uint[{255}]', start: kf.GLTypes.uint, length: kf.GLTypes.uint) -> 'void':
    """Extract substring"""
    str_len = string_length(str_array)
    out : uint[255]
    
    out_idx : uint = uint(0)
    found : bool = False
    for i in range(255):
        src_idx : uint = start + i
        if i < length and src_idx < str_len and not found:
            out[out_idx] = str_array[src_idx]
            out_idx = out_idx + uint(1)
        else:
            found = True
    
    for i in range(out_idx, 255):
        out[i] = uint(0xFFFFFFFF)
    
    return out


# ============================================================================
# NUMERIC CONVERSION

@engine.function({
    'str_array': f'uint[{255}]'
}, return_type=kf.GLTypes.int)
def string_to_int(str_array: f'uint[{255}]') -> kf.GLTypes.int:
    """Convert string to integer"""
    length = string_length(str_array)
    result : int = int(0)
    is_negative : bool = bool(false)
    start_idx : uint = uint(0)
    
    if length > uint(0) and str_array[0] == uint(12):
        is_negative = bool(true)
        start_idx = uint(1)
    
    for i in range(255):
        idx : uint = start_idx + i
        if idx < length:        
            ch = str_array[idx]
            if is_digit(ch):
                digit = int(ch) - int(16)
                result = result * int(10) + digit
    
    if is_negative:
        result = -result
    
    return result


@engine.function({
    'str_array': f'uint[{255}]'
}, return_type=kf.GLTypes.float)
def string_to_float(str_array: f'uint[{255}]') -> kf.GLTypes.float:
    """Convert string to float"""
    length = string_length(str_array)
    int_part : int = int(0)
    frac_part : float = float(0)
    is_negative : bool = bool(false)
    found_dot : bool = bool(false)
    start_idx : uint = uint(0)
    divisor : float = float(1)
    
    if length > uint(0) and str_array[0] == uint(12):
        is_negative = bool(true)
        start_idx = uint(1)
    
    for i in range(255):
        idx : uint = start_idx + i
        if idx >= length:
            break
        
        ch = str_array[idx]
        
        if ch == uint(14):
            found_dot = bool(true)
        elif is_digit(ch):
            digit = int(ch) - int(16)
            
            if not found_dot:
                int_part = int_part * int(10) + digit
            else:
                divisor = divisor * float(10)
                frac_part = frac_part + float(digit) / divisor
    
    result : float = float(int_part) + frac_part
    
    if is_negative:
        result = -result
    
    return result


@engine.function({
    'value': kf.GLTypes.int,
}, return_type=f'uint[{255}]')
def int_to_string(value: kf.GLTypes.int) -> f'uint[{255}]':
    """Convert integer to string"""
    idx : uint = uint(0)
    out : uint[255]
    if value < int(0):
        if idx < 255:
            out[0] = uint(12)
            idx = idx + uint(1)
        value = -value
    
    temp : int = value
    digit_count : uint = uint(0)
    
    if temp == int(0):
        digit_count = uint(1)
    else:
        while temp > int(0):
            temp = temp / int(10)
            digit_count = digit_count + uint(1)
    
    for i in range(digit_count):
        divisor = int(1)
        for j in range(digit_count - uint(1) - i):
            divisor = divisor * int(10)
        digit = value / divisor
        value = value % divisor
        
        if idx < 255:
            out[idx] = uint(16) + uint(digit)
            idx = idx + uint(1)
    
    if idx < 255:
        out[idx] = uint(0xFFFFFFFF)
    
    for i in range(idx + 1, 255):
        out[i] = uint(0xFFFFFFFF)

    return out


@engine.function({
    'value': kf.GLTypes.float,
    'precision': kf.GLTypes.uint
}, return_type=f'uint[{255}]')
def float_to_string(value: kf.GLTypes.float, precision: kf.GLTypes.uint) -> f'uint[{255}]':
    """Convert float to string with given precision"""
    idx : uint = uint(0)
    out : uint[255]
    if value < float(0):
        if idx < 255:
            out[0] = uint(12)
            idx = idx + uint(1)
        value = -value
    
    int_part = int(value)
    temp : int = int_part
    digit_count : uint = uint(0)
    
    if temp == int(0):
        digit_count = uint(1)
    else:
        while temp > int(0):
            temp = temp / int(10)
            digit_count = digit_count + uint(1)
    
    for i in range(digit_count):
        divisor = int(1)
        for j in range(digit_count - uint(1) - i):
            divisor = divisor * int(10)
        digit = int_part / divisor
        int_part = int_part % divisor
        
        if idx < 255:
            out[idx] = uint(16) + uint(digit)
            idx = idx + uint(1)
    
    if precision > uint(0):
        if idx < 255:
            out[idx] = uint(14)
            idx = idx + uint(1)
        
        frac_part = value - float(int(value))
        for i in range(precision):
            frac_part *= float(10)
            digit = int(frac_part) % int(10)
            
            if idx < 255:
                out[idx] = uint(16) + uint(digit)
                idx = idx + uint(1)
    
    if idx < 255:
        out[idx] = uint(0xFFFFFFFF)
    
    for i in range(idx + 1, 255):
        out[i] = uint(0xFFFFFFFF)

    return out


# ============================================================================
# STRING UTILITIES

@engine.function({
    'str_array': f'uint[{255}]'
}, return_type=kf.GLTypes.bool)
def string_is_numeric(str_array: f'uint[{255}]') -> kf.GLTypes.bool:
    """Check if string contains only digits and optionally one dot and minus sign"""
    length = string_length(str_array)
    seen_dot : bool = bool(false)
    seen_minus : bool = bool(false)
    
    for i in range(255):
        if i >= length:
            break
        ch = str_array[i]
        
        if is_digit(ch):
            continue
        elif ch == uint(14):
            if seen_dot or i == uint(0) or (seen_minus and i == uint(1)):
                return bool(false)
            seen_dot = bool(true)
        elif ch == uint(12):
            if seen_minus or i != uint(0):
                return bool(false)
            seen_minus = bool(true)
        else:
            return bool(false)
    
    return bool(true)


@engine.function({
    'str_array': f'uint[{255}]'
}, return_type=kf.GLTypes.uint)
def string_hash(str_array: f'uint[{255}]') -> kf.GLTypes.uint:
    """Simple string hash function (djb2 algorithm)"""
    length = string_length(str_array)
    val : uint = uint(5381)
    
    for i in range(255):
        if i < length:
            val = ((val << uint(5)) + val) + str_array[i]
    
    return val

# ============================================================================
# Vertex Text Renderer Helper

@engine.function({
    'text_array': f'uint[{255}]',
    'color': kf.GLTypes.vec4,
    'cols': kf.GLTypes.uint,
    'rows': kf.GLTypes.uint,
    'x': kf.GLTypes.float,
    'y': kf.GLTypes.float,
    'charmap_texture': kf.GLTypes.sampler2D,
    'char_uvs': 'vec4[48]'
}, return_type=kf.GLTypes.vec4)
def render_text(
    text_array: f'uint[{255}]',
    color: kf.GLTypes.vec4,
    cols: kf.GLTypes.uint,
    rows: kf.GLTypes.uint,
    x: kf.GLTypes.float,
    y: kf.GLTypes.float,
    charmap_texture: kf.GLTypes.sampler2D,
    char_uvs: 'vec4[48]'
) -> kf.GLTypes.vec4:
    """Render text using character map texture"""
    cell_width: float = 1.0 / float(cols)
    cell_height: float = 1.0 / float(rows)
    
    char_x: uint = uint(floor(x / cell_width))
    char_y: uint = uint(floor((1.0 - y) / cell_height))
    
    char_index: uint = char_y * cols + char_x
    
    char_code: uint = uint(0xFFFFFFFF)
    if char_index < 255:
        char_code = text_array[int(char_index)]
    
    if char_code == uint(0xFFFFFFFF):
        return vec4(0.0, 0.0, 0.0, 0.0)
    
    uv_pair_index: int = int(char_code)
    uv_vec_index: int = uv_pair_index / 2
    uv_component: int = uv_pair_index % 2
    
    if uv_vec_index >= 48:
        return vec4(0.0, 0.0, 0.0, 0.0)
    
    uv_data: vec4 = char_uvs[uv_vec_index]
    base_uv: vec2 = vec2(0.0, 0.0)
    
    if uv_component == 0:
        base_uv = vec2(uv_data.x, uv_data.y)
    else:
        base_uv = vec2(uv_data.z, uv_data.w)
    
    local_x: float = fract(x / cell_width)
    local_y: float = fract((1.0 - y) / cell_height)
    
    final_uv: vec2 = vec2(
        base_uv.x + local_x / 16.0,
        base_uv.y + (1.0 - local_y) / 6.0
    )
    
    tex_color: vec4 = texture(charmap_texture, final_uv)
    
    result: vec4 = vec4(
        tex_color.r * color.r,
        tex_color.g * color.g,
        tex_color.b * color.b,
        tex_color.a * color.a
    )
    
    return result

# ============================================================================
# CHARACTER ENCODING/HELPER FUNCTIONS

def encode_string(val: str) -> list:
    """Helper function to encode strings to a list of integers for Panda3D"""
    result = []
    for char in val:
        if char in CHAR_TO_INDEX:
            result.append(CHAR_TO_INDEX[char])
        else:
            result.append(0)
    
    # Pad with -1 (which is 0xFFFFFFFF in two's complement / unsigned interpretation)
    # This works with Panda3D's C++ backend which uses signed int conversion
    while len(result) < 255:
        result.append(-1)
    
    return result[:255]


def encode_string_glsl(val: str) -> str:
    """Helper function to encode strings to GLSL format"""
    encoded = []
    for char in val:
        if char in CHAR_TO_INDEX:
            encoded.append(f"uint({CHAR_TO_INDEX[char]})")
        else:
            encoded.append("uint(0)")
    
    while len(encoded) < 255:
        encoded.append("uint(0xFFFFFFFF)")  # Use hex literal with 'u' suffix
    
    return f"uint[{255}]({', '.join(encoded[:255])})"

engine.encode_string   = encode_string
engine.encode_string_glsl    = encode_string_glsl

engine.STRLIB_CHARS          = CHARS
engine.STRLIB_CHARS_FLAT     = CHARS_FLAT
engine.STRLIB_CHAR_TO_INDEX  = CHAR_TO_INDEX
engine.STRLIB_CHAR_UV_LOOKUP = CHAR_UV_LOOKUP
engine.STRLIB_PACKED_UVS     = PACKED_UVS
engine.STRLIB_MAX_LENGTH     = MAX_STRING_LENGTH
engine.STRLIB_CHAR_UV_LENGTH = len(PACKED_UVS)
engine.STRLIB_TERMINATOR     = STRING_TERMINATOR
engine.charmap_texture = engine.load_texture(
    CHARMAP_PATH,
    min_filter = kf.TextureFilter.nearest,
    max_filter = kf.TextureFilter.nearest,
    wrap_u = kf.WrapMode.clamp,
    wrap_v = kf.WrapMode.clamp
)