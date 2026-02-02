# KungFU

## Libraries
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

