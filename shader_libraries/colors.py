import kungfu as kf
import engine

# ============================================================================
# GRAYSCALE CONVERSIONS

@engine.function({
    'r': kf.GLTypes.float,
    'g': kf.GLTypes.float,
    'b': kf.GLTypes.float
}, return_type=kf.GLTypes.float)
def grayscale_rgb(
    r: kf.GLTypes.float,
    g: kf.GLTypes.float,
    b: kf.GLTypes.float
) -> kf.GLTypes.float:
    """Convert RGB to grayscale using standard luminance weights"""
    return 0.299 * r + 0.587 * g + 0.114 * b


@engine.function({
    'r': kf.GLTypes.float,
    'g': kf.GLTypes.float,
    'b': kf.GLTypes.float,
    'a': kf.GLTypes.float
}, return_type=kf.GLTypes.float)
def grayscale_rgba(
    r: kf.GLTypes.float,
    g: kf.GLTypes.float,
    b: kf.GLTypes.float,
    a: kf.GLTypes.float
) -> kf.GLTypes.float:
    """Convert RGBA to grayscale, preserving alpha"""
    return grayscale_rgb(r, g, b) * a


@engine.function({
    'color': kf.GLTypes.vec3
}, return_type=kf.GLTypes.float)
def grayscale_vec3(color: kf.GLTypes.vec3) -> kf.GLTypes.float:
    """Convert vec3 color to grayscale"""
    return 0.299 * color.r + 0.587 * color.g + 0.114 * color.b


@engine.function({
    'color': kf.GLTypes.vec4
}, return_type=kf.GLTypes.float)
def grayscale_vec4(color: kf.GLTypes.vec4) -> kf.GLTypes.float:
    """Convert vec4 color to grayscale, preserving alpha"""
    return (0.299 * color.r + 0.587 * color.g + 0.114 * color.b) * color.a


# ============================================================================
# RGB <-> HSV CONVERSIONS

@engine.function({
    'r': kf.GLTypes.float,
    'g': kf.GLTypes.float,
    'b': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def rgb_to_hsv(
    r: kf.GLTypes.float,
    g: kf.GLTypes.float,
    b: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Convert RGB to HSV. Returns vec3(hue, saturation, value)"""
    cmax = max(max(r, g), b)
    cmin = min(min(r, g), b)
    delta = cmax - cmin
    
    # Hue calculation
    h = 0.0
    if delta > 0.0:
        if cmax == r:
            h = 60.0 * (uint((g - b) / delta) % uint(6.0))
        elif cmax == g:
            h = 60.0 * (uint((b - r) / delta) + uint(2.0))
        else:
            h = 60.0 * (uint((r - g) / delta) + uint(4.0))
    
    # Saturation calculation
    s = 0.0
    if cmax > 0.0:
        s = delta / cmax
    
    # Value
    v = cmax
    
    return vec3(h, s, v)


@engine.function({
    'color': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def rgb_to_hsv_vec3(color: kf.GLTypes.vec3) -> kf.GLTypes.vec3:
    """Convert vec3 RGB to HSV"""
    return rgb_to_hsv(color.r, color.g, color.b)


@engine.function({
    'h': kf.GLTypes.float,
    's': kf.GLTypes.float,
    'v': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def hsv_to_rgb(
    h: kf.GLTypes.float,
    s: kf.GLTypes.float,
    v: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Convert HSV to RGB. H in [0, 360], S and V in [0, 1]"""
    c = v * s
    x = c * (1.0 - abs(  uint(h / 60.0) % uint(2.0) - 1.0))
    m = v - c
    
    h_sector = h / 60.0
    r = 0.0
    g = 0.0
    b = 0.0
    
    if h_sector < 1.0:
        r = c
        g = x
        b = 0.0
    elif h_sector < 2.0:
        r = x
        g = c
        b = 0.0
    elif h_sector < 3.0:
        r = 0.0
        g = c
        b = x
    elif h_sector < 4.0:
        r = 0.0
        g = x
        b = c
    elif h_sector < 5.0:
        r = x
        g = 0.0
        b = c
    else:
        r = c
        g = 0.0
        b = x
    
    return vec3(r + m, g + m, b + m)


@engine.function({
    'hsv': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def hsv_to_rgb_vec3(hsv: kf.GLTypes.vec3) -> kf.GLTypes.vec3:
    """Convert vec3 HSV to RGB"""
    return hsv_to_rgb(hsv.x, hsv.y, hsv.z)


# ============================================================================
# RGB <-> HSL CONVERSIONS

@engine.function({
    'r': kf.GLTypes.float,
    'g': kf.GLTypes.float,
    'b': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def rgb_to_hsl(
    r: kf.GLTypes.float,
    g: kf.GLTypes.float,
    b: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Convert RGB to HSL. Returns vec3(hue, saturation, lightness)"""
    cmax = max(max(r, g), b)
    cmin = min(min(r, g), b)
    delta = cmax - cmin
    
    # Lightness
    l = (cmax + cmin) / 2.0
    
    # Hue calculation
    h = 0.0
    if delta > 0.0:
        if cmax == r:
            h = 60.0 * (uint((g - b) / delta) % uint(6.0))
        elif cmax == g:
            h = 60.0 * (((b - r) / delta) + 2.0)
        else:
            h = 60.0 * (((r - g) / delta) + 4.0)
    
    # Saturation calculation
    s = 0.0
    if delta > 0.0:
        s = delta / (1.0 - abs(2.0 * l - 1.0))
    
    return vec3(h, s, l)


@engine.function({
    'color': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def rgb_to_hsl_vec3(color: kf.GLTypes.vec3) -> kf.GLTypes.vec3:
    """Convert vec3 RGB to HSL"""
    return rgb_to_hsl(color.r, color.g, color.b)


@engine.function({
    'h': kf.GLTypes.float,
    's': kf.GLTypes.float,
    'l': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def hsl_to_rgb(
    h: kf.GLTypes.float,
    s: kf.GLTypes.float,
    l: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Convert HSL to RGB. H in [0, 360], S and L in [0, 1]"""
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - uint(abs((h / 60.0))) % uint(2.0) - 1.0)
    m = l - c / 2.0
    
    h_sector = h / 60.0
    r = 0.0
    g = 0.0
    b = 0.0
    
    if h_sector < 1.0:
        r = c
        g = x
        b = 0.0
    elif h_sector < 2.0:
        r = x
        g = c
        b = 0.0
    elif h_sector < 3.0:
        r = 0.0
        g = c
        b = x
    elif h_sector < 4.0:
        r = 0.0
        g = x
        b = c
    elif h_sector < 5.0:
        r = x
        g = 0.0
        b = c
    else:
        r = c
        g = 0.0
        b = x
    
    return vec3(r + m, g + m, b + m)


@engine.function({
    'hsl': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def hsl_to_rgb_vec3(hsl: kf.GLTypes.vec3) -> kf.GLTypes.vec3:
    """Convert vec3 HSL to RGB"""
    return hsl_to_rgb(hsl.x, hsl.y, hsl.z)


# ============================================================================
# COLOR ADJUSTMENTS

@engine.function({
    'color': kf.GLTypes.vec3,
    'amount': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def brightness(color: kf.GLTypes.vec3, amount: kf.GLTypes.float) -> kf.GLTypes.vec3:
    """Adjust brightness by adding amount to each channel"""
    return color + vec3(amount, amount, amount)


@engine.function({
    'color': kf.GLTypes.vec3,
    'amount': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def contrast(color: kf.GLTypes.vec3, amount: kf.GLTypes.float) -> kf.GLTypes.vec3:
    """Adjust contrast. amount = 1.0 is no change, < 1.0 reduces, > 1.0 increases"""
    return (color - 0.5) * amount + 0.5


@engine.function({
    'color': kf.GLTypes.vec3,
    'amount': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def saturation(color: kf.GLTypes.vec3, amount: kf.GLTypes.float) -> kf.GLTypes.vec3:
    """Adjust saturation. amount = 1.0 is no change, 0.0 is grayscale"""
    gray = grayscale_vec3(color)
    return mix(kf.GLTypes.vec3(gray, gray, gray), color, amount)


@engine.function({
    'color': kf.GLTypes.vec3,
    'degrees': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def hue_shift(color: kf.GLTypes.vec3, degrees: kf.GLTypes.float) -> kf.GLTypes.vec3:
    """Shift hue by degrees (0-360)"""
    hsv : vec3 = rgb_to_hsv_vec3(color)
    x : uint = uint((hsv.x + degrees)) % uint(360.0)
    return hsv_to_rgb_vec3(vec3(x, hsv.y, hsv.z))

@engine.function({
    'color': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def invert(color: kf.GLTypes.vec3) -> kf.GLTypes.vec3:
    """Invert color"""
    return vec3(1.0, 1.0, 1.0) - color

# ============================================================================
# BLENDING

@engine.function({
    'base': kf.GLTypes.vec3,
    'blend': kf.GLTypes.vec3,
    'opacity': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def blend_add(
    base: kf.GLTypes.vec3,
    blend: kf.GLTypes.vec3,
    opacity: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Additive blend mode"""
    result : vec3 = min(base + blend, vec3(1.0, 1.0, 1.0))
    return mix(base, result, opacity)


@engine.function({
    'base': kf.GLTypes.vec3,
    'blend': kf.GLTypes.vec3,
    'opacity': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def blend_subtract(
    base: kf.GLTypes.vec3,
    blend: kf.GLTypes.vec3,
    opacity: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Subtractive blend mode"""
    result = max(base - blend, vec3(0.0, 0.0, 0.0))
    return mix(base, result, opacity)


@engine.function({
    'base': kf.GLTypes.vec3,
    'blend': kf.GLTypes.vec3,
    'opacity': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def blend_difference(
    base: kf.GLTypes.vec3,
    blend: kf.GLTypes.vec3,
    opacity: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Difference blend mode"""
    return mix(base, abs(base - blend), opacity)


# ============================================================================
# EFFECTS

@engine.function({
    'color': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def sepia(color: kf.GLTypes.vec3) -> kf.GLTypes.vec3:
    # Apply sepia effect
    r = color.r * 0.393 + color.g * 0.769 + color.b * 0.189
    g = color.r * 0.349 + color.g * 0.686 + color.b * 0.168
    b = color.r * 0.272 + color.g * 0.534 + color.b * 0.131
    return vec3(r, g, b)


@engine.function({
    'color': kf.GLTypes.vec3,
    'threshold': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def posterize(color: kf.GLTypes.vec3, threshold: kf.GLTypes.float) -> kf.GLTypes.vec3:
    # """Posterize color to threshold levels"""
    return floor(color * threshold) / threshold


@engine.function({
    'color': kf.GLTypes.vec3,
    'tint': kf.GLTypes.vec3,
    'amount': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def apply_tint(
    color: kf.GLTypes.vec3,
    tint: kf.GLTypes.vec3,
    amount: kf.GLTypes.float
) -> kf.GLTypes.vec3:
    """Apply a tint to the color"""
    return mix(color, color * tint, amount)

# ============================================================================
# COLOR UTILS

@engine.function({
    'color': kf.GLTypes.vec3
}, return_type=kf.GLTypes.float)
def luminance(color: kf.GLTypes.vec3) -> kf.GLTypes.float:
    """Calculate luminance (same as grayscale_vec3)"""
    return grayscale_vec3(color)


@engine.function({
    'color': kf.GLTypes.vec3
}, return_type=kf.GLTypes.float)
def perceived_brightness(color: kf.GLTypes.vec3) -> kf.GLTypes.float:
    """Calculate perceived brightness using sRGB weights"""
    return sqrt(0.299 * color.r * color.r + 0.587 * color.g * color.g + 0.114 * color.b * color.b)


@engine.function({
    'c1': kf.GLTypes.vec3,
    'c2': kf.GLTypes.vec3
}, return_type=kf.GLTypes.float)
def color_distance(c1: kf.GLTypes.vec3, c2: kf.GLTypes.vec3) -> kf.GLTypes.float:
    """Calculate Euclidean distance between two colors"""
    diff = c1 - c2
    return sqrt(dot(diff, diff))


@engine.function({
    'color': kf.GLTypes.vec3,
    'threshold': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def threshold_color(color: kf.GLTypes.vec3, threshold: kf.GLTypes.float) -> kf.GLTypes.vec3:
    """Threshold color to black or white based on luminance"""
    value = float(luminance(color) > threshold)
    return vec3(value, value, value)


# ============================================================================
# PALETTES

@engine.function({
    't': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def rainbow_gradient(t: kf.GLTypes.float) -> kf.GLTypes.vec3:
    """Generate rainbow color from t [0, 1]"""
    hue = t * 360.0
    return hsv_to_rgb(hue, 1.0, 1.0)


@engine.function({
    't': kf.GLTypes.float
}, return_type=kf.GLTypes.vec3)
def heat_map(t: kf.GLTypes.float) -> kf.GLTypes.vec3:
    """Generate heat map color from t [0, 1] (black -> red -> yellow -> white)"""
    result : vec3 = vec3(0,0,0)
    if t < 0.25:
        # Black to red
        result : vec3 = vec3(t * 4.0, 0.0, 0.0)
    elif t < 0.5:
        # Red to yellow
        result : vec3 = vec3(1.0, (t - 0.25) * 4.0, 0.0)
    elif t < 0.75:
        # Yellow to white
        result : vec3 = vec3(1.0, 1.0, (t - 0.5) * 4.0)
    else:
        # White
        return vec3(1.0, 1.0, 1.0)
    return result

@engine.function({
    't': kf.GLTypes.float,
    'c1': kf.GLTypes.vec3,
    'c2': kf.GLTypes.vec3,
    'c3': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def gradient_3_colors(
    t: kf.GLTypes.float,
    c1: kf.GLTypes.vec3,
    c2: kf.GLTypes.vec3,
    c3: kf.GLTypes.vec3
) -> kf.GLTypes.vec3:
    """Interpolate between three colors"""
    if t < 0.5:
        return mix(c1, c2, t * 2.0)
    else:
        return mix(c2, c3, (t - 0.5) * 2.0)

@engine.function({
    't': kf.GLTypes.float,
    'c1': kf.GLTypes.vec3,
    'c2': kf.GLTypes.vec3,
    'c3': kf.GLTypes.vec3,
    'c4': kf.GLTypes.vec3
}, return_type=kf.GLTypes.vec3)
def gradient_4_colors(
    t: kf.GLTypes.float,
    c1: kf.GLTypes.vec3,
    c2: kf.GLTypes.vec3,
    c3: kf.GLTypes.vec3,
    c4: kf.GLTypes.vec3
) -> kf.GLTypes.vec3:
    """Interpolate between four colors"""
    if t < 0.3333:
        return mix(c1, c2, t * 2.0)
    elif t < 0.6666:
        return mix(c2, c3, (t - 0.3333) * 2.0)
    else:
        return mix(c3, c4, (t - 0.6666) * 2.0)