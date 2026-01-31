from panda3d.core import Texture

class TextureFormat:
    depth_stencil   = Texture.F_depth_stencil
    color_index     = Texture.F_color_index
    red             = Texture.F_red
    green           = Texture.F_green
    blue            = Texture.F_blue
    alpha           = Texture.F_alpha
    rgb             = Texture.F_rgb     # any suitable RGB mode, whatever the hardware prefers

    rgb5            = Texture.F_rgb5    # 5 bits per R,G,B channel
    rgb8            = Texture.F_rgb8    # 8 bits per R,G,B channel
    rgb12           = Texture.F_rgb12   # 12 bits per R,G,B channel
    rgb332          = Texture.F_rgb332  # 3 bits per R & G, 2 bits for B

    rgba            = Texture.F_rgba    # any suitable RGBA mode, whatever the hardware prefers

    # Again, the following bitdepth requests are only for the GSG; within the
    # Texture object itself, these are all equivalent.
    rgbm            = Texture.F_rgbm    # as above, but only requires 1 bit for alpha (i.e. mask)
    rgba4           = Texture.F_rgba4   # 4 bits per R,G,B,A channel
    rgba5           = Texture.F_rgba5   # 5 bits per R,G,B channel, 1 bit alpha
    rgba8           = Texture.F_rgba8   # 8 bits per R,G,B,A channel
    rgba12          = Texture.F_rgba12  # 12 bits per R,G,B,A channel

    luminance           = Texture.F_luminance
    luminance_alpha     = Texture.F_luminance_alpha      # 8 bits luminance, 8 bits alpha
    luminance_alphamask = Texture.F_luminance_alphamask  # 8 bits luminance, only needs 1 bit of alpha

    rgba16              = Texture.F_rgba16  # 16 bits per R,G,B,A channel
    rgba32              = Texture.F_rgba32  # 32 bits per R,G,B,A channel

    depth_component     = Texture.F_depth_component
    depth_component16   = Texture.F_depth_component16
    depth_component24   = Texture.F_depth_component24
    depth_component32   = Texture.F_depth_component32

    r16                 = Texture.F_r16
    rg16                = Texture.F_rg16
    rgb16               = Texture.F_rgb16

    # These formats are in the sRGB color space.  RGB is 2.2 gamma corrected,
    # alpha is always linear.
    srgb                = Texture.F_srgb
    srgb_alpha          = Texture.F_srgb_alpha
    sluminance          = Texture.F_sluminance
    sluminance_alpha    = Texture.F_sluminance_alpha

    r32i            = Texture.F_r32i  # 32-bit integer, used for atomic access
    r32             = Texture.F_r32
    rg32            = Texture.F_rg32
    rgb32           = Texture.F_rgb32

    r8i             = Texture.F_r8i # 8 integer bits per R channel
    rg8i            = Texture.F_rg8i # 8 integer bits per R,G channel
    rgb8i           = Texture.F_rgb8i # 8 integer bits per R,G,B channel
    rgba8i          = Texture.F_rgba8i # 8 integer bits per R,G,B,A channel

    r11_g11_b10     = Texture.F_r11_g11_b10 # unsigned floating-point, 11 Red, 11 Green, 10 Blue Bits
    rgb9_e5         = Texture.F_rgb9_e5
    rgb10_a2        = Texture.F_rgb10_a2

    rg              = Texture.F_rg

    r16i            = Texture.F_r16i
    rg16i           = Texture.F_rg16i
    rgb16i          = Texture.F_rgb16i # not recommended
    rgba16i         = Texture.F_rgba16i

    rg32i           = Texture.F_rg32i
    rgb32i          = Texture.F_rgb32i
    rgba32i         = Texture.F_rgba32i