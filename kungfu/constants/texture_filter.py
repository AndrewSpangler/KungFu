from panda3d.core import Texture

class TextureFilter:
    nearest                 = Texture.FT_nearest
    linear                  = Texture.FT_linear
    nearest_mipmap_nearest  = Texture.FT_nearest_mipmap_nearest
    linear_mipmap_nearest   = Texture.FT_linear_mipmap_nearest
    nearest_mipmap_linear   = Texture.FT_nearest_mipmap_linear
    linear_mipmap_linear    = Texture.FT_linear_mipmap_linear
    shadow                  = Texture.FT_shadow
    default                 = Texture.FT_default