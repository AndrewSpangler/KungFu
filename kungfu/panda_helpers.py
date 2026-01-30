import os
import numpy as np
from panda3d.core import Texture
from .gl_typing import (
    TextureFormat, TextureFilter,
    TextureType, TextureComponentType,
    WrapMode
)

def load_texture(
    path            : os.PathLike,
    min_filter      : TextureFilter = TextureFilter.linear,
    max_filter      : TextureFilter = TextureFilter.linear,
    wrap_u          : WrapMode      = WrapMode.repeat,
    wrap_v          : WrapMode      = WrapMode.repeat,
) -> Texture:
    """Load Panda3D texture from file"""
    prepare_texture(
        (tex := loader.loadTexture(path)),
        min_filter  = min_filter,
        max_filter  = max_filter,
        wrap_u      = wrap_u,
        wrap_v      = wrap_v,
    )
    return tex

def load_texture_from_array(
    name            : str,
    array           : np.ndarray,
    height          : int,
    width           : int,
    data_type       : np.dtype      = float,
    texture_format  : TextureFormat = TextureFormat.rgba,
    min_filter      : TextureFilter = TextureFilter.linear,
    max_filter      : TextureFilter = TextureFilter.linear,
    wrap_u          : WrapMode      = WrapMode.repeat,
    wrap_v          : WrapMode      = WrapMode.repeat,
) -> Texture:
    """Load and prepare a Panda3D texture"""
    tex = Texture(name)
    tex.setup2dTexture(height, width, data_type, texture_format)
    tex.setRamImage(array.tobytes())
    prepare_texture(
        tex,
        min_filter  = min_filter,
        max_filter  = max_filter,
        wrap_u      = wrap_u,
        wrap_v      = wrap_v,
    )
    return tex

def prepare_texture(
    tex             : Texture,
    min_filter      : TextureFilter = TextureFilter.linear,
    max_filter      : TextureFilter = TextureFilter.linear,
    wrap_u          : WrapMode      = WrapMode.repeat,
    wrap_v          : WrapMode      = WrapMode.repeat,
) -> None:
    """Prepare a texture for hardware filtering"""
    tex.setMinfilter(min_filter)
    tex.setMagfilter(max_filter)
    tex.setWrapU(wrap_u)
    tex.setWrapV(wrap_v)