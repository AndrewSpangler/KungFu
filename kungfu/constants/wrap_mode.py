from panda3d.core import SamplerState

class WrapMode:
    clamp        = SamplerState.WM_clamp
    repeat       = SamplerState.WM_repeat
    mirror       = SamplerState.WM_mirror
    mirror_once  = SamplerState.WM_mirror_once
    border_color = SamplerState.WM_border_color