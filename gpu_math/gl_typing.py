import numpy as np
from panda3d.core import Vec2, Vec3, Vec4, LVecBase2f, LVecBase3f, LVecBase4f

class IOTypes:
    buffer  = "buffer"
    uniform = "uniform"
    texture = "texture"

class GLTypes:
    float   = "float"
    double  = "double"
    int     = "int"
    uint    = "uint"
    bool    = "bool"
    void    = "void"
    vec2    = "vec2"
    vec3    = "vec3"
    vec4    = "vec4"

class NP_GLTypes:
    float   = (np.float32,  GLTypes.float)
    double  = (np.float64,  GLTypes.double)
    int     = (np.int32,    GLTypes.int)
    uint    = (np.uint32,   GLTypes.uint)
    bool    = (np.bool_,    GLTypes.bool)

class Vec_GLTypes:
    vec2    = (Vec2, GLTypes.vec2)
    vec3    = (Vec3, GLTypes.vec3)
    vec4    = (Vec4, GLTypes.vec4)

GLSL_TO_NP = {
    GLTypes.float:  np.float32,
    GLTypes.double: np.float64,
    GLTypes.int:    np.int32,
    GLTypes.uint:   np.uint32,
    GLTypes.bool:   np.bool_,
    GLTypes.vec2:   np.float32,
    GLTypes.vec3:   np.float32,
    GLTypes.vec4:   np.float32,
    GLTypes.void:   None,
}

NP_TO_GLSL = {
    bool:       GLTypes.bool,
    np.bool_:   GLTypes.bool,
    np.int32:   GLTypes.int,
    np.uint32:  GLTypes.uint,
    np.float32: GLTypes.float,
    np.float64: GLTypes.double,
}

VEC_TO_GLSL = {
    Vec2:       GLTypes.vec2,
    Vec3:       GLTypes.vec3,
    Vec4:       GLTypes.vec4,
    LVecBase2f: GLTypes.vec2,
    LVecBase3f: GLTypes.vec3,
    LVecBase4f: GLTypes.vec4,
}