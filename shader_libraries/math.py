import kungfu as kf
import engine

@engine.function({
    'a': kf.GLTypes.float,
    'b': kf.GLTypes.float,
},  return_type=kf.GLTypes.float)
def dist(a, b) -> kf.GLTypes.float:
    return sqrt(a * a + b * b)