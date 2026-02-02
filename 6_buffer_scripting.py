import numpy as np
import kungfu as kf
from direct.showbase.ShowBase import ShowBase

app = ShowBase()
engine = kf.GPUMath(app, headless=False)

vals = np.linspace(0, 1, 1000)
handle = engine.sin(vals)
handle = engine.mult(handle, vals)
handle = engine.add(handle, vals)
handle = engine.add(handle, 3)

print(engine.fetch(handle))