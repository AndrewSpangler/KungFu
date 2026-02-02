import numpy as np

class CastBuffer:
    """Wrapper to handle numpy buffers"""
    def __init__(
        self,
        buff:np.ndarray, 
        n_items:int=None,
        cast=np.float32
    ):
        self.buffer = buff
        self.n_items = n_items if not n_items is None else buff.shape[0]
        self.cast = cast
        
    def __len__(self):
        return self.n_items