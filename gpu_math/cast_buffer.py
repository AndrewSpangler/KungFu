import numpy as np

class CastBuffer:
    """Wrapper to handle numpy buffers"""
    def __init__(
        self,
        buff:np.ndarray, 
        n_items:int,
        cast=np.float32
    ):
        self.buffer = buff
        self.n_items = n_items
        self.cast = cast
        
    def __len__(self):
        return self.n_items