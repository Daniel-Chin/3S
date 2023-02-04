__all__ = [
    'TRAJ_FILENAME', 
    
    'torch2PIL', 'torch2np', 'reparameterize', 

    'EnumStr', 'Note', 'Song', 
]

from typing import *
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torchWork import *

TRAJ_FILENAME = 'trajectory.json'

def torch2np(torchImg: torch.Tensor) -> np.ndarray:
    return (
        torchImg.cpu().detach().clamp(0, 1)
        .permute(1, 2, 0) * 255
    ).round().numpy().astype(np.uint8)

def torch2PIL(torchImg: torch.Tensor, mode):
    npImg = torch2np(torchImg)
    if mode == 'L':
        npImg = npImg[:, :, 0]
    return Image.fromarray(npImg, mode)

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu

class EnumStr: 
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    # behave like primitives
    def __copy__(self):
        return self
    def __deepcopy__(self, _):
        return self
    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

class Note:
    def __init__(self, is_rest: bool, pitch: Optional[int] = None):
        assert is_rest == (pitch is None)
        self.is_rest = is_rest
        if not is_rest:
            self.pitch = pitch
    
    def toJSON(self):
        if self.is_rest:
            return -1
        else:
            return self.pitch
    
    @staticmethod
    def fromJSON(obj):
        if obj == -1:
            return Note(True, None)
        else:
            return Note(False, obj)

class Song:
    def __init__(self, notes: List[Note]) -> None:
        self.notes = notes
    
    def toJSON(self):
        return [x.toJSON() for x in self.notes]
    
    @staticmethod
    def fromJSON(obj):
        return __class__([Note.fromJSON(x) for x in obj])
    
    @lru_cache()
    def toTensor(self):
        x = torch.zeros((len(self.notes), 2))
        for i, note in enumerate(self.notes):
            if note.is_rest:
                x[i, 1] = 1
            else:
                x[i, 0] = note.pitch
        return x
