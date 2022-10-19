from __future__ import annotations
from typing import List, Callable
import numpy as np

__all__ = [
    'Body', 'stepTime', 
]

FINE_DT = .01   # very fine dt. This is for sim, not for DL. 

class Body:
    def __init__(self) -> None:
        self.position = np.zeros((3, ))
        self.velocity = np.zeros((3, ))
    
    def snapshot(self):
        body = Body()
        body.position = self.position.copy()
        body.velocity = self.velocity.copy()
        return body
    
    def __repr__(self):
        return f'<ball {self.position}>'
    
    def toJSON(self):
        return [list(self.position), list(self.velocity)]
    
    def fromJSON(self, x: List[List[float]], /):
        self.position = np.array(x[0])
        self.velocity = np.array(x[1])
        return self

def stepTime(
    time, stepFineTime: Callable[[float], None], 
    fine_dt=FINE_DT, 
):
    while time > 0:
        if time < fine_dt:
            dt = time
            time = -1   # make sure to exit loop
        else:
            dt = fine_dt
            time -= dt
        stepFineTime(dt)
