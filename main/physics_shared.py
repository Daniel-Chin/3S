from __future__ import annotations
from typing import List, Callable

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import norm
import numpy as np

try:
    from auto_bin import autoBin
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'auto_bin', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

__all__ = [
    'Body', 'stepTime', 'analyzeDistribution', 
]

FINE_DT = .001   # very fine dt. This is for sim, not for DL. 

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

def analyzeDistribution(data, ax: Axes = None):
    ax = ax or plt
    data = data - np.mean(data)
    data = data / np.std(data)
    ax.hist(data, bins=autoBin(data), density=True)
    X = np.linspace(-4, 4, 1000)
    Y = norm.pdf(X)
    ax.plot(X, Y)
