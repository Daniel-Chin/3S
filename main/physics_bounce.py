from __future__ import annotations
from typing import List
import numpy as np

__all__ = [
    'Body', 'initLegalBody', 'oneLegalRun', 
    'BALL_RADIUS', 
]

POSITION_MU = np.array([0, 6, 2])
POSITION_STD = (2, 2, 1)
VELOCITY_STD = 2
BIG_G = 10
FINE_DT = .01   # very fine dt. This is for sim, not for DL. 
BALL_RADIUS = .5

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
    
    def stepFineTime(self, dt):
        self.velocity[2] -= dt * BIG_G
        if (
            self.velocity[2] < 0 
            and self.position[2] - BALL_RADIUS
        ) < 0:
            self.velocity[2] *= -1
        self.position += dt * self.velocity
    
    def stepTime(self, time):
        while time > 0:
            if time < FINE_DT:
                dt = time
                time = -1   # make sure to exit loop
            else:
                dt = FINE_DT
                time -= dt
            self.stepFineTime(dt)
    
    def toJSON(self):
        return [list(self.position), list(self.velocity)]
    
    def fromJSON(self, x: List[List[float]], /):
        self.position = np.array(x[0])
        self.velocity = np.array(x[1])
        return self

def initBody():
    body = Body()
    for i in range(3):
        body.position[i] = np.random.uniform(
            POSITION_MU[i] - 2 * POSITION_STD[i], 
            POSITION_MU[i] + 2 * POSITION_STD[i], 
            1, 
        )[0]
    body.velocity = np.random.uniform(
        - 2 * VELOCITY_STD, 2 * VELOCITY_STD, 3, 
    )
    return body

def initLegalBody():
    # rejection sampling
    while True:
        body = initBody()
        if body.position[2] < .5 + BALL_RADIUS:
            raise RejectThisSampleException
        return body    

def oneRun(dt, n_frames):
    body = initLegalBody()
    trajectory: List[List[Body]] = []
    for _ in range(n_frames):
        trajectory.append([body.snapshot()])
        body.stepTime(dt)
        verify(body)
    return trajectory

class RejectThisSampleException(Exception): pass

def verify(body: Body):
    if np.linalg.norm(body.position[:2] - POSITION_MU[:2]) > 4:
        print('rej xy')
        raise RejectThisSampleException
    if not 0 <= body.position[2] < 4:
        print('rej z')
        raise RejectThisSampleException

def oneLegalRun(*a, **kw):
    # rejection sampling
    while True:
        try:
            trajectory = oneRun(*a, **kw)
        except RejectThisSampleException:
            continue
        else:
            print('Accept')
            return trajectory

if __name__ == '__main__':
    print(*oneLegalRun(1, 30), sep='\n')
