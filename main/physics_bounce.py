from __future__ import annotations
from typing import List
import numpy as np

from physics_shared import *

__all__ = [
    'Body', 'initLegalBody', 'oneLegalRun', 
    'BALL_RADIUS', 
]

POSITION_MU = np.array([0, 6, 2])
POSITION_STD = (2, 2, 1)
VELOCITY_STD = 2
BIG_G = 10
BALL_RADIUS = .5

def stepFineTime(dt, body: Body):
    body.velocity[2] -= dt * BIG_G
    if (
        body.velocity[2] < 0 
        and body.position[2] - BALL_RADIUS
    ) < 0:
        body.velocity[2] *= -1
    body.position += dt * body.velocity

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
        stepTime(dt, lambda x : stepFineTime(x, body))
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
