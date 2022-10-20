from __future__ import annotations

from typing import List
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt

from physics_shared import *

__all__ = [
    'initBodies', 'oneLegalRun', 
    'VIEW_RADIUS', 'BALL_RADIUS', 
]

POSITION_STD = 1
VELOCITY_STD = 1
G = 1
BALL_RADIUS = .5
VIEW_RADIUS = 7

class EmptyFrameException(Exception): pass
class CollisionException(Exception): pass
class RejectThisSampleException(Exception): pass

def stepFineTime(a: Body, b: Body, /, dt: float):
    # O(2n) = O(n). Readability first. 
    for self, other in ((a, b), (b, a)):
        displace = other.position - self.position
        r = np.linalg.norm(displace)
        unit_displace = displace / r
        force = G / r ** 2 * unit_displace
        self.velocity += dt * force
        self.position += dt * self.velocity

def initBodies():
    bodies: List[Body] = []
    for _ in range(2):
        body = Body()
        bodies.append(body)
        body.position = np.random.normal(
            0, POSITION_STD, 3, 
        )
        body.velocity = np.random.normal(
            0, VELOCITY_STD, 3, 
        )
    return bodies

def oneRun(dt: float, n_frames: int, rejectable_start: int):
    bodies = initBodies()
    trajectory: List[List[Body]] = []
    acc = 0
    for t in range(n_frames):
        trajectory.append([x.snapshot() for x in bodies])
        stepTime(dt, lambda x : stepFineTime(*bodies, x))
        try:
            verify(bodies)
        except EmptyFrameException:
            if t < rejectable_start:
                raise RejectThisSampleException
            else:
                acc += 1
        except CollisionException:
            raise RejectThisSampleException
    return trajectory, acc

def verify(bodies: List[Body]):
    for body in bodies:
        if np.linalg.norm(body.position) > VIEW_RADIUS:
            raise EmptyFrameException
    if np.linalg.norm(
        bodies[0].position - bodies[1].position
    ) < 2 * BALL_RADIUS:
        raise CollisionException

def oneLegalRun(*a, **kw):
    # rejection sampling
    while True:
        try:
            trajectory, n_empty = oneRun(*a, **kw)
        except RejectThisSampleException:
            print('rej')
            continue
        else:
            print('Accept')
            return trajectory, n_empty

def manyLegalRuns(
    n: int, dt: float, n_frames: int, rejectable_start: int, 
):
    trajs: List[List[List[Body]]] = []
    acc = 0
    for _ in range(n):
        trajectory, n_empty = oneLegalRun(
            dt, n_frames, rejectable_start, 
        ) 
        trajs.append(trajectory)
        acc += n_empty
    print(f'Emptry frames: {acc / n_frames / n : .1%}')
    return trajs

def test():
    pprint(oneLegalRun(.15, 20, 6))
    ts = [0, 3, 9, 20]
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    axes = [*axes[0], *axes[1]]
    for t, ax in zip(ts, axes):
        runs = manyLegalRuns(30, .15, 20, t)
        positions = []
        for run in runs:
            for bodies in run:
                for body in bodies:
                    positions.extend(body.position)
        analyzeDistribution(positions, ax)
        ax.set_title(f't < {t}')
        ax.set_xlim(-4, 4)
    fig.suptitle('Reject empty frame when t...')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    test()
