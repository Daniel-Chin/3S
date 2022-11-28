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

POSITION_STD = 2.5
VELOCITY_STD = .3
G = 1
BALL_RADIUS = .8
VIEW_RADIUS = 7
OCCLUSION_THRES = .998

class Reject(Exception): pass
class StrongRejection(Reject): pass

class EmptyFrameException(Reject): pass

class CollisionException(StrongRejection): pass
class OccludedException(StrongRejection): pass
class AngularVelocityTooLarge(StrongRejection): pass
class LessThanOneLoop(StrongRejection): pass
class WeakDepthVariation(StrongRejection): pass

def stepFineTime(a: Body, b: Body, /, dt: float):
    # O(2n) = O(n). Readability first. 
    for self, other in ((a, b), (b, a)):
        displace = other.position - self.position
        r = np.linalg.norm(displace)
        unit_displace = displace / r
        force = G / r ** 2 * unit_displace
        self.velocity += dt * force
        self.position += dt * self.velocity

def toUnit(x: np.ndarray, /):
    return x / np.linalg.norm(x)

def initBodies(center_of_mass_constant: bool):
    bodies = [Body(), Body()]
    for body in bodies:
        body.position = np.random.normal(
            0, POSITION_STD, 3, 
        )
        body.velocity = np.random.normal(
            0, VELOCITY_STD, 3, 
        )
    del body
    self, other = bodies
    position_unit = toUnit(self.position)
    normal_to_sphere = np.dot(
        self.velocity, position_unit
    ) * position_unit
    self.velocity -= normal_to_sphere   # make it tangent

    other.position = - self.position
    other.velocity = - self.velocity
    energy = systemEnergy(*bodies)
    # print(f'{energy = }')
    if energy > -.07:
        raise LessThanOneLoop(f'{energy = }')
    if energy < -.10:
        raise AngularVelocityTooLarge(f'{energy = }')
    if not center_of_mass_constant:
        offset = np.random.normal(
            0, POSITION_STD * .7, 3, 
        )
        for body in bodies:
            body.position += offset
    return bodies

def systemEnergy(*bodies: Body):
    a, b = bodies
    r = np.linalg.norm(a.position - b.position)
    e_potential = - G / r
    e_kinetic = sum([np.linalg.norm(x.velocity) ** 2 / 2 for x in bodies])
    return e_potential + e_kinetic

def oneRun(
    dt: float, n_frames: int, 
    center_of_mass_stationary: bool, 
    rejectable_start: int, eye_pos, 
):
    bodies = initBodies(center_of_mass_stationary)
    trajectory: List[List[Body]] = []
    acc = 0
    for t in range(n_frames):
        try:
            verify(t, bodies, eye_pos)
        except EmptyFrameException:
            if t < rejectable_start:
                raise
            else:
                acc += 1
        trajectory.append([x.snapshot() for x in bodies])
        stepTime(dt, lambda x : stepFineTime(*bodies, x))
    return trajectory, acc

def verify(t, bodies: List[Body], eye_pos: np.ndarray):
    if np.linalg.norm(
        bodies[0].position - bodies[1].position
    ) < 2 * BALL_RADIUS:
        raise CollisionException(f'{t = }')
    
    displace = [b.position - eye_pos for b in bodies]
    unit_displace = [x / np.linalg.norm(x) for x in displace]
    if np.dot(*unit_displace) > OCCLUSION_THRES:
        raise OccludedException(f'{t = }')
    
    for body in bodies:
        if np.linalg.norm(body.position) > VIEW_RADIUS:
            raise EmptyFrameException(f'{t = }')

def oneLegalRun(*a, **kw):
    # rejection sampling
    while True:
        try:
            trajectory, n_empty = oneRun(*a, **kw)
            traj_array = np.array([
                (b0.position, b1.position) for (b0, b1) in trajectory
            ])
            y_std = traj_array[:, 0, 1].std()
            # print(f'{y_std = }')
            if y_std < 1.4:
                raise WeakDepthVariation(f'{y_std = }')
        except Reject as e:
            # print('rej:', repr(e))
            continue
        else:
            # print('Accept')
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
