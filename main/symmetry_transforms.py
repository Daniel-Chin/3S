from typing import Callable, Dict, List, Tuple
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from torchWork import DEVICE

__all__ = [
    'Transform', 'TUT', 'Range', 
    'Trivial', 'Translate', 'Rotate', 
    'SymmetryAssumption', 
]

Transform = Callable[[torch.Tensor], torch.Tensor]
TUT = Tuple[Transform, Transform]
Range = Tuple[int, int]

class Group(metaclass=ABCMeta):
    @abstractmethod
    def sample(self) -> TUT:
        raise NotImplemented

def identity(x):
    return x

class Trivial(Group):   # The Trivial Group
    def sample(self) -> TUT:
        return identity, identity

class Translate(Group):
    def __init__(self, n_dim: int, std: float) -> None:
        self.n_dim = n_dim
        self.std = std

    def sample(self) -> TUT:
        '''
        The second dimension is the spatial coordinates. 
        e.g. input is shape (100, 3). 
        '''
        translate = self.std * torch.randn(
            (1, self.n_dim), device=DEVICE, dtype=torch.float32, 
        )
        def transform(x: torch.Tensor):
            return x + translate
        def untransform(x: torch.Tensor):
            return x - translate
        return transform, untransform

class Rotate(Group):
    def __init__(self, n_dim: int) -> None:
        self.n_dim = n_dim
    
    def sample(self) -> TUT:
        A = torch.randn(
            (self.n_dim, self.n_dim), 
            dtype=torch.float32, device=DEVICE, 
        )
        rotate, _ = torch.linalg.qr(A)
        unrotate = torch.linalg.inv(rotate)
        def transform(x: torch.Tensor):
            return (x @ rotate)
        def untransform(x: torch.Tensor):
            return (x @ unrotate)
        return transform, untransform

class SymmetryAssumption:
    def __init__(self) -> None:
        self.latent_dim: int
        self.rule: List[Tuple[Range, List[Group]]] = {}
    
    def ready(self):
        acc = 0
        for (start, stop), _ in self.rule:
            assert start == acc
            assert stop > start
            acc = stop
        assert acc == self.latent_dim
    
    def apply(
        self, x: torch.Tensor, /, 
        instance: List[Tuple[Range, List[TUT]]], 
        trans_or_untrans: int, 
    ): 
        out = []
        for (start, stop), tut_seq in instance:
            x_slice = x[:, start:stop]
            for tut in (
                identity if trans_or_untrans == 0 else reversed
            )(tut_seq):
                f = tut[trans_or_untrans]
                x_slice = f(x_slice)
            out.append(x_slice)
        return torch.cat(out, dim=1)
    
    def sample(self) -> TUT:
        # instantiate
        instance: List[Tuple[Range, List[TUT]]] = []
        for dim_range, group_seq in self.rule:
            tut_seq: List[TUT] = []
            instance.append((dim_range, tut_seq))
            for group in group_seq:
                tut_seq.append(group.sample())
        
        # def
        def trans(x):
            return self.apply(x, instance, 0)
        def untrans(x):
            return self.apply(x, instance, 1)
        
        return trans, untrans

def test(size=100):
    symm = SymmetryAssumption()
    symm.latent_dim = 3
    symm.rule = [
        ((0, 2), [Translate(2, 1), Rotate(2)]), 
        ((2, 3), [Trivial()]), 
    ]
    symm.ready()

    points = torch.randn((size, 3))
    trans, untrans = symm.sample()
    poof = trans(points)
    print('trans untrans', (points - untrans(poof)).norm(2))
    print('altitude change', (points[:, 2] - poof[:, 2]).norm(2))
    from matplotlib import pyplot as plt
    from random import random
    n_points = 10
    for _ in range(8):
        theta = random() * 2 * np.pi
        k = np.tan(theta)
        b = (random() - .5) * 2
        X = torch.linspace(-2, 2, n_points)
        Y = k * X + b
        points = torch.zeros((n_points, 3))
        points[:, 0] = X
        points[:, 1] = Y
        poof = trans(points)
        plt.scatter(X, Y, c='b')
        plt.scatter(poof[:, 0], poof[:, 1], c='r')
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    test()
