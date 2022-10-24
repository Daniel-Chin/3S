from typing import Callable, List, Set, Tuple
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
import torch
from torchWork import DEVICE

__all__ = [
    'Transform', 'TUT', 'Slice', 
    'Trivial', 'Translate', 'Rotate', 
    'SymmetryAssumption', 
]

Transform = Callable[[torch.Tensor], torch.Tensor]
TUT = Tuple[Transform, Transform]

class Slice:
    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop
    
    def __repr__(self) -> str:
        return f'{self.start}:{self.stop}'

    def __hash__(self):
        return hash(self.start) ^ hash(self.stop)

class Group(metaclass=ABCMeta):
    @abstractmethod
    def sample(self) -> TUT:
        raise NotImplemented

def identity(x):
    return x

class Trivial(Group):   # The Trivial Group
    def sample(self) -> TUT:
        return identity, identity
    
    def __repr__(self) -> str:
        return 'I()'

class Translate(Group):
    def __init__(self, n_dim: int, std: float) -> None:
        self.n_dim = n_dim
        self.std = std
    
    def __repr__(self) -> str:
        return f'T({self.n_dim}, {self.std})'

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
    
    def __repr__(self) -> str:
        return f'R({self.n_dim})'

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

class Cattor:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[Slice, torch.Tensor]] = []
    
    def eat(self, dim_slice: Slice, data: torch.Tensor):
        self.buffer.append((dim_slice, data))
    
    def cat(self, dim: int):
        self.buffer.sort(key = lambda x : x[0].start)
        return torch.cat([
            data for _, data in self.buffer
        ], dim=dim)

class SymmetryAssumption:
    Instance = List[Tuple[List[TUT], Set[Slice]]]
    
    def __init__(
        self, latent_dim: int, 
        rule: List[Tuple[List[Group], Set[Slice]]], 
    ) -> None:
        self.latent_dim = latent_dim
        self.rule = rule

        dim_set = set()
        for _, slice_set in self.rule:
            for dim_slice in slice_set:
                new_dim_set = set(range(dim_slice.start, dim_slice.stop))
                assert not dim_set.intersection(new_dim_set)
                dim_set.update(new_dim_set)
        assert dim_set == set(range(self.latent_dim))
    
    def __repr__(self) -> str:
        return f'<symm {self.rule}>'
    
    def apply(
        self, x: torch.Tensor, /, 
        instance: Instance, 
        trans_or_untrans: int, 
    ): 
        cattor = Cattor(self.latent_dim)
        for tut_seq, slice_set in instance:
            for dim_slice in slice_set:
                x_slice = x[:, dim_slice.start : dim_slice.stop]
                for tut in (
                    identity if trans_or_untrans == 0 else reversed
                )(tut_seq):
                    f = tut[trans_or_untrans]
                    x_slice = f(x_slice)
                cattor.eat(dim_slice, x_slice)
        return cattor.cat(dim=1)
    
    def sample(self) -> TUT:
        # instantiate
        instance: __class__.Instance = []
        for group_seq, slice_set in self.rule:
            tut_seq: List[TUT] = []
            instance.append((tut_seq, slice_set))
            for group in group_seq:
                tut_seq.append(group.sample())
        
        # def
        def trans(x):
            return self.apply(x, instance, 0)
        def untrans(x):
            return self.apply(x, instance, 1)
        
        return trans, untrans
    
    def copy(self):
        other = __class__(
            self.latent_dim, 
            deepcopy(self.rule), 
        )

def test(size=100):
    symm = SymmetryAssumption(3, [
        ([Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        ([Trivial()], {Slice(2, 3)}), 
    ])

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
