from typing import Callable, List, Set, Tuple
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import random

import numpy as np
import torch
from torchWork import DEVICE

__all__ = [
    'Transform', 'TUT', 'Slice', 
    'Trivial', 'Translate', 'Rotate', 
    'SymmetryAssumption', 'GusMethod', 
    'SAMPLE_TRANS', 'COMPOSE_TRANS', 
]

Transform = Callable[[torch.Tensor], torch.Tensor]
TUT = Tuple[Transform, Transform]

class HowTransCombine: 
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return self.name
    # behave like primitives
    def __copy__(self):
        return self
    def __deepcopy__(self, _):
        return self
SAMPLE_TRANS  = HowTransCombine('SAMPLE_TRANS')
COMPOSE_TRANS = HowTransCombine('COMPOSE_TRANS')

class Slice:
    def __init__(self, start: int, stop: int) -> None:
        self.start = start
        self.stop = stop
    
    def __repr__(self) -> str:
        return f'{self.start}:{self.stop}'

    def __hash__(self):
        return hash(self.start) ^ hash(self.stop)
    
    def __eq__(self, other):
        assert isinstance(other, __class__)
        return self.start == other.start and self.stop == other.stop

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

    def __eq__(self, other):
        assert isinstance(other, __class__)
        return True

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

    def __eq__(self, other):
        assert isinstance(other, __class__)
        return self.n_dim == other.n_dim and self.std == other.std

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

    def __eq__(self, other):
        assert isinstance(other, __class__)
        return self.n_dim == other.n_dim

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
        rule: List[Tuple[HowTransCombine, List[Group], Set[Slice]]], 
        identity_prob: float = 0.0, 
    ) -> None:
        self.latent_dim = latent_dim
        self.rule = rule
        self.identity_prob = identity_prob

        dim_set = set()
        for _, _, slice_set in self.rule:
            for dim_slice in slice_set:
                new_dim_set = set(range(dim_slice.start, dim_slice.stop))
                assert not dim_set.intersection(new_dim_set)
                dim_set.update(new_dim_set)
        assert dim_set == set(range(self.latent_dim))
    
    def __repr__(self) -> str:
        return f'<symm {self.rule} I={self.identity_prob}>'
    
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
        if random.random() < self.identity_prob:
            return identity, identity
        
        # instantiate
        instance: __class__.Instance = []
        for howCombine, group_seq, slice_set in self.rule:
            tut_seq: List[TUT] = []
            instance.append((tut_seq, slice_set))
            if howCombine is SAMPLE_TRANS:
                group = random.choice(group_seq)
                tut_seq.append(group.sample())
            elif howCombine is COMPOSE_TRANS:
                for group in group_seq:
                    tut_seq.append(group.sample())
            else:
                raise ValueError(f'what is {howCombine}')
        
        # def
        def trans(x):
            return self.apply(x, instance, 0)
        def untrans(x):
            return self.apply(x, instance, 1)
        
        return trans, untrans
    
    def __deepcopy__(self, memo):
        return __class__(
            self.latent_dim, 
            deepcopy(self.rule, memo), 
            self.identity_prob, 
        )
    
    def __eq__(self, other):
        assert isinstance(other, __class__)
        return (
            self.latent_dim == other.latent_dim 
            and self.rule == other.rule
            and self.identity_prob == other.identity_prob
        )

class GusMethod(SymmetryAssumption):
    def __init__(self):
        self.latent_dim = 3
        self.t = SymmetryAssumption(
            3, [
                (COMPOSE_TRANS, [Translate(2, 1)], {Slice(0, 2)}), 
                (COMPOSE_TRANS, [Trivial()], {Slice(2, 3)}), 
            ], 
        )
        self.r = SymmetryAssumption(
            3, [
                (COMPOSE_TRANS, [Rotate(2)], {Slice(0, 2)}), 
                (COMPOSE_TRANS, [Trivial()], {Slice(2, 3)}), 
            ], 
        )
    
    def sample(self):
        if random.random() < .5:
            symm = self.t
        else:
            symm = self.r
        return symm.sample()
    
    def __repr__(self) -> str:
        return '<symm GusMethod>'
    
    def __deepcopy__(self):
        return __class__()

def test(size=100):
    # symm = SymmetryAssumption(3, [
    #     (COMPOSE_TRANS, [Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
    #     (COMPOSE_TRANS, [Trivial()], {Slice(2, 3)}), 
    # ])
    symm = SymmetryAssumption(3, [
        (SAMPLE_TRANS, [Translate(2, 1), Rotate(2)], {Slice(0, 2)}), 
        (SAMPLE_TRANS, [Trivial()], {Slice(2, 3)}), 
    ])
    # symm = GusMethod()

    points = torch.randn((size, 3))
    for _ in range(8):
        trans, untrans = symm.sample()
        poof = trans(points)
        print('trans untrans', (points - untrans(poof)).norm(2))
        print('altitude change', (points[:, 2] - poof[:, 2]).norm(2))
        from matplotlib import pyplot as plt
        from random import random
        n_points = 10
        for i in range(5):
            # if True:
            if i == 0:
                N = 5
                points = torch.zeros((N ** 2, 3))
                for x in range(N):
                    for y in range(N):
                        pos = (x, y)
                        for dim in range(2):
                            points[x * N + y, dim] = pos[dim] - (N-1) / 2
                
                # pp = points
            else:
                theta = random() * 2 * np.pi
                k = np.tan(theta)
                b = (random() - .5) * 2
                X = torch.linspace(-2, 2, n_points)
                Y = k * X + b
                points = torch.zeros((n_points, 3))
                points[:, 0] = X
                points[:, 1] = Y

                # points = torch.cat([points, pp])

            poof = trans(points)
            plt.scatter(points[:, 0], points[:, 1], c='b')
            plt.scatter(poof  [:, 0], poof  [:, 1], c='r')
            plt.axis('equal')
            plt.show()

if __name__ == '__main__':
    test()
