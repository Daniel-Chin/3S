import numpy as np
import torch

from shared import LATENT_DIM

assert LATENT_DIM == 3

def sampleTranslate(device, std=1):
    '''
    The second dimension is the spatial coordinates. 
    e.g. input is shape (100, 3). 
    '''
    translate = std * torch.randn(
        (3, 1), device=device, dtype=torch.float32, 
    )
    translate[2] = 0
    def transform(x: torch.Tensor):
        return (x.T + translate).T
    def untransform(x: torch.Tensor):
        return (x.T - translate).T
    return transform, untransform

def sampleRotate(device):
    '''
    The second dimension is the spatial coordinates. 
    e.g. input is shape (100, 3). 
    '''
    theta = np.random.uniform(0, 2 * np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    rotate   = torch.tensor([
        [c, s, 0], [-s, c, 0], [0, 0, 1], 
    ], device=device, dtype=torch.float32).T
    unrotate = torch.tensor([
        [c, -s, 0], [s, c, 0], [0, 0, 1], 
    ], device=device, dtype=torch.float32).T
    def transform(x: torch.Tensor):
        return (rotate @ x.T).T
    def untransform(x: torch.Tensor):
        return (unrotate @ x.T).T
    return transform, untransform

def sampleTR(device, translate_std=1):
    '''
    The second dimension is the spatial coordinates. 
    e.g. input is shape (100, 3). 
    '''
    t, unt = sampleTranslate(device, translate_std)
    r, unr = sampleRotate(device)
    def transform(x: torch.Tensor):
        return t(r(x))
    def untransform(x: torch.Tensor):
        return unr(unt(x))
    return transform, untransform

def identity(x):
    return x

def test(size=100):
    points = torch.randn((size, 3))
    trans, untrans = sampleTR(torch.device("cpu"))
    poof = trans(points)
    print('trans untrans', (points - untrans(poof)).norm(2))
    print('altitude change', (points[2, :] - poof[2, :]).norm(2))
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
