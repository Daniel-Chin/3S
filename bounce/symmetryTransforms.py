import numpy as np
import torch

from vae import LATENT_DIM

assert LATENT_DIM == 3

def sampleTransforms(device, translate_std=1):
    '''
    The first dimension is the spatial coordinates. 
    e.g. input is shape (3, 100). 
    '''
    theta = np.random.uniform(0, 2 * np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    rotate   = torch.Tensor([
        [c, s, 0], [-s, c, 0], [0, 0, 1], 
    ]).T.to(device)
    unrotate = torch.Tensor([
        [c, -s, 0], [s, c, 0], [0, 0, 1], 
    ]).T.to(device)
    translate   = (torch.randn((3, 1)) * translate_std).to(
        device, 
    )
    translate[2] = 0
    def transform(x):
        return rotate @ x + translate
    def untransform(x):
        return unrotate @ (x - translate)
    return transform, untransform

def identity(x):
    return x

def test(size=100):
    points = torch.randn((3, size))
    trans, untrans = sampleTransforms(torch.device("cpu"))
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
        points = torch.zeros((3, n_points))
        points[0, :] = X
        points[1, :] = Y
        poof = trans(points)
        plt.scatter(X, Y, c='b')
        plt.scatter(poof[0, :], poof[1, :], c='r')
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    test()
