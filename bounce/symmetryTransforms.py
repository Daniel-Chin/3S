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
    print((points - untrans(trans(points))).norm(2))

if __name__ == '__main__':
    test()
