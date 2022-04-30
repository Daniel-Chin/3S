import numpy as np
import torch

from vae import LATENT_DIM

assert LATENT_DIM == 2

def sampleTransforms(device, translate_std=1):
    '''
    The first dimension is the spatial coordinates. 
    e.g. input is shape (2, 100). 
    '''
    theta = np.random.uniform(0, 2 * np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    rotate   = torch.Tensor([[c, s], [-s, c]]).to(device)
    unrotate = torch.Tensor([[c, -s], [s, c]]).to(device)
    translate   = (torch.randn((2, 1)) * translate_std).to(
        device, 
    )
    def transform(x):
        return rotate @ x + translate
    def untransform(x):
        return unrotate @ (x - translate)
    return transform, untransform

def identity(x):
    return x

def test(size=100):
    points = torch.randn((2, size))
    trans, untrans = sampleTransforms(torch.device("cpu"))
    print((points - untrans(trans(points))).norm(2))

if __name__ == '__main__':
    test()
