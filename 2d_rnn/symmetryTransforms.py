import numpy as np
import torch

def sampleTransforms(translate_std=1):
    theta = np.random.uniform(0, 2 * np.pi)
    c = np.cos(theta)
    s = np.sin(theta)
    rotate   = torch.Tensor([[c, s], [-s, c]])
    unrotate = torch.Tensor([[c, -s], [s, c]])
    translate   = torch.randn((2, 1)) * translate_std
    def transform(x):
        return rotate @ x + translate
    def untransform(x):
        return unrotate @ (x - translate)
    return transform, untransform

def test(size=100):
    points = torch.randn((2, size))
    trans, untrans = sampleTransforms()
    print((points - untrans(trans(points))).norm(2))

if __name__ == '__main__':
    test()
