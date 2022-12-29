import torch
from torch.linalg import lstsq
import torch.nn.functional as F
from shared import *

def projectionMSE(X: torch.Tensor, Y: torch.Tensor):
    Y = Y - Y.mean(dim=0)
    Y = Y / Y.std(dim=0)

    X = F.pad(X, (1, 0), value=1)

    try:
        solution, residuals, _, _ = lstsq(X, Y)
    except RuntimeError: 
        assert torch.isnan(X).any()
        print('Warning: nan encountered during linear proj. Returning -1.')
        return torch.tensor(-1)
    solution: torch.Tensor

    # Some versions of pyTorch has bugs with `residuals`. 
    # Let's not use it. It doesn't save much time anyways. 
    
    Y_hat = X @ solution

    return (Y - Y_hat).square().mean().cpu()

def projectionMSELockHeight(X: torch.Tensor, Y: torch.Tensor):
    mse_0 = projectionMSE(X[:, :2], Y[:, :2])
    mse_1 = projectionMSE(X[:, 2:], Y[:, 2:])
    return (2 * mse_0 + mse_1) / 3
