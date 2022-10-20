import torch
from torch.linalg import lstsq
from shared import *

def projectionMSE(X: torch.Tensor, Y: torch.Tensor):
    # A better method is to pad 1. I didn't know that! 
    Y /= Y.std(dim=0)

    X_mean = X.mean(dim=0)
    X_centered = X - X_mean 
    Y_mean = Y.mean(dim=0)
    Y_centered = Y - Y_mean 

    solution_centered, residuals, _, _ = lstsq(X_centered, Y_centered)
    solution_centered: torch.Tensor

    # Some versions of pyTorch has bugs with `residuals`. 
    # Let's not use it. It doesn't save much time anyways. 
    
    coef = solution_centered
    intercept = Y_mean - X_mean @ solution_centered

    Y_hat = X @ coef + intercept

    return (Y - Y_hat).square().mean().cpu()
