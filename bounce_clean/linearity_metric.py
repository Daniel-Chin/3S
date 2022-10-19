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
    residuals: torch.Tensor

    if residuals.numel():
        print('Tell dev: pyTorch returned residuals!')
        # https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html  
        # Semantically `residuals` should not be empty.  
        # Either pyTorch messed up or I don't understand LA.  

        # return residuals.mean().cpu() / LATENT_DIM
        # Still, before using it, test if residuals equals to what we think. 
    
    coef = solution_centered
    intercept = Y_mean - X_mean @ solution_centered

    Y_hat = X @ coef + intercept

    return (Y - Y_hat).square().mean().cpu()
