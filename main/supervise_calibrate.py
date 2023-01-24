import torch
from torch.linalg import lstsq
import torch.nn.functional as F

from vae import VAE

def superviseCalibrate(
    vae: VAE, image_set: torch.Tensor, traj_set: torch.Tensor, 
):
    vae.eval()
    with torch.no_grad():
        Z, _ = vae.encode(image_set)
    padded_traj = F.pad(traj_set, (1, 0), value=1)
    solution, _, _, _ = lstsq(padded_traj, Z)
    solution: torch.Tensor
    return lambda x : (
        F.pad(x, (1, 0), value=1) @ solution
    )
