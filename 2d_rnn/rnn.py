import torch
from torch import nn
from vae import LATENT_DIM

class RNN(nn.Module):
    def __init__(self, hidden_dim=16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.updateHidden = nn.Sequential(
            nn.Linear(hidden_dim + LATENT_DIM, hidden_dim), 
            nn.Tanh(), 
        )
        
        self.projHead = nn.Linear(hidden_dim, LATENT_DIM)

        self.hidden = None

        print('RNN # of params:', sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ))
    
    def zeroHidden(self, batch_size, device):
        self.hidden = torch.zeros((
            batch_size, self.hidden_dim, 
        )).to(device)
    
    def stepTime(self, z: torch.Tensor, time: int, trans):
        '''
        `z` is (BATCH_SIZE, SEQ_LEN, LATENT_DIM). 
        '''
        self.hidden = self.updateHidden(
            torch.cat((self.hidden, trans(
                z[:, time, :].T, 
            ).T), dim=1)
        )
