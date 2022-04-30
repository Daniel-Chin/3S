import torch
from torch import nn
from vae import LATENT_DIM

HIDDEN_DIM = 16

class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.updateHidden = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, HIDDEN_DIM), 
            nn.Tanh(), 
        )
        
        self.projHead = nn.Linear(HIDDEN_DIM, LATENT_DIM)

        self.hidden = None

        print('RNN # of params:', sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ))
    
    def zeroHidden(self, batch_size, device):
        self.hidden = torch.zeros((
            batch_size, HIDDEN_DIM, 
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
