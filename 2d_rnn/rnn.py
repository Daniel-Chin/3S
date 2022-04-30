import torch
from torch import nn
from vae import LATENT_DIM

HIDDEN_DIM = 16

class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.updateHidden = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, HIDDEN_DIM), 
            nn.LeakyReLU(), 
        )
        
        self.projHead = nn.Linear(HIDDEN_DIM, LATENT_DIM)

        self.hidden = None
    
    def zeroHidden(self, batch_size):
        self.hidden = torch.zeros((batch_size, HIDDEN_DIM))
    
    def stepTime(self, z: torch.Tensor, time: int):
        '''
        `z` is (BATCH_SIZE, SEQ_LEN, LATENT_DIM). 
        '''
        self.hidden = self.updateHidden(
            torch.cat((self.hidden, z[:, time, :]), dim=1)
        )
