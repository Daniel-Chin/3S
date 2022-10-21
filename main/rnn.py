import torch
from torch import nn
from shared import *

class RNN(nn.Module):
    def __init__(self, hyperParams: HyperParams) -> None:
        super().__init__()
        self.hParams = hyperParams
        hidden_dim = hyperParams.rnn_width

        self.updateHidden = nn.Sequential(
            nn.Linear(hidden_dim + hyperParams.symm.latent_dim, hidden_dim), 
            nn.Tanh(), 
        )
        
        self.projHead = nn.Linear(hidden_dim, hyperParams.symm.latent_dim)
        if hyperParams.vvrnn:
            self.logVarHead = nn.Linear(hidden_dim, hyperParams.symm.latent_dim)

        self.hidden = None
        
        print('RNN # of params:', sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ), flush=True)
    
    def zeroHidden(self, batch_size, device):
        self.hidden = torch.zeros((
            batch_size, self.hParams.rnn_width, 
        ), device=device)
    
    def stepTime(self, z_t: torch.Tensor):
        '''
        `z` is (BATCH_SIZE, SEQ_LEN, latent_dim). 
        '''
        output = self.updateHidden(torch.cat((
            self.hidden, z_t, 
        ), dim=1))
        if self.hParams.residual:
            self.hidden = self.hidden + output
        else:
            self.hidden = output
