import torch
from torch import nn
from shared import *

class PredRNN(nn.Module):
    def __init__(self, hyperParams: HyperParams) -> None:
        super().__init__()
        self.hParams = hyperParams
        hidden_dim = hyperParams.rnn_width
        ensemble = hyperParams.rnn_ensemble

        self.updateHidden = [nn.Sequential(
            nn.Linear(hidden_dim + hyperParams.symm.latent_dim, hidden_dim), 
            nn.Tanh(), 
        ) for _ in range(ensemble)]
        self.dropout = nn.Dropout(hyperParams.dropout)
        
        self.projHead = [nn.Linear(
            hidden_dim, hyperParams.symm.latent_dim, 
        ) for _ in range(ensemble)]
        if hyperParams.vvrnn:
            self.logVarHead = [nn.Linear(
                hidden_dim, hyperParams.symm.latent_dim, 
            ) for _ in range(ensemble)]

        self.hidden = None
        
        print('Pred RNN # of params:', sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ), flush=True)
    
    def zeroHidden(self, batch_size, device):
        self.hidden = [
            torch.zeros((
                batch_size, self.hParams.rnn_width, 
            ), device=device) 
            for _ in range(self.hParams.rnn_ensemble)
        ]
    
    def stepTime(self, z_t: torch.Tensor):
        '''
        `z` is (BATCH_SIZE, SEQ_LEN, latent_dim). 
        '''
        for i in range(self.hParams.rnn_ensemble):
            output = self.updateHidden[i](torch.cat((
                self.dropout(self.hidden[i]), z_t, 
            ), dim=1))
            if self.hParams.residual:
                self.hidden[i] = self.hidden[i] * self.hParams.residual + output
            else:
                self.hidden[i] = output

class EnergyRNN(nn.Module):
    def __init__(self, hyperParams: HyperParams) -> None:
        super().__init__()
        self.hParams = hyperParams
        hidden_dim = hyperParams.rnn_width

        self.updateHidden = nn.Sequential(
            nn.Linear(hidden_dim + hyperParams.symm.latent_dim, hidden_dim), 
            nn.Tanh(), 
        )
        
        self.projHead = nn.Linear(hidden_dim, 1)

        self.hidden = None
        
        print('Energy RNN # of params:', sum(
            p.numel() for p in self.parameters() 
            if p.requires_grad
        ), flush=True)
    
    def zeroHidden(self, batch_size, device):
        self.hidden = torch.zeros((
            batch_size, self.hParams.rnn_width, 
        ), device=device)
    
    def stepTime(self, z_t: torch.Tensor):
        '''
        `z_t` is (BATCH_SIZE, latent_dim). 
        '''
        output = self.updateHidden(torch.cat((
            self.hidden, z_t, 
        ), dim=1))
        if self.hParams.residual:
            self.hidden = self.hidden + output
        else:
            self.hidden = output
