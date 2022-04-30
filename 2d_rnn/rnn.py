from torch import nn
from vae import LATENT_DIM

HIDDEN_DIM = 16

class RNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.stepTime = nn.Sequential(
            nn.Linear(HIDDEN_DIM + LATENT_DIM, HIDDEN_DIM), 
            nn.LeakyReLU(), 
        )
        
        self.projHead = nn.Linear(HIDDEN_DIM, LATENT_DIM)
