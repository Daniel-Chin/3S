from shared import *
from vae import VAE
from rnn import RNN
from train import HAS_CUDA

def loadModels(config: Config):
    # future: load model from disk
    vae = VAE(config)
    rnn = RNN(config)
    if HAS_CUDA:
        vae = vae.cuda()
        rnn = rnn.cuda()
    return vae, rnn
