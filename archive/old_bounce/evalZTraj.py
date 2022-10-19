import os
from os import path
import importlib.util

import torch
from matplotlib import pyplot as plt

from shared import *
from vae import LATENT_DIM, VAE
from loadModels import loadModels
from train import DEVICE
from loadDataset import loadDataset, TRAIN_PATH, VALIDATE_PATH

EXP_PATH = 'C:/Users/iGlop/d/symmetry/danRepo/bounce/results/xuanjie'
LOCK_EPOCH = None

RAND_INIT_I = 0
RADIUS = 2

def getExp():
    spec = importlib.util.spec_from_file_location(
        "experiments", path.abspath('experiments.py'), 
    )
    experiments = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiments)
    return experiments

def getModels(rand_init_i, config):
    vae, rnn = loadModels(config)
    
    exp_path = renderExperimentPath(rand_init_i, config)
    if LOCK_EPOCH is None:
        max_epoch = 0
        for filename in os.listdir(exp_path):
            try:
                epoch = int(filename.split('_vae.pt', 1)[0])
            except ValueError:
                continue
            else:
                max_epoch = max(max_epoch, epoch)
        epoch = max_epoch
    else:
        epoch = LOCK_EPOCH
    print('taking epoch', epoch)
    for name, thing in (('vae', vae), ('rnn', rnn)):
        thing.load_state_dict(torch.load(path.join(
            exp_path, f'{epoch}_{name}.pt', 
        ), map_location=DEVICE))
    return vae, rnn

def main():
    train_set   , _ = loadDataset(
        TRAIN_PATH,       TRAIN_SET_SIZE, DEVICE, 
    )
    validate_set, _ = loadDataset(
        VALIDATE_PATH, VALIDATE_SET_SIZE, DEVICE, 
    )
    dataset = train_set
    # exp_path = input(
    #     'Drag experiments folder here: ', 
    # ).strip('"')
    exp_path = EXP_PATH
    os.chdir(exp_path)
    experiments = getExp()
    for exp_name, config in experiments.EXPERIMENTS:
        print(exp_name)
        print(config)
        for rand_init_i in range(experiments.RAND_INIT_TIMES):
            if rand_init_i == RAND_INIT_I:
                print('rand init', rand_init_i)
                vae, rnn = getModels(rand_init_i, config)
                vae.eval()
                evalZTraj(vae, rnn, dataset)

def evalZTraj(vae: VAE, rnn, batch):
    batch_size = batch.shape[0]
    flat_batch = batch.view(
        batch_size * SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    flat_mu, _ = vae.encode(flat_batch)
    mu = flat_mu.detach().view(batch_size, SEQ_LEN, LATENT_DIM)
    for i in range(batch_size):
        print(f'{i=}')
        plt.plot(mu[i, :, 0], mu[i, :, 1])
        if i % 8 == 0:
            plt.show()

main()
