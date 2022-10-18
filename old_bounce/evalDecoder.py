import os
from os import path
import importlib.util

import torch
import tkinter as tk
from PIL import ImageTk

from shared import *
from vae import LATENT_DIM, VAE
from loadModels import loadModels
from train import DEVICE

EXP_PATH = 'C:/Users/iGlop/d/symmetry/danRepo/bounce/results/sigmoid'
LOCK_EPOCH = None

RADIUS = 2
TICK_INTERVAL = 0.05

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

class TestUI:
    def __init__(self, vae, exp_name):
        self.win = tk.Tk()
        self.win.title(exp_name)
        self.z = torch.zeros(
            (LATENT_DIM, ), dtype=torch.float, 
        ).to(DEVICE)
        self.sliders = []
        self.initSliders()
        self.photo = None
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.RIGHT)
        self.vae: VAE = vae

    def onSliderUpdate(self, value, index):
        value = float(value)
        self.z[index] = value
        print(self.z)
        self.sliders[index].set(value)
        img = self.decode()
        img = img.resize((350, 350))
        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)

    def decode(self):
        recon = self.vae.decode(self.z.unsqueeze(0))
        return torch2PIL(recon[0, :, :, :])

    def initSliders(self):
        for i in range(0, LATENT_DIM):
            slider = tk.Scale(
                self.win,
                variable=tk.DoubleVar(value=self.z[i]),
                command=lambda value, index=i : (
                    self.onSliderUpdate(value, index)
                ),
                from_=- RADIUS, to=+ RADIUS,
                resolution=0.01, tickinterval=TICK_INTERVAL, 
                length=600,
            )
            slider.pack(side=tk.LEFT)
            self.sliders.append(slider)

def main():
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
            print('rand init', rand_init_i)
            vae, _ = getModels(rand_init_i, config)
            vae.eval()
            test_ui = TestUI(vae, exp_name)
            test_ui.win.mainloop()

main()
