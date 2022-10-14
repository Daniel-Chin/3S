import os
from os import path
import tkinter as tk
from typing import List

import torch
from torchWork import ExperimentGroup, loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from PIL import ImageTk

from shared import LATENT_DIM, torch2PIL
from vae import VAE
from rnn import RNN

EXPERIMENT_PATH = path.join('./experiments', '''
supervised_rnn_width_2022_Oct_13_22;48;34
'''.strip())
LOCK_EPOCH = None

RADIUS = 2
TICK_INTERVAL = 0.05

class TestUI:
    def __init__(self, vae, group_name):
        self.vae: VAE = vae
        self.win = tk.Tk()
        self.win.title(group_name)
        self.z = torch.zeros(
            (LATENT_DIM, ), dtype=torch.float, device=DEVICE, 
        )
        self.sliders: List[tk.Scale] = []
        self.photo = None
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.RIGHT)

        self.initSliders()

    def initSliders(self):
        for i in range(LATENT_DIM):
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

    def onSliderUpdate(self, value, index):
        value = float(value)
        self.z[index] = value
        # print(self.z)
        self.sliders[index].set(value)
        img = decode(self.vae, self.z)
        img = img.resize((350, 350))
        self.photo = ImageTk.PhotoImage(img)
        self.label.config(image=self.photo)

def decode(vae: VAE, z: torch.Tensor):
    recon = vae.decode(z.unsqueeze(0))
    return torch2PIL(recon[0, :, :, :])

def main():
    exp_name, n_rand_inits, groups = loadExperiment(path.join(
        EXPERIMENT_PATH, EXPERIMENT_PY_FILENAME, 
    ))
    print(f'{exp_name = }')
    for group in groups:
        print(group.name())
        group.hyperParams.print(depth=1)
        for rand_init_i in range(n_rand_inits):
            print(f'{rand_init_i = }')
            vae = loadLatestModels(EXPERIMENT_PATH, group, rand_init_i, dict(
                vae=VAE, rnn=RNN, 
            ), LOCK_EPOCH)['vae']
            vae.eval()
            test_ui = TestUI(vae, group.name())
            test_ui.win.mainloop()

main()
