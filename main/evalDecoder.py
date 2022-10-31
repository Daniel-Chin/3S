from os import path
import tkinter as tk
from typing import List
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except ImportError:
    pass    # not Windows

import torch
from torchWork import loadExperiment, DEVICE
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME, loadLatestModels
from PIL import ImageTk
from PIL.Image import Resampling

from shared import torch2PIL
from vae import VAE
from template_bounce import MyExpGroup

EXPERIMENT_PATH = path.join('./experiments', '''
2022_m10_d28@13_46_57_energy
'''.strip())
LOCK_EPOCH = None

RADIUS = 2
TICK_INTERVAL = .5
IMG_SIZE = 500

class UI:
    def __init__(
        self, groups: List[MyExpGroup], n_rand_inits, 
        experiment_path, lock_epoch, 
    ):
        self.win = tk.Tk()
        self.win.title('Eval decoder')

        self.groups = groups
        self.n_row = n_rand_inits
        self.n_col = len(groups)
        self.vaes = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.photoLabels: List[List[tk.Label]] = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.photos = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.max_latent_dim = 0
        variable_names = set()
        for col_i, group in enumerate(groups):
            self.max_latent_dim = max(
                self.max_latent_dim, 
                group.hyperParams.symm.latent_dim, 
            )
            variable_names.add(group.variable_name)
            groupLabel = tk.Label(
                self.win, text=group.variable_value, 
            )
            groupLabel.grid(
                row = 1, 
                column = self.max_latent_dim + col_i, 
                # padx=5, pady=5, 
            )
            for row_i in range(self.n_row):
                epoch, models = loadLatestModels(
                    experiment_path, group, row_i, dict(
                        vae=VAE, 
                    ), lock_epoch, 
                )
                vae = models['vae']
                vae.eval()
                self.vaes[row_i][col_i] = vae

                label = tk.Label(self.win)
                label.grid(
                    row = row_i + 2, 
                    column = self.max_latent_dim + col_i, 
                    padx=5, pady=5, 
                )
                self.photoLabels[row_i][col_i] = label

        self.z = torch.zeros((
            self.max_latent_dim, 
        ), dtype=torch.float, device=DEVICE)
        topLabel = tk.Label(self.win, text=variable_names.pop())
        assert not variable_names
        topLabel.grid(
            row=0, 
            column=self.max_latent_dim, 
            columnspan=self.n_col, 
            # padx=5, pady=5, 
        )

        self.sliders: List[tk.Scale] = []
        self.initSliders()

    def initSliders(self):
        for i in range(self.max_latent_dim):
            slider = tk.Scale(
                self.win,
                variable=tk.DoubleVar(value=self.z[i]),
                command=lambda value, index=i : (
                    self.onSliderUpdate(index, value)
                ),
                from_=+ RADIUS, to=- RADIUS,
                resolution=0.01, 
                tickinterval=TICK_INTERVAL if i == 0 else 0, 
                length=2000, width=100, sliderlength=100,
            )
            slider.grid(
                row=0, rowspan=self.n_row + 2, column=i, 
                padx=10, 
            )
            self.sliders.append(slider)

    def onSliderUpdate(self, index, value):
        value = float(value)
        self.z[index] = value
        # print(self.z)
        self.sliders[index].set(value)
        for row_i, vae_row in enumerate(self.vaes):
            for col_i, vae in enumerate(vae_row):
                img = decode(vae, self.z)
                img = img.resize((
                    IMG_SIZE, IMG_SIZE, 
                ), resample=Resampling.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self.photos[row_i][col_i] = photo
                self.photoLabels[row_i][col_i].config(image=photo)

def decode(vae: VAE, z: torch.Tensor):
    recon = vae.decode(z.unsqueeze(0))
    return torch2PIL(recon[0, :, :, :])

def main(experiment_path, lock_epoch):
    exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
        experiment_path, EXPERIMENT_PY_FILENAME, 
    ))
    print(f'{exp_name = }')
    ui = UI(groups, n_rand_inits, experiment_path, lock_epoch)
    ui.win.mainloop()

if __name__ == '__main__':
    main(EXPERIMENT_PATH, LOCK_EPOCH)
