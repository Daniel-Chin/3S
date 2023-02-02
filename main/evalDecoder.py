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
from tqdm import tqdm

from shared import torch2PIL
from load_dataset import getImageSet, Dataset
from vae import VAE
from supervise_calibrate import superviseCalibrate
from template_bounce import MyExpGroup

try:
    from workspace import EXP_PATH, LOCK_EPOCH, filterGroups, CAP_N_RAND_INITS
except ImportError:
    EXP_PATH = input('EXP_PATH=')
    LOCK_EPOCH = None

SUPERVISE_CALIBRATE = True
CALIBRATE_SET_SIZE = 4  # number of videos

RADIUS = 2
TICK_INTERVAL = .5
IMG_SIZE = 300

class UI:
    def __init__(
        self, groups: List[MyExpGroup], n_rand_inits, 
        experiment_path, lock_epoch, experiment, 
    ):
        computeDist = self.ComputeDist(experiment)

        self.win = tk.Tk()
        self.win.title('Eval decoder')

        self.groups = groups
        self.n_row = n_rand_inits
        self.n_col = len(groups)
        self.vaes: List[List[VAE]] = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.projHeads = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.photoLabels: List[List[tk.Label]] = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.photos = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        self.z_dists = [
            [None] * self.n_col for _ in range(self.n_row)
        ]
        max_latent_dim = 0
        for col_i, group in enumerate(groups):
            group.hyperParams.fillDefaults()
            max_latent_dim = max(
                max_latent_dim, 
                group.hyperParams.symm.latent_dim, 
            )
        if SUPERVISE_CALIBRATE:
            self.n_sliders = experiment.DATASET_INSTANCE.ACTUAL_DIM
        else:
            self.n_sliders = max_latent_dim
        variable_names = set()
        for col_i, group in enumerate(groups):
            variable_names.add(group.variable_name)
            groupLabel = tk.Label(
                self.win, text=group.variable_value, 
            )
            groupLabel.grid(
                row = 1, 
                column = self.n_sliders + col_i, 
                # padx=5, pady=5, 
            )
            for row_i in range(self.n_row):
                epoch, models = loadLatestModels(
                    experiment_path, group, row_i, dict(
                        vae=(VAE, 1), 
                    ), lock_epoch, 
                )
                vae = models['vae'][0]
                vae.eval()
                self.vaes[row_i][col_i] = vae

                label = tk.Label(self.win)
                label.grid(
                    row = row_i + 2, 
                    column = self.n_sliders + col_i, 
                    padx=5, pady=5, 
                )
                self.photoLabels[row_i][col_i] = label

                if not SUPERVISE_CALIBRATE:
                    next(computeDist)
                    mean, std = computeDist.send(vae)
                    self.z_dists[row_i][col_i] = mean, std
                    print('z space: ', end='')
                    for sym, _mean, _std in zip('xyz', mean, std):
                        print(f'{sym}: mean {_mean:+.1f}, std {_std:.1f}', end='; ')
                    print()

        self.knobs = torch.zeros((
            self.n_sliders, 
        ), dtype=torch.float, device=DEVICE)
        self.knob_lim = [
            -RADIUS * torch.ones((self.n_sliders, )), 
            +RADIUS * torch.ones((self.n_sliders, )), 
        ]
        topLabel = tk.Label(self.win, text=variable_names.pop())
        assert not variable_names
        topLabel.grid(
            row=0, 
            column=self.n_sliders, 
            columnspan=self.n_col, 
            # padx=5, pady=5, 
        )

        if SUPERVISE_CALIBRATE:
            self.initCalibrate(experiment)
        
        self.sliders: List[tk.Scale] = []
        self.initSliders()
    
    def ComputeDist(self, experiment):
        validateSet = experiment.getDataset(
            is_train_not_validate=False, size=None, 
            device=DEVICE, 
        )
        image_set, _ = getImageSet(validateSet)
        while True:
            vae: VAE = yield
            Z, _ = vae.encode(image_set)
            yield Z.mean(dim=0), Z.std(dim=0)

    def initSliders(self):
        for i, knob in enumerate(self.knobs):
            slider = tk.Scale(
                self.win,
                # variable=tk.DoubleVar(value=self.knobs[i]),
                command=lambda value, index=i : (
                    self.onSliderUpdate(index, value)
                ),
                from_=self.knob_lim[1][i].item(), 
                to   =self.knob_lim[0][i].item(),
                resolution=0.01, 
                tickinterval=TICK_INTERVAL if (
                    not SUPERVISE_CALIBRATE and i == 0
                ) else 0, 
                length=2000, width=100, sliderlength=100,
            )
            slider.set(knob.item())
            slider.grid(
                row=0, rowspan=self.n_row + 2, column=i, 
                padx=10, 
            )
            self.sliders.append(slider)

    def onSliderUpdate(self, index: int, value):
        value = float(value)
        self.knobs[index] = value
        # print(self.z_z_score)
        self.sliders[index].set(value)
        for row_i, vae_row in enumerate(self.vaes):
            for col_i, vae in enumerate(vae_row):
                if SUPERVISE_CALIBRATE:
                    z = self.projHeads[row_i][col_i](self.knobs)
                else:
                    mean, std = self.z_dists[row_i][col_i]
                    z = self.knobs * std + mean
                # if row_i == 0 and col_i == 0:
                #     print(z)
                img = decode(vae, z)
                img = img.resize((
                    IMG_SIZE, IMG_SIZE, 
                ), resample=Resampling.NEAREST)
                photo = ImageTk.PhotoImage(img)
                self.photos[row_i][col_i] = photo
                self.photoLabels[row_i][col_i].config(image=photo)
    
    def initCalibrate(self, experiment):
        validateSet: Dataset = experiment.getDataset(
            is_train_not_validate=False, size=None, 
            device=DEVICE, 
        )
        image_set, traj_set = getImageSet(validateSet.truncate(
            CALIBRATE_SET_SIZE, 
        ))
        _mean = traj_set.mean(dim=0)
        _std  = traj_set.std (dim=0)
        self.knob_lim[0] = _mean - 2 * _std
        self.knob_lim[1] = _mean + 2 * _std
        self.knobs = _mean
        
        for vae_row, projHead_row in tqdm(
            [*zip(self.vaes, self.projHeads)], 
            'supervise calibrate', 
        ):
            for col_i, vae in enumerate(vae_row):
                projHead_row[col_i] = superviseCalibrate(
                    vae, image_set, traj_set, 
                )

def decode(vae: VAE, z: torch.Tensor):
    recon = vae.decode(z.unsqueeze(0))[0, :, :, :]
    if recon.shape[0] == 1:
        return torch2PIL(recon, 'L')
    elif recon.shape[0] == 3:
        return torch2PIL(recon, 'RGB')
    assert False

def main(experiment_path, lock_epoch):
    with torch.no_grad():
        exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
            experiment_path, EXPERIMENT_PY_FILENAME, 
        ))
        groups: List[MyExpGroup]
        did_override = False
        _groups = filterGroups(groups)
        if _groups != groups:
            groups = _groups
            print('Taking exp groups:')
            print(' ', *[group.variable_value for group in groups])
            did_override = True
        if CAP_N_RAND_INITS is not None and n_rand_inits > CAP_N_RAND_INITS:
            n_rand_inits = CAP_N_RAND_INITS
            print(f'Forced {n_rand_inits = }')
            did_override = True
        if did_override:
            input('Enter...')
        print(f'{exp_name = }')
        ui = UI(groups, n_rand_inits, experiment_path, lock_epoch, experiment)
        ui.win.mainloop()

if __name__ == '__main__':
    main(EXP_PATH, LOCK_EPOCH)
