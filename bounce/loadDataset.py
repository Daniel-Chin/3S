import os
import pickle
import numpy as np
import torch
from PIL import Image

from shared import IMG_N_CHANNELS
from render_dataset import (
    TRAIN_PATH, VALIDATE_PATH, SEQ_LEN, RESOLUTION, 
)

def loadDataset(dataset_path, device):
    prev_cwd = os.getcwd()
    os.chdir(dataset_path)
    list_dir = os.listdir()
    n_datapoints = max([int(x) for x in list_dir]) + 1
    assert n_datapoints == len(list_dir)
    dataset = torch.zeros((
        n_datapoints, 
        SEQ_LEN, 
        IMG_N_CHANNELS, 
        RESOLUTION, 
        RESOLUTION, 
    ))
    for data_i in range(n_datapoints):
        for t in range(SEQ_LEN):
            img = Image.open(os.path.join(
                str(data_i), f'{t}.png', 
            ))
            torchImg = img2Tensor(img)
            for c in range(IMG_N_CHANNELS):
                dataset[
                    data_i, t, c, :, :
                ] = torchImg[:, :, c]
    os.chdir(prev_cwd)
    return dataset.to(device)

def img2Tensor(img):
    np_img = np.asarray(img)
    return (
        torch.from_numpy(np_img / 256).float()
    )

if __name__ == '__main__':
    dataset = loadDataset(
        TRAIN_PATH, torch.device("cpu"), 
    )
    from console import console
    console({**globals(), **locals()})
