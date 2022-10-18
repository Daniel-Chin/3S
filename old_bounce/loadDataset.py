import os
import pickle

import numpy as np
import torch
from PIL import Image
import tqdm

from shared import *
from physics import Body

def loadDataset(dataset_path, size, device):
    prev_cwd = os.getcwd()
    os.chdir(dataset_path)
    # list_dir = os.listdir()
    # n_datapoints = max([int(x) for x in list_dir]) + 1
    # assert n_datapoints == len(list_dir)
    video_set = torch.zeros((
        size, 
        SEQ_LEN, 
        IMG_N_CHANNELS, 
        RESOLUTION, 
        RESOLUTION, 
    ))
    label_set = torch.zeros((
        size, 
        SEQ_LEN, 
        SPACE_DIM, 
    ))
    for data_i in tqdm.trange(size, desc='load dataset'):
        with open(os.path.join(
            str(data_i), TRAJ_FILENAME, 
        ), 'rb') as f:
            trajectory = pickle.load(f)
        for t in range(SEQ_LEN):
            label_set[data_i, t, :] = torch.from_numpy(
                trajectory[t].position, 
            )
            img = Image.open(os.path.join(
                str(data_i), f'{t}.png', 
            ))
            torchImg = img2Tensor(img)
            for c in range(IMG_N_CHANNELS):
                video_set[
                    data_i, t, c, :, :
                ] = torchImg[:, :, c]
    os.chdir(prev_cwd)
    return video_set.to(device), label_set.to(device)

def img2Tensor(img):
    np_img = np.asarray(img)
    return (
        torch.from_numpy(np_img / 256).float()
    )

if __name__ == '__main__':
    dataset = loadDataset(
        TRAIN_PATH, TRAIN_SET_SIZE, torch.device("cpu"), 
    )
    from console import console
    console({**globals(), **locals()})
