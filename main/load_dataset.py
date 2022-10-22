import os
import json
from typing import List

import numpy as np
import torch
import torch.utils.data
from PIL import Image
import tqdm
from physics_shared import Body

from shared import *

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, dataset_path, size, 
        ACTUAL_DIM: int, device=None, 
    ) -> None:
        super().__init__()

        prev_cwd = os.getcwd()
        os.chdir(dataset_path)
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
            ACTUAL_DIM, 
        ))
        for data_i in tqdm.trange(size, desc='load dataset'):
            with open(os.path.join(
                str(data_i), TRAJ_FILENAME, 
            ), 'r') as f:
                trajectory: List[List[Body]] = []
                for bodies_json in json.load(f):
                    bodies = []
                    trajectory.append(bodies)
                    for body_json in bodies_json:
                        bodies.append(Body().fromJSON(body_json))
            for t in range(SEQ_LEN):
                for body_i, body in enumerate(trajectory[t]):
                    label_set[
                        data_i, t, 3 * body_i : 3 * (body_i + 1)
                    ] = torch.from_numpy(body.position)
                img = Image.open(os.path.join(
                    str(data_i), f'{t}.png', 
                ))
                torchImg = img2Tensor(img)
                for c in range(IMG_N_CHANNELS):
                    video_set[
                        data_i, t, c, :, :
                    ] = torchImg[:, :, c]
        os.chdir(prev_cwd)

        if device is not None:
            print('Moving dataset to GPU...', flush=True)
            video_set = video_set.to(device)
            label_set = label_set.to(device)

        self.size = size
        self.video_set = video_set
        self.label_set = label_set
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return (
            self.video_set[index, :, :, :, :], 
            self.label_set[index, :, :], 
        )

def img2Tensor(img):
    np_img = np.asarray(img)
    return (
        torch.from_numpy(np_img / 256).float()
    )

def PersistentLoader(dataset, batch_size):
    while True:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, 
            num_workers=0, 
        )
        for video_batch, traj_batch in loader:
            if video_batch.shape[0] != batch_size:
                break
            yield video_batch, traj_batch

if __name__ == '__main__':
    dataset = Dataset('../datasets/bounce/train', 128, 3)
    loader = PersistentLoader(dataset, 32)
    for i, (x, y) in enumerate(loader):
        print(i, x.shape, y.shape)
