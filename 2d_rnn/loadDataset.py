import os
import pickle
import numpy as np
import torch
from PIL import Image

from makeDataset import (
    TRAIN_PATH, VALIDATE_PATH, SEQ_LEN, RESOLUTION, 
    GIF_INTERVAL, 
)

def loadDataset(dataset_path, device):
    prev_cwd = os.getcwd()
    os.chdir(dataset_path)
    with open('root.pickle', 'rb') as f:
        root = pickle.load(f)
    dataset = torch.zeros((
        len(root), 
        SEQ_LEN, 
        1, 
        RESOLUTION, 
        RESOLUTION, 
    ))
    for data_i, (filename, _) in enumerate(root):
        img = Image.open(filename)
        t = 0
        while True:
            for _ in range(
                img.info['duration'] // GIF_INTERVAL, 
            ):
                dataset[data_i, t, 0, :, :] = img2Tensor(img)
                t += 1
            try:
                img.seek(img.tell() + 1)
            except EOFError:
                break
        assert t == SEQ_LEN
    os.chdir(prev_cwd)
    return dataset.to(device)

def img2Tensor(img):
    np_img = np.array(img)
    return (
        torch.from_numpy(np_img).float()
    )

if __name__ == '__main__':
    dataset = loadDataset(
        TRAIN_PATH, torch.device("cpu"), 
    )
    from console import console
    console({**globals(), **locals()})
