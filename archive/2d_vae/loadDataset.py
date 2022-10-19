import os
import pickle
import numpy as np
import torch
from PIL import Image

from makeDataset import TRAIN_PATH, VALIDATE_PATH

def loadDataset(dataset_path, device):
    prev_cwd = os.getcwd()
    os.chdir(dataset_path)
    with open('root.pickle', 'rb') as f:
        root = pickle.load(f)
    images = []
    coordinates = []
    for filename, (x, y) in root:
        img = Image.open(filename)
        images.append(img2Tensor(img))
        coordinates.append(torch.Tensor((x, y)))
    images      = torch.stack(images     , dim=0).to(device)
    coordinates = torch.stack(coordinates, dim=0).to(device)
    os.chdir(prev_cwd)
    return images, coordinates

def img2Tensor(img):
    np_img = np.array(img)[:, :, 1]
    return (
        torch.from_numpy(np_img).float() / 128
    ).unsqueeze(0)

if __name__ == '__main__':
    images, coordinates = loadDataset(
        TRAIN_PATH, torch.device("cpu"), 
    )
    from console import console
    console({**globals(), **locals()})
