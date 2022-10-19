import os
import pickle
import numpy as np
import torch
from PIL import Image

from makeDataset import PATH

def loadDataset(device):
    prev_cwd = os.getcwd()
    os.chdir(PATH)
    with open('root.pickle', 'rb') as f:
        root = pickle.load(f)
    images = []
    coordinates = []
    for filename, x in root:
        np_img = np.array(Image.open(filename))[:, :, 1]
        torch_img = (
            torch.from_numpy(np_img).float() / 128
        ).unsqueeze(0)
        images.append(torch_img)
        coordinates.append(torch.tensor(x))
    images      = torch.stack(images     , dim=0).to(device)
    coordinates = torch.stack(coordinates, dim=0).to(device)
    os.chdir(prev_cwd)
    return images, coordinates.unsqueeze(1)

if __name__ == '__main__':
    images, coordinates = loadDataset(
        torch.device("cpu"), 
    )
    from console import console
    console({**globals(), **locals()})
