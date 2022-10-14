from os import path

from torchWork import DEVICE, loadExperiment, loadLatestModels
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME
from matplotlib import pyplot as plt

from shared import *
from load_dataset import Dataset
from vae import LATENT_DIM, VAE

EXPERIMENT_PATH = path.join('./experiments', '''
supervised_rnn_width_2022_Oct_13_22;48;34
'''.strip())
RAND_INIT_I = 0
LOCK_EPOCH = None

RADIUS = 2

def main():
    dataset, _ = Dataset(
        TRAIN_PATH, 256, DEVICE, 
    )
    # dataset, _ = Dataset(
    #     VALIDATE_PATH, VALIDATE_SET_SIZE, DEVICE, 
    # )

    exp_name, n_rand_inits, groups = loadExperiment(path.join(
        EXPERIMENT_PATH, EXPERIMENT_PY_FILENAME, 
    ))
    print(f'{exp_name = }')
    for group in groups:
        print(group.name())
        group.hyperParams.print(depth=1)
        print(f'{RAND_INIT_I = }')
        vae = loadLatestModels(EXPERIMENT_PATH, group, RAND_INIT_I, dict(
            vae=VAE, 
        ), LOCK_EPOCH)['vae']
        vae.eval()
        evalZTraj(vae, dataset)

def evalZTraj(vae: VAE, dataset: Dataset):
    # Entire dataset as one batch. Not optimized for GPU! 
    batch_size = dataset.size
    flat_batch = dataset.video_set.view(
        batch_size * SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    flat_mu, _ = vae.encode(flat_batch)
    mu = flat_mu.detach().view(batch_size, SEQ_LEN, LATENT_DIM)
    for i in range(batch_size):
        print(f'{i=}')
        plt.plot(mu[i, :, 0], mu[i, :, 1])
        if i % 8 == 0:
            plt.show()

main()
