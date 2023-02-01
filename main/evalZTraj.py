from os import path

from torchWork import DEVICE, loadExperiment, loadLatestModels
from torchWork.experiment_control import EXPERIMENT_PY_FILENAME
from matplotlib import pyplot as plt

from shared import *
from load_dataset import Dataset
from vae import VAE

EXPERIMENT_PATH = path.join('./experiments', '''
supervised_rnn_width_2022_Oct_13_22;48;34
'''.strip())
RAND_INIT_I = 0
LOCK_EPOCH = None

RADIUS = 2

def main():
    exp_name, n_rand_inits, groups, experiment = loadExperiment(path.join(
        EXPERIMENT_PATH, EXPERIMENT_PY_FILENAME, 
    ))
    print(f'{exp_name = }')
    dataset: Dataset = experiment.getDataset(
        is_train_not_validate=False, size=None, device=DEVICE, 
    )

    for group in groups:
        print(group.name())
        group.hyperParams.print(depth=1)
        print(f'{RAND_INIT_I = }')
        vae = loadLatestModels(EXPERIMENT_PATH, group, RAND_INIT_I, dict(
            vae=(VAE, 1), 
        ), LOCK_EPOCH)['vae'][0]
        vae.eval()
        evalZTraj(vae, dataset, group.hyperParams, experiment)

def evalZTraj(vae: VAE, dataset: Dataset, hParams: HyperParams, experiment):
    # Entire dataset as one batch. Not optimized for GPU! 
    SEQ_LEN = experiment.DATASET_INSTANCE.SEQ_LEN
    RESOLUTION = experiment.DATASET_INSTANCE.RESOLUTION
    batch_size = dataset.size
    flat_batch = dataset.video_set.view(
        batch_size * SEQ_LEN, experiment.DATASET_INSTANCE.IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    flat_mu, _ = vae.encode(flat_batch)
    mu = flat_mu.detach().view(batch_size, SEQ_LEN, hParams.latent_dim)
    for i in range(batch_size):
        print(f'{i=}')
        plt.plot(mu[i, :, 0], mu[i, :, 1])
        if i % 8 == 0:
            plt.show()

main()
