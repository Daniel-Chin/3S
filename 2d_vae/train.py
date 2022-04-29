'''
eval disentanglement: plot grid on 2D. scatter datapoints.
'''
import os
from os import path
import torch
from PIL import Image

try:
    from myTorch import LossLogger
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'myTorch', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e
from loadDataset import loadDataset, TRAIN_PATH, VALIDATE_PATH
from makeDataset import TRAIN_SET_SIZE, VALIDATE_SET_SIZE
from vae import VAE

N_EPOCHS = 1000
BATCH_SIZE = 64

RECONSTRUCT_PATH = './reconstruct'
CHECKPOINTS_PATH = './checkpoints'

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    print('We have CUDA.')
else:
    DEVICE = CPU
    print("We DON'T have CUDA.")

def loadModel():
    # future: load model from disk
    vae = VAE()
    if HAS_CUDA:
        vae = vae.cuda()
    return vae

def train(beta, rand_init_i):
    this_checkpoints_path = path.join(
        CHECKPOINTS_PATH, f'{beta}_{rand_init_i}', 
    )
    os.makedirs(this_checkpoints_path, exist_ok=True)
    vae = loadModel()
    optim = torch.optim.Adam(
        vae.parameters(), lr=.001, 
    )
    train_images,    _ = loadDataset(   TRAIN_PATH, DEVICE)
    validate_images, _ = loadDataset(VALIDATE_PATH, DEVICE)
    assert TRAIN_SET_SIZE % BATCH_SIZE == 0
    assert VALIDATE_SET_SIZE == BATCH_SIZE
    n_batches = TRAIN_SET_SIZE // BATCH_SIZE
    lossLogger = LossLogger('losses.log')
    lossLogger.clearFile()
    try:
        for epoch in range(N_EPOCHS):
            vae.train()
            epoch_recon_loss = 0
            epoch_kld___loss = 0
            for batch_pos in range(0, TRAIN_SET_SIZE, BATCH_SIZE):
                batch = train_images[
                    batch_pos : batch_pos + BATCH_SIZE
                ]
                batch_pos = (
                    batch_pos + BATCH_SIZE
                ) % TRAIN_SET_SIZE
                loss, recon_loss, kld_loss = vae.computeLoss(
                    batch, *vae.forward(batch), beta, 
                )
                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_recon_loss += recon_loss / n_batches
                epoch_kld___loss +=   kld_loss / n_batches

            vae.eval()
            with torch.no_grad():
                (
                    _, validate_recon_loss, validate_kld___loss, 
                ) = vae.computeLoss(
                    validate_images, 
                    *vae.forward(validate_images), 
                    beta, 
                )
            lossLogger.eat(
                epoch, 
                train____recon_loss=epoch_recon_loss, 
                train____kld___loss=epoch_kld___loss, 
                validate_recon_loss=validate_recon_loss, 
                validate_kld___loss=validate_kld___loss, 
            )
            if epoch % 1 == 0:
                torch.save(vae.state_dict(), path.join(
                    this_checkpoints_path, f'{epoch}.pt', 
                ))
    except KeyboardInterrupt:
        print('Received ^C.')
    # reconstructValidateSet(
    #     vae.cpu(), validate_images, 
    # )

def visualize(reconstruct):
    return Image.fromarray(
        reconstruct.numpy() * 128
    ).convert('L')

def reconstructValidateSet(vae: VAE, validate_images):
    os.makedirs(RECONSTRUCT_PATH, exist_ok=True)
    with torch.no_grad():
        reconstructions, _, _ = vae.forward(validate_images)
        for i in range(validate_images.shape[0]):
            recon = visualize(reconstructions[i, 0, :, :])
            truth = visualize(validate_images[i, 0, :, :])
            truth.save(
                path.join(RECONSTRUCT_PATH, f'{i}.gif'), 
                save_all=True, append_images=[recon], 
                duration=300, loop=0, 
            )
    print(f'Saved reconstructions to `{RECONSTRUCT_PATH}`')
