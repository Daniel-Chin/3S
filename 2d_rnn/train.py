import os
from os import path
import torch
import torch.nn.functional as F
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
from makeDataset import (
    RESOLUTION, SEQ_LEN, TRAIN_SET_SIZE, VALIDATE_SET_SIZE, 
)
from vae import LATENT_DIM, VAE
from rnn import HIDDEN_DIM, RNN

N_EPOCHS = 1000
BATCH_SIZE = 32
VAE_LOSS_COEF = 1
RNN_LOSS_COEF = 1
RNN_MIN_CONTEXT = 3

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
    rnn = RNN()
    if HAS_CUDA:
        vae = vae.cuda()
        rnn = rnn.cuda()
    return vae, rnn

def train(rand_init_i=0, beta=0.001):
    # this_checkpoints_path = path.join(
    #     CHECKPOINTS_PATH, f'{beta}_{rand_init_i}', 
    # )
    # os.makedirs(this_checkpoints_path, exist_ok=True)
    vae, rnn = loadModel()
    optim = torch.optim.Adam(
        [*vae.parameters(), *rnn.parameters()], lr=.001, 
    )
    train_set    = loadDataset(   TRAIN_PATH, DEVICE)
    validate_set = loadDataset(VALIDATE_PATH, DEVICE)
    assert TRAIN_SET_SIZE % BATCH_SIZE == 0
    assert VALIDATE_SET_SIZE == BATCH_SIZE
    n_batches = TRAIN_SET_SIZE // BATCH_SIZE
    lossLogger = LossLogger('losses.log')
    lossLogger.clearFile()
    try:
        for epoch in range(N_EPOCHS):
            vae.train()
            rnn.train()
            epoch_recon__loss = 0
            epoch_kld____loss = 0
            epoch_z_pred_loss = 0
            for batch_pos in range(0, TRAIN_SET_SIZE, BATCH_SIZE):
                batch = train_set[
                    batch_pos : batch_pos + BATCH_SIZE, 
                    :, :, :, :, 
                ]
                batch_pos = (
                    batch_pos + BATCH_SIZE
                ) % TRAIN_SET_SIZE

                (
                    total_loss, recon_loss, kld_loss, rnn_loss, 
                ) = oneBatch(
                    vae, rnn, batch, beta, 
                )
                
                optim.zero_grad()
                total_loss.backward()
                optim.step()

                epoch_recon__loss += recon_loss / n_batches
                epoch_kld____loss +=   kld_loss / n_batches
                epoch_z_pred_loss +=   rnn_loss / n_batches

            vae.eval()
            rnn.eval()
            with torch.no_grad():
                (
                    _, validate_recon__loss, 
                    validate_kld____loss, validate_z_pred_loss, 
                ) = oneBatch(
                    vae, rnn, validate_set, beta, 
                )
            lossLogger.eat(
                epoch, 
                train____recon__loss=epoch_recon__loss, 
                validate_recon__loss=validate_recon__loss, 
                train____kld____loss=epoch_kld____loss, 
                validate_kld____loss=validate_kld____loss, 
                train____z_pred_loss=epoch_z_pred_loss, 
                validate_z_pred_loss=validate_z_pred_loss, 
            )
            # if epoch % 1 == 0:
            #     torch.save(vae.state_dict(), path.join(
            #         this_checkpoints_path, f'{epoch}.pt', 
            #     ))
    except KeyboardInterrupt:
        print('Received ^C.')
    # reconstructValidateSet(
    #     vae.cpu(), validate_images, 
    # )

def oneBatch(vae: VAE, rnn: RNN, batch: torch.Tensor, beta):
    flat_batch = batch.view(
        BATCH_SIZE * SEQ_LEN, 1, RESOLUTION, RESOLUTION, 
    )
    reconstructions, mu, log_var = vae.forward(flat_batch)
    vae_loss, recon_loss, kld_loss = vae.computeLoss(
        flat_batch, reconstructions, mu, log_var, beta, 
    )

    z: torch.Tensor = mu.view(BATCH_SIZE, SEQ_LEN, LATENT_DIM)
    z_hat = torch.zeros((
        BATCH_SIZE, SEQ_LEN - RNN_MIN_CONTEXT, LATENT_DIM, 
    ))
    rnn.zeroHidden(BATCH_SIZE)
    for t in range(RNN_MIN_CONTEXT):
        rnn.stepTime(z, t)
    for t in range(RNN_MIN_CONTEXT, SEQ_LEN):
        z_hat[:, t - RNN_MIN_CONTEXT, :] = rnn.projHead(
            rnn.hidden, 
        )
        # that probably gonna break autograd
        # wait what it didn't
        rnn.stepTime(z, t)
    rnn_loss = F.mse_loss(z_hat, z[:, RNN_MIN_CONTEXT:, :])
    
    total_loss = (
        vae_loss * VAE_LOSS_COEF + 
        rnn_loss * RNN_LOSS_COEF
    )
    return total_loss, recon_loss, kld_loss, rnn_loss

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

if __name__ == '__main__':
    train()
