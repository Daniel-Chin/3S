'''
evalZ: plot rnn prediction under symmetry.
beta warmup? 
set beta=0 may deknot the z space? increase beta later? beta oscillation? 
'''
import torch
import torch.nn.functional as F

try:
    from myTorch import LossLogger
    from streamProfiler import StreamProfiler
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'myTorch', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e
from shared import *
from vae import LATENT_DIM, VAE
from rnn import RNN
from symmetryTransforms import sampleTransforms, identity

BATCH_SIZE = 64
RNN_MIN_CONTEXT = 3

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    print('We have CUDA.')
else:
    DEVICE = CPU
    print("We DON'T have CUDA.")

def oneEpoch(
    profiler: StreamProfiler, epoch, 
    vae: VAE, rnn: RNN, optim: torch.optim.Optimizer, 
    train_set, validate_set, 
    lossLogger: LossLogger, 
    beta, vae_loss_coef, rnn_loss_coef, do_symmetry, 
    variational_rnn, rnn_width, deep_spread, vae_channels, 
    vvrnn, vvrnn_static, 
):
    profiler.gonna('pre')
    beta = beta(epoch)
    n_batches = TRAIN_SET_SIZE // BATCH_SIZE
    vae.train()
    rnn.train()
    epoch_recon__loss = 0
    epoch_kld____loss = 0
    epoch_z_pred_loss = 0
    epoch_rnn_stdnorm = 0
    for batch_pos in range(0, TRAIN_SET_SIZE, BATCH_SIZE):
        profiler.gonna('b_pre')
        batch = train_set[
            batch_pos : batch_pos + BATCH_SIZE, 
            :, :, :, :, 
        ]
        batch_pos = (
            batch_pos + BATCH_SIZE
        ) % TRAIN_SET_SIZE

        profiler.gonna('1b')
        (
            total_loss, recon_loss, kld_loss, rnn_loss, 
            rnn_stdnorm, 
        ) = oneBatch(
            vae, rnn, batch, beta, 
            vae_loss_coef, rnn_loss_coef, do_symmetry, 
            variational_rnn, vvrnn, vvrnn_static, 
        )
        
        profiler.gonna('bp')
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        epoch_recon__loss += recon_loss / n_batches
        epoch_kld____loss +=   kld_loss / n_batches
        epoch_z_pred_loss +=   rnn_loss / n_batches
        epoch_rnn_stdnorm += rnn_stdnorm / n_batches

    profiler.gonna('ev')
    vae.eval()
    rnn.eval()
    with torch.no_grad():
        (
            _, validate_recon__loss, 
            validate_kld____loss, validate_z_pred_loss, 
            validate_rnn_std_norm, 
        ) = oneBatch(
            vae, rnn, validate_set, beta, 
            vae_loss_coef, rnn_loss_coef, do_symmetry, 
            variational_rnn, vvrnn, vvrnn_static, 
        )
    lossLogger.eat(
        epoch, True, 
        train____recon__loss=epoch_recon__loss, 
        validate_recon__loss=validate_recon__loss, 
        train____kld____loss=epoch_kld____loss, 
        validate_kld____loss=validate_kld____loss, 
        train____z_pred_loss=epoch_z_pred_loss, 
        validate_z_pred_loss=validate_z_pred_loss, 
        train____rnn_std_norm=epoch_rnn_stdnorm, 
        validate_rnn_std_norm=validate_rnn_std_norm, 
        # vae___grad_norm=vae___grad_norm, 
        # total_grad_norm=total_grad_norm, 
    )
    profiler.display()

def oneBatch(
    vae: VAE, rnn: RNN, batch: torch.Tensor, beta, 
    vae_loss_coef, rnn_loss_coef, do_symmetry, 
    variational_rnn, vvrnn, vvrnn_static, 
    visualize=False, batch_size = BATCH_SIZE, 
):
    flat_batch = batch.view(
        batch_size * SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    reconstructions, mu, log_var, z_flat = vae.forward(
        flat_batch, 
    )
    vae_loss, recon_loss, kld_loss = vae.computeLoss(
        flat_batch, reconstructions, mu, log_var, beta, 
    )

    if not variational_rnn:
        z_flat = mu
    z: torch.Tensor = z_flat.view(
        batch_size, SEQ_LEN, LATENT_DIM, 
    )
    z_hat_transed = torch.zeros((
        batch_size, SEQ_LEN - RNN_MIN_CONTEXT, LATENT_DIM, 
    )).to(DEVICE)
    log_var = torch.ones((
        batch_size, SEQ_LEN - RNN_MIN_CONTEXT, LATENT_DIM, 
    )).to(DEVICE) * vvrnn_static
    rnn.zeroHidden(batch_size, DEVICE)
    if do_symmetry:
        trans, untrans = sampleTransforms(DEVICE)
    else:
        trans = untrans = identity
    for t in range(RNN_MIN_CONTEXT):
        rnn.stepTime(z, t, trans)
    for t in range(SEQ_LEN - RNN_MIN_CONTEXT):
        z_hat_transed[:, t, :] = rnn.  projHead(rnn.hidden)
        if vvrnn:
            log_var  [:, t, :] = rnn.logVarHead(rnn.hidden)
        rnn.stepTime(z_hat_transed, t, identity)
    flat_z_hat = untrans(z_hat_transed.view(
        -1, LATENT_DIM, 
    ).T).T
    flat_log_var = log_var.view(-1, LATENT_DIM)
    if vvrnn:
        flat_z_hat = reparameterize(flat_z_hat, flat_log_var)
    predictions = vae.decode(flat_z_hat).view(
        batch_size, SEQ_LEN - RNN_MIN_CONTEXT, 
        IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    rnn_loss = F.mse_loss(predictions, batch[
        :, RNN_MIN_CONTEXT:, :, :, :, 
    ])
    # rnn_loss = F.mse_loss(z_hat, z[:, RNN_MIN_CONTEXT:, :])
    
    total_loss = (
        vae_loss * vae_loss_coef + 
        rnn_loss * rnn_loss_coef
    )

    if visualize:
        return predictions, reconstructions.view(
            batch_size, SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
        )
    else:
        return (
            total_loss, recon_loss, kld_loss, rnn_loss, 
            torch.exp(0.5 * log_var).norm(2) / batch_size, 
        )

def getGradNorm(optim: torch.optim.Optimizer):
    s = 0
    for param in optim.param_groups[0]['params']:
        if param.grad is not None:
            s += param.grad.norm(2).item() ** 2
    return s ** .5
