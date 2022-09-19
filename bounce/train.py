'''
evalZ: plot rnn prediction under symmetry.
beta warmup? 
set beta=0 may deknot the z space? increase beta later? beta oscillation? 
'''
import random
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
from symmetryTransforms import (
    sampleTranslate, sampleRotate, sampleTR, identity, 
)

BATCH_SIZE = 64

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    print('We have CUDA.')
    gpu_name = torch.cuda.get_device_name(
        torch.cuda.current_device(), 
    )
    with open('gpu_name.txt', 'w') as f:
        print(gpu_name, file=f)
else:
    DEVICE = CPU
    print("We DON'T have CUDA.", flush=True)

def oneEpoch(
    profiler: StreamProfiler, epoch: int, 
    vae: VAE, rnn: RNN, optim: torch.optim.Optimizer, 
    train_videos, validate_videos, 
    train_traj, validate_traj, 
    lossLogger: LossLogger, 
    config: Config, 
    beta, vae_loss_coef, img_pred_loss_coef, do_symmetry, 
    variational_rnn, rnn_width, deep_spread, vae_channels, 
    vvrnn, vvrnn_static, rnn_min_context, z_pred_loss_coef, 
    T, R, TR, I, lr, do_residual, grad_clip, BCE_not_MSE, 
    teacher_forcing_duration, supervised_rnn, 
):
    profiler.gonna('pre')
    imgCriterion = config.imgCriterion
    beta = beta(epoch)
    n_batches = TRAIN_SET_SIZE // BATCH_SIZE
    vae.train()
    rnn.train()
    epoch_recon__loss = 0
    epoch_kld____loss = 0
    epoch_img_pred_loss = 0
    epoch_z_pred_loss = 0
    epoch_rnn_stdnorm = 0
    for batch_pos in range(0, TRAIN_SET_SIZE, BATCH_SIZE):
        profiler.gonna('b_pre')
        video_batch = train_videos[
            batch_pos : batch_pos + BATCH_SIZE, 
            :, :, :, :, 
        ]
        traj_batch = train_traj[
            batch_pos : batch_pos + BATCH_SIZE, 
            :, :, 
        ]
        batch_pos = (
            batch_pos + BATCH_SIZE
        ) % TRAIN_SET_SIZE
        try:
            teacher_forcing_rate = 1 - min(
                1, epoch / teacher_forcing_duration, 
            )
        except ZeroDivisionError:
            teacher_forcing_rate = 0

        profiler.gonna('1b')
        print('batch position', batch_pos, end='  \r', flush=True)
        (
            total_loss, recon_loss, kld_loss, 
            img_pred_loss, z_pred_loss, 
            rnn_stdnorm, 
        ) = oneBatch(
            vae, rnn, video_batch, traj_batch, beta, 
            vae_loss_coef, img_pred_loss_coef, do_symmetry, 
            variational_rnn, vvrnn, vvrnn_static, 
            rnn_min_context, z_pred_loss_coef, 
            T, R, TR, I, imgCriterion, teacher_forcing_rate, 
            supervised_rnn, 
        )
        
        profiler.gonna('bp')
        optim.zero_grad()
        total_loss.backward()
        grad_norm = getGradNorm(optim)
        torch.nn.utils.clip_grad_norm_(
            getParams(optim), grad_clip, 
        )
        optim.step()

        epoch_recon__loss += recon_loss / n_batches
        epoch_kld____loss +=   kld_loss / n_batches
        epoch_img_pred_loss += img_pred_loss / n_batches
        epoch_z_pred_loss += z_pred_loss / n_batches
        epoch_rnn_stdnorm += rnn_stdnorm / n_batches

    profiler.gonna('ev')
    vae.eval()
    rnn.eval()
    with torch.no_grad():
        (
            _, validate_recon__loss, 
            validate_kld____loss, 
            validate_img_pred_loss, validate_z_pred_loss, 
            validate_rnn_std_norm, 
        ) = oneBatch(
            vae, rnn, validate_videos, validate_traj, beta, 
            vae_loss_coef, img_pred_loss_coef, do_symmetry, 
            variational_rnn, vvrnn, vvrnn_static, 
            rnn_min_context, z_pred_loss_coef, 
            T, R, TR, I, imgCriterion, teacher_forcing_rate=0, 
            supervised_rnn=supervised_rnn, 
        )
    lossLogger.eat(
        epoch, True, 
        train____recon__loss=epoch_recon__loss, 
        validate_recon__loss=validate_recon__loss, 
        train____kld____loss=epoch_kld____loss, 
        validate_kld____loss=validate_kld____loss, 
        train____img_pred_loss=epoch_img_pred_loss, 
        validate_img_pred_loss=validate_img_pred_loss, 
        train____z_pred_loss=epoch_z_pred_loss, 
        validate_z_pred_loss=validate_z_pred_loss, 
        # train____rnn_std_norm=epoch_rnn_stdnorm, 
        # validate_rnn_std_norm=validate_rnn_std_norm, 
        grad_norm=grad_norm, 
    )
    profiler.display()

def oneBatch(
    vae: VAE, rnn: RNN, video_batch: torch.Tensor, 
    traj_batch: torch.Tensor, beta, 
    vae_loss_coef, img_pred_loss_coef, do_symmetry, 
    variational_rnn, vvrnn, vvrnn_static, rnn_min_context, 
    z_pred_loss_coef, T, R, TR, I, imgCriterion, 
    teacher_forcing_rate, supervised_rnn, 
    visualize=False, batch_size = BATCH_SIZE, 
):
    flat_video_batch = video_batch.view(
        batch_size * SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    reconstructions, mu, log_var, z_flat = vae.forward(
        flat_video_batch, 
    )
    vae_loss, recon_loss, kld_loss = vae.computeLoss(
        flat_video_batch, reconstructions, mu, log_var, beta, 
    )

    if not variational_rnn:
        z_flat = mu
    z: torch.Tensor = z_flat.view(
        batch_size, SEQ_LEN, LATENT_DIM, 
    )

    t = random.choice(
        ['I' ] * I + 
        ['T' ] * T + 
        ['R' ] * R + 
        ['TR'] * TR
    )
    if t == 'I':
        trans, untrans = identity, identity
    elif t == 'T':
        trans, untrans = sampleTranslate(DEVICE)
    elif t == 'R':
        trans, untrans = sampleRotate(DEVICE)
    elif t == 'TR':
        trans, untrans = sampleTR(DEVICE)
    img_pred_loss, z_pred_loss, predictions = oneTrans(
        vae, rnn, video_batch, traj_batch, 
        vvrnn, vvrnn_static, rnn_min_context, imgCriterion, 
        teacher_forcing_rate, supervised_rnn, 
        batch_size, 
        z, trans, untrans, 
    )
    
    # `predictions` is that of the last trans. 

    total_loss = (
        vae_loss * vae_loss_coef + 
        img_pred_loss * img_pred_loss_coef + 
        z_pred_loss   * z_pred_loss_coef
    )

    if visualize:
        return predictions, reconstructions.view(
            batch_size, SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
        )
    else:
        return (
            total_loss, recon_loss, kld_loss, 
            img_pred_loss, z_pred_loss, 
            torch.exp(0.5 * log_var).norm(2) / batch_size, 
        )

def getParams(optim: torch.optim.Optimizer):
    s = []
    for param in optim.param_groups[0]['params']:
        if param.grad is not None:
            s.append(param)
    return s

def getGradNorm(optim: torch.optim.Optimizer):
    s = 0
    for param in getParams(optim):
        s += param.grad.norm(2).item() ** 2
    return s ** .5

def oneTrans(
    vae: VAE, rnn: RNN, 
    video_batch: torch.Tensor, traj_batch: torch.Tensor, 
    vvrnn, vvrnn_static, rnn_min_context, imgCriterion, 
    teacher_forcing_rate, supervised_rnn, 
    batch_size, 
    z, trans, untrans, 
):
    if supervised_rnn:
        z = traj_batch
    z_hat_transed = torch.zeros((
        batch_size, SEQ_LEN - rnn_min_context, LATENT_DIM, 
    )).to(DEVICE)
    log_var = torch.ones((
        batch_size, SEQ_LEN - rnn_min_context, LATENT_DIM, 
    )).to(DEVICE) * vvrnn_static
    rnn.zeroHidden(batch_size, DEVICE)
    for t in range(rnn_min_context):
        rnn.stepTime(z, t, trans)
    for t in range(SEQ_LEN - rnn_min_context):
        z_hat_transed[:, t, :] = rnn.  projHead(rnn.hidden)
        if vvrnn:
            log_var  [:, t, :] = rnn.logVarHead(rnn.hidden)
        if random.random() < teacher_forcing_rate:
            rnn.stepTime(z, t, trans)
        else:
            rnn.stepTime(z_hat_transed, t, identity)
    flat_z_hat = untrans(z_hat_transed.view(
        -1, LATENT_DIM, 
    ).T).T
    flat_log_var = log_var.view(-1, LATENT_DIM)
    r_flat_z_hat = reparameterize(flat_z_hat, flat_log_var)
    predictions = vae.decode(r_flat_z_hat).view(
        batch_size, SEQ_LEN - rnn_min_context, 
        IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    img_pred_loss = imgCriterion(predictions, video_batch[
        :, rnn_min_context:, :, :, :, 
    ])

    z_hat = flat_z_hat.view(
        batch_size, SEQ_LEN - rnn_min_context, LATENT_DIM, 
    )
    z_pred_loss = F.mse_loss(
        z_hat, z[:, rnn_min_context:, :], 
    )

    return img_pred_loss, z_pred_loss, predictions
