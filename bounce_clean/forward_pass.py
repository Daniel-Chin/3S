import random

import torch
import torch.nn.functional as F
import torchWork
from torchWork import DEVICE

from shared import *
from losses import Loss_root
from vae import VAE
from rnn import RNN
from symmetry_transforms import (
    sampleTranslate, sampleRotate, sampleTR, identity, 
)

def forward(
    epoch, hParams: HyperParams, 
    video_batch: torch.Tensor, traj_batch: torch.Tensor, 
    vae: VAE, rnn: RNN, 
    profiler: torchWork.Profiler, 
    require_img_predictions: bool = True, 
):
    batch_size = hParams.batch_size
    lossTree = Loss_root()

    # remove time axis
    flat_video_batch = video_batch.view(
        batch_size * SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    flat_traj_batch = traj_batch.view(
        batch_size * SEQ_LEN, SPACE_DIM, 
    )

    # vae forward pass
    with profiler.goodTime():
        mu, log_var = vae.encode(flat_video_batch)
        flat_z = reparameterize(mu, log_var)
        reconstructions = vae.decode(flat_z)
        lossTree.self_recon = hParams.imgCriterion(
            reconstructions, flat_video_batch, 
        )
        lossTree.kld = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1, 
        ), dim=0)
    
    if hParams.supervise_vae:
        lossTree.supervise.vae.encode = F.mse_loss(
            mu, flat_traj_batch, 
        )
        with profiler.goodTime():
            synthesis = vae.decode(flat_traj_batch)
            lossTree.supervise.vae.decode = hParams.imgCriterion(
                synthesis, flat_video_batch, 
            )

    if not hParams.variational_rnn:
        flat_z = mu
    if hParams.supervise_rnn:
        flat_z = flat_traj_batch
    
    t = random.choice(
        ['I' ] * hParams.I + 
        ['T' ] * hParams.T + 
        ['R' ] * hParams.R + 
        ['TR'] * hParams.TR
    )
    if t == 'I':
        trans, untrans = identity, identity
    elif t == 'T':
        trans, untrans = sampleTranslate(DEVICE)
    elif t == 'R':
        trans, untrans = sampleRotate(DEVICE)
    elif t == 'TR':
        trans, untrans = sampleTR(DEVICE)
        
    flat_z_transed = trans(flat_z)

    # restore time axis
    z = flat_z.view(
        batch_size, SEQ_LEN, LATENT_DIM, 
    )
    z_transed = flat_z_transed.view(
        batch_size, SEQ_LEN, LATENT_DIM, 
    )
    
    # rnn forward pass
    min_context = hParams.rnn_min_context
    teacher_rate = hParams.getTeacherForcingRate(epoch)
    z_hat_transed = torch.zeros((
        batch_size, SEQ_LEN - min_context, LATENT_DIM, 
    ), device=DEVICE)
    log_var = torch.ones((
        batch_size, SEQ_LEN - min_context, LATENT_DIM, 
    ), device=DEVICE)
    if hParams.vvrnn_static is not None:
        log_var *= hParams.vvrnn_static
    rnn.zeroHidden(batch_size, DEVICE)
    with profiler.goodTime():
        for global_t in range(min_context):
            rnn.stepTime(z_transed[:, global_t, :])
        for global_t in range(min_context, SEQ_LEN):
            t = global_t - min_context
            z_hat_transed[:, t, :] = rnn.  projHead(rnn.hidden)
            if hParams.vvrnn:
                log_var  [:, t, :] = rnn.logVarHead(rnn.hidden)
            if random.random() < teacher_rate:
                rnn.stepTime(z_transed    [:, global_t, :])
            else:
                rnn.stepTime(z_hat_transed[:,        t, :])
    flat_z_hat = untrans(z_hat_transed.view(
        -1, LATENT_DIM, 
    ))
    flat_log_var = log_var.view(-1, LATENT_DIM)
    r_flat_z_hat = reparameterize(flat_z_hat, flat_log_var)
    if (
        not require_img_predictions and 
        hParams.lossWeightTree['predict']['image'] == 0
    ):
        img_predictions = torch.zeros((
            batch_size, RESOLUTION, RESOLUTION, 
            IMG_N_CHANNELS, 
        ))
    else:
        with profiler.goodTime():
            flat_predictions = vae.decode(r_flat_z_hat)
        img_predictions = flat_predictions.view(
            batch_size, SEQ_LEN - min_context, 
            IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
        )
        with profiler.goodTime():
            lossTree.predict.image = hParams.imgCriterion(
                img_predictions, 
                video_batch[:, min_context:, :, :, :], 
            )

    z_hat = flat_z_hat.view(
        batch_size, SEQ_LEN - min_context, LATENT_DIM, 
    )
    z_loss = F.mse_loss(z_hat, z[:, min_context:, :])
    if hParams.supervise_rnn:
        lossTree.supervise.rnn = z_loss
    else:
        lossTree.predict.z = z_loss

    mean_square_vrnn_std = torch.exp(
        0.5 * log_var
    ).norm(2) ** 2 / batch_size

    return (
        lossTree, reconstructions.view(
            batch_size, SEQ_LEN, 
            IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
        ), img_predictions, 
        [
            ('mean_square_vrnn_std', mean_square_vrnn_std), 
        ], 
    )
