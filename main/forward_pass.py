import random
from typing import Callable

import torch
import torch.nn.functional as F
import torchWork
from torchWork import DEVICE

from shared import *
from losses import Loss_root
from vae import VAE
from rnn import RNN
from linearity_metric import projectionMSE
from symmetry_transforms import identity

def forward(
    epoch, experiment, hParams: HyperParams, 
    video_batch: torch.Tensor, traj_batch: torch.Tensor, 
    vae: VAE, rnn: RNN, 
    profiler: torchWork.Profiler, 
    require_img_predictions: bool = True, 
):
    batch_size = video_batch.shape[0]
    lossTree = Loss_root()

    # remove time axis
    flat_video_batch = video_batch.view(
        batch_size * SEQ_LEN, IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
    )
    flat_traj_batch = traj_batch.view(
        batch_size * SEQ_LEN, -1, 
    )

    # vae forward pass
    with profiler('good'):
        mu, log_var = vae.encode(flat_video_batch)
        flat_z = reparameterize(mu, log_var)
        reconstructions = vae.decode(flat_z)
        lossTree.self_recon = hParams.imgCriterion(
            reconstructions, flat_video_batch, 
        ).cpu()
        lossTree.kld = torch.mean(-0.5 * torch.sum(
            1 + log_var - mu ** 2 - log_var.exp(), dim=1, 
        ), dim=0).cpu()
    
    if hParams.supervise_vae:
        lossTree.supervise.vae.encode = F.mse_loss(
            flat_z, flat_traj_batch, 
        ).cpu()
        with profiler('good'):
            synthesis = vae.decode(flat_traj_batch)
            lossTree.supervise.vae.decode = hParams.imgCriterion(
                synthesis, flat_video_batch, 
            ).cpu()

    if not hParams.variational_rnn:
        flat_z = mu
    if hParams.supervise_rnn:
        flat_z = flat_traj_batch
    
    trans, untrans = hParams.symm.sample()    
    flat_z_transed = trans(flat_z)

    # restore time axis
    z = flat_z.view(
        batch_size, SEQ_LEN, hParams.symm.latent_dim, 
    )
    z_transed = flat_z_transed.view(
        batch_size, SEQ_LEN, hParams.symm.latent_dim, 
    )
    
    # rnn forward pass
    min_context = hParams.rnn_min_context
    if (
        require_img_predictions
        or hParams.lossWeightTree['predict']['image'] != 0
    ):
        flat_z_hat_aug, r_flat_z_hat_aug, log_var = rnnForward(
            rnn, z_transed, untrans, 
            batch_size, hParams, epoch, profiler, 
        )
        with profiler('good'):
            flat_predictions = vae.decode(r_flat_z_hat_aug)
        img_predictions = flat_predictions.view(
            batch_size, SEQ_LEN - min_context, 
            IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
        )
        with profiler('good'):
            lossTree.predict.image = hParams.imgCriterion(
                img_predictions, 
                video_batch[:, min_context:, :, :, :], 
            ).cpu()
    else:
        img_predictions = torch.zeros((
            batch_size, RESOLUTION, RESOLUTION, 
            IMG_N_CHANNELS, 
        ))

    if (
        hParams.lossWeightTree['predict']['z'] != 0
        or hParams.supervise_rnn
    ):
        if hParams.jepa_stop_grad_encoder:
            flat_z_hat_aug, r_flat_z_hat_aug, log_var = rnnForward(
                rnn, z_transed.detach(), untrans, 
                batch_size, hParams, epoch, profiler, 
            )
        z_hat = flat_z_hat_aug.view(
            batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
        )
        _z = z[:, min_context:, :]
        if hParams.jepa_stop_grad_encoder:
            _z = _z.detach()
        z_loss = F.mse_loss(z_hat, _z)
        if hParams.supervise_rnn:
            lossTree.supervise.rnn = z_loss.cpu()
        else:
            lossTree.predict.z = z_loss.cpu()

    mean_square_vrnn_std = torch.exp(
        0.5 * log_var
    ).norm(2).cpu() ** 2 / batch_size

    if (
        hParams.lossWeightTree['symm_self_consistency'] != 0
    ):
        assert not hParams.jepa_stop_grad_encoder
        # As long as we are replicating Will's results, 
        # stopping grad would make VAE truly untouched. 
        flat_z_hat, r_flat_z_hat, log_var = rnnForward(
            rnn, z, identity, 
            batch_size, hParams, epoch, profiler, 
        )
        lossTree.symm_self_consistency = F.mse_loss(
            flat_z_hat_aug, flat_z_hat, 
        ).cpu()

    with profiler('eval_linearity'):
        linear_proj_mse = projectionMSE(
            mu, flat_traj_batch, 
        )
    
    return (
        lossTree, reconstructions.view(
            batch_size, SEQ_LEN, 
            IMG_N_CHANNELS, RESOLUTION, RESOLUTION, 
        ), img_predictions, z, z_hat, 
        [
            ('mean_square_vrnn_std', mean_square_vrnn_std), 
            ('linear_proj_mse', linear_proj_mse)
        ], 
    )

def rnnForward(
    rnn: RNN, 
    z_transed, untrans: Callable[[torch.Tensor], torch.Tensor], 
    batch_size, hParams: HyperParams, epoch, 
    profiler, 
):
    min_context = hParams.rnn_min_context
    teacher_rate = hParams.getTeacherForcingRate(epoch)
    z_hat_transed = torch.zeros((
        batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
    ), device=DEVICE)
    log_var = torch.ones((
        batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
    ), device=DEVICE)
    if hParams.vvrnn_static is not None:
        log_var *= hParams.vvrnn_static
    rnn.zeroHidden(batch_size, DEVICE)
    with profiler('good'):
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
    flat_z_hat_transed = z_hat_transed.view(
        -1, hParams.symm.latent_dim, 
    )
    flat_log_var = log_var.view(-1, hParams.symm.latent_dim)
    return untrans(flat_z_hat_transed), untrans(reparameterize(
        flat_z_hat_transed, flat_log_var, 
    )), log_var
