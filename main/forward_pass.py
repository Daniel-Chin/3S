import random
from typing import Callable, List, Optional
from functools import lru_cache

import torch
import torch.nn.functional as F
import torchWork
from torchWork import DEVICE, HAS_CUDA, CPU

from shared import *
from losses import Loss_root
from vae import VAE
from rnn import PredRNN, EnergyRNN
from linearity_metric import projectionMSE
from symmetry_transforms import identity

def forward(
    epoch, batch_i, experiment, hParams: HyperParams, 
    video_batch: torch.Tensor, traj_batch: torch.Tensor, 
    vae: VAE, 
    predRnns: List[PredRNN], energyRnns: List[EnergyRNN], 
    profiler: torchWork.Profiler, 
    require_img_predictions_and_z_hat: bool = True, 
    validating: bool = False, 
):
    SEQ_LEN        = experiment.DATASET_INSTANCE.SEQ_LEN
    IMG_N_CHANNELS = experiment.DATASET_INSTANCE.IMG_N_CHANNELS
    batch_size = video_batch.shape[0]
    imgCriterion = hParams.sched_image_loss.get(epoch)
    lossTree = Loss_root()

    # remove time axis
    flat_video_batch = video_batch.view(
        batch_size * SEQ_LEN, IMG_N_CHANNELS, 
        *hParams.signal_resolution, 
    )
    flat_traj_batch = traj_batch.view(
        batch_size * SEQ_LEN, -1, 
    )

    # vae forward pass
    with profiler('good'):
        mu, log_var = vae.encode(flat_video_batch)
        if hParams.vae_is_actually_ae:
            flat_z = mu
        else:
            flat_z = reparameterize(mu, log_var)
        if validating or hParams.lossWeightTree['self_recon'].weight:
            flat_reconstructions = vae.decode(flat_z)
            lossTree.self_recon = imgCriterion(
                flat_reconstructions, flat_video_batch, 
            ).cpu()
            reconstructions = flat_reconstructions.detach().view(
                batch_size, SEQ_LEN, 
                IMG_N_CHANNELS, *hParams.signal_resolution, 
            )
        else:
            reconstructions = None
        if hParams.lossWeightTree['kld'].weight:
            lossTree.kld = torch.mean(-0.5 * torch.sum(
                1 + log_var - mu ** 2 - log_var.exp(), dim=1, 
            ), dim=0).cpu()
    
    if hParams.supervise_vae:
        if hParams.supervise_vae_only_xy:
            my_slice = slice(0, 2)
            to_decode = torch.cat((
                flat_traj_batch[:, :2], 
                flat_z[:, 2:], 
            ), dim=1)
        else:
            my_slice = slice(0, 3)
            to_decode = flat_traj_batch
        if hParams.vae_is_actually_ae:
            pass
        else:
            to_decode = reparameterize(to_decode, -3)
        my_z_hat = flat_z.clone()[:, my_slice]
        my_z = flat_traj_batch.clone()[:, my_slice]
        lossTree.supervise.vae.encode = F.mse_loss(
            my_z_hat, my_z, 
        ).cpu()
        with profiler('good'):
            synthesis = vae.decode(to_decode)
            lossTree.supervise.vae.decode = imgCriterion(
                synthesis, flat_video_batch, 
            ).cpu()
        del my_slice, to_decode, my_z_hat, my_z

    if hParams.variational_rnn:
        flat_z_for_rnn = flat_z
    else:
        flat_z_for_rnn = mu
    if hParams.supervise_rnn:
        flat_z_for_rnn = flat_traj_batch
    
    # restore time axis
    z = flat_z_for_rnn.view(
        batch_size, SEQ_LEN, hParams.symm.latent_dim, 
    )

    # rnn forward pass
    min_context = hParams.rnn_min_context
    if validating or hParams.K == 1:
        small_batch_size = batch_size
        sampled_z = z
        sampled_flat_z = flat_z_for_rnn
        sample_video_batch = video_batch
    else:
        small_batch_size = batch_size // hParams.K
        perm = torch.randperm(batch_size)
        idx = perm[: small_batch_size]
        sampled_z = z[idx, :, :]
        sampled_flat_z = sampled_z.view(
            small_batch_size * SEQ_LEN, hParams.symm.latent_dim, 
        )
        sample_video_batch = video_batch[idx, :, :, :, :]

    lossTree.predict.image = []
    lossTree.predict.z     = []
    lossTree.supervise.rnn = []
    lossTree.vicreg.variance = []
    lossTree.vicreg.invariance = []
    lossTree.vicreg.covariance = []
    z_std_l = []
    z_std_r = []
    for predRnn in predRnns:
        for K_i in range(hParams.K):
        # The current implementation of K > 1 is inefficient. 
        # That is because some batch size is wasted. 
        # If we turn out to use K > 1, optimize. 
            trans, untrans = hParams.symm.sample()
            flat_z_transed = trans(sampled_flat_z)

            # restore time axis
            z_transed = flat_z_transed.view(
                small_batch_size, SEQ_LEN, hParams.symm.latent_dim, 
            )
            
            flat_z_hat_aug, r_flat_z_hat_aug, log_var = rnnForward(
                predRnn, z_transed, untrans, 
                small_batch_size, experiment, hParams, epoch, batch_i, profiler, 
            )

            # predict image
            if (
                require_img_predictions_and_z_hat
                or validating 
                or hParams.lossWeightTree['predict']['image'].weight != 0
            ):
                z_hat_aug = flat_z_hat_aug.view(
                    small_batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
                )
                with profiler('good'):
                    flat_predictions = vae.decode(r_flat_z_hat_aug)
                img_predictions = flat_predictions.view(
                    small_batch_size, SEQ_LEN - min_context, 
                    IMG_N_CHANNELS, *hParams.signal_resolution, 
                )
                with profiler('good'):
                    lossTree.predict.image.append(imgCriterion(
                        img_predictions, 
                        sample_video_batch[:, min_context:, :, :, :], 
                    ))
            else:
                img_predictions = None
                z_hat_aug = None

            # predict z
            if (
                validating 
                or hParams.lossWeightTree['predict']['z'].weight != 0
                or hParams.supervise_rnn 
                or hParams.lossWeightTree['vicreg'].weight
            ):
                if hParams.jepa_stop_grad_l_encoder:
                    flat_z_hat_aug, r_flat_z_hat_aug, log_var = rnnForward(
                        predRnn, z_transed.detach(), untrans, 
                        small_batch_size, experiment, hParams, epoch, batch_i, profiler, 
                    )
                z_hat_aug = flat_z_hat_aug.view(
                    small_batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
                )
                _z = sampled_z[:, min_context:, :]
                if hParams.jepa_stop_grad_r_encoder:
                    _z = _z.detach()
                if validating or hParams.lossWeightTree['vicreg'].weight:
                    flat_batch_size = small_batch_size * (SEQ_LEN - min_context)
                    if validating:
                        expander = identity
                    else:
                        expander = vae.expander
                    flat_emb_r = expander(_z.reshape(
                        flat_batch_size, 
                        hParams.symm.latent_dim, 
                    ))
                    flat_emb_l = expander(z_hat_aug.reshape(
                        flat_batch_size, 
                        hParams.symm.latent_dim, 
                    ))

                    with profiler('good'):
                        # invariance
                        if hParams.vicreg_invariance_on_Y:
                            l, r = z_hat_aug, _z
                        else:
                            l, r = flat_emb_l, flat_emb_r
                        lossTree.vicreg.invariance.append(
                            F.mse_loss(l, r), 
                        )
                        del l, r
                    
                        # variance
                        if not hParams.vicreg_cross_traj:
                            std_emb_l = torch.sqrt(flat_emb_l.var(dim=0) + 1e-4)
                            std_emb_r = torch.sqrt(flat_emb_r.var(dim=0) + 1e-4)
                        else:   # should be equivalent
                            emb_l = flat_emb_l.view(
                                small_batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
                            )
                            emb_r = flat_emb_r.view(
                                small_batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
                            )
                            std_emb_l = torch.sqrt(emb_l.var(dim=0).mean(dim=0) + 1e-4)
                            std_emb_r = torch.sqrt(emb_r.var(dim=0).mean(dim=0) + 1e-4)
                        lossTree.vicreg.variance.append(
                            F.relu(1 - std_emb_l).mean() +
                            F.relu(1 - std_emb_r).mean()
                        )
                        z_std_l.append(std_emb_l.mean())
                        z_std_r.append(std_emb_r.mean())

                        # covariance
                        if not hParams.vicreg_cross_traj:
                            flat_emb_l = flat_emb_l - flat_emb_l.mean(dim=0)
                            flat_emb_r = flat_emb_r - flat_emb_r.mean(dim=0)
                        else:
                            emb_l = flat_emb_l.view(
                                small_batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
                            )
                            emb_r = flat_emb_r.view(
                                small_batch_size, SEQ_LEN - min_context, hParams.symm.latent_dim, 
                            )
                            emb_l = emb_l - emb_l.mean(dim=0)
                            emb_r = emb_r - emb_r.mean(dim=0)
                            flat_emb_l = emb_l.view(
                                flat_batch_size, hParams.symm.latent_dim, 
                            )
                            flat_emb_r = emb_r.view(
                                flat_batch_size, hParams.symm.latent_dim, 
                            )
                        cov_emb_l = (flat_emb_l.T @ flat_emb_l) / (flat_batch_size - 1)
                        cov_emb_r = (flat_emb_r.T @ flat_emb_r) / (flat_batch_size - 1)

                        lossTree.vicreg.covariance.append(
                            (offDiagonalMask2d(hParams.vicreg_emb_dim) * cov_emb_l).pow_(2).sum() / hParams.vicreg_emb_dim + 
                            (offDiagonalMask2d(hParams.vicreg_emb_dim) * cov_emb_r).pow_(2).sum() / hParams.vicreg_emb_dim
                        )
                else:
                    with profiler('good'):
                        z_loss = F.mse_loss(z_hat_aug, _z)
                    if hParams.supervise_rnn:
                        lossTree.supervise.rnn.append(z_loss)
                    else:
                        lossTree.predict.z.append(z_loss)
    def _aggregate(x):
        if x:
            return torch.stack(x).mean().cpu()
        else:
            return torch.tensor(0, device=CPU)
    lossTree.predict.image     = _aggregate(lossTree.predict.image    )
    lossTree.predict.z         = _aggregate(lossTree.predict.z        )
    lossTree.supervise.rnn     = _aggregate(lossTree.supervise.rnn    )
    lossTree.vicreg.variance   = _aggregate(lossTree.vicreg.variance  )
    lossTree.vicreg.invariance = _aggregate(lossTree.vicreg.invariance)
    lossTree.vicreg.covariance = _aggregate(lossTree.vicreg.covariance)
    z_std_l = _aggregate(z_std_l)
    z_std_r = _aggregate(z_std_r)
    del _aggregate

    if hParams.vvrnn or hParams.vvrnn_static is not None:
        mean_square_vrnn_std = torch.exp(
            0.5 * log_var
        ).norm(2).cpu() ** 2 / small_batch_size
        # isn't this wrong? |log_var| is prolly flat_small_batch_size. 
    else:
        mean_square_vrnn_std = torch.tensor(0, device=CPU)

    # symm_self_consistency
    if hParams.lossWeightTree['symm_self_consistency'].weight != 0:
        assert not (
            hParams.jepa_stop_grad_l_encoder or 
            hParams.jepa_stop_grad_r_encoder
            # As long as we are replicating Will's results, 
            # stopping grad would make VAE truly untouched. 
            # 
            # Just generally, it is not clear how 
            # `symm_self_consistency` should interact with
            # `jepa_stop_grad...`. So, unimplemented for now. 
        )
        flat_z_hat, r_flat_z_hat, log_var = rnnForward(
            predRnn, z, identity, 
            batch_size, experiment, hParams, epoch, batch_i, profiler, 
        )
        lossTree.symm_self_consistency = F.mse_loss(
            flat_z_hat_aug, flat_z_hat, 
        ).cpu()

    # seq energy
    if hParams.lossWeightTree['seq_energy'].weight != 0:
        energyRnn = energyRnns[0]
        RATIO = 64
        noise = torch.randn((
            RATIO * batch_size, SEQ_LEN, hParams.symm.latent_dim, 
        ), device=DEVICE) * hParams.energy_noise_std
        with profiler('noising'):
            for i in range(RATIO):
                # Maybe in-place is faster, maybe batch op is faster...
                noise[
                    i * batch_size : (i+1) * batch_size, :, :, 
                ] += z_transed.detach()
        energies: List[torch.Tensor] = []
        for seq in z_transed, noise:
            energy = torch.zeros((
                seq.shape[0], SEQ_LEN - min_context, 
            ), device=DEVICE)
            energyRnn.zeroHidden(seq.shape[0], DEVICE)
            for t in range(SEQ_LEN):
                energyRnn.stepTime(seq[:, t, :])
                if t >= min_context:
                    energy[:, t - min_context] = energyRnn.projHead(
                        energyRnn.hidden, 
                    )[:, 0]
            energies.append(energy)
        data_energy, noise_energy = energies
        # hinge loss
        lossTree.seq_energy.real = (
            1 +  data_energy.clamp(min=-1).mean().cpu()
        )
        lossTree.seq_energy.fake = (
            1 - noise_energy.clamp(max=+1).mean().cpu()
        )
    
    # cycle consistency
    if hParams.lossWeightTree['cycle'].weight != 0:
        cycled_z, _ = vae.encode(vae.decode(flat_z))
        lossTree.cycle = F.mse_loss(cycled_z, flat_z).cpu()

    with profiler('eval_linearity'):
        linear_proj_mse = projectionMSE(
            mu.detach(), flat_traj_batch.detach(), 
        )
    
    return (
        lossTree, reconstructions, 
        tryDetach(img_predictions), 
        tryDetach(z), tryDetach(z_hat_aug), 
        [
            ('mean_square_vrnn_std', mean_square_vrnn_std.detach()), 
            ('linear_proj_mse', linear_proj_mse.detach()), 
            ('z_std_l', z_std_l.detach()), 
            ('z_std_r', z_std_r.detach()), 
        ], 
    )

def tryDetach(x: Optional[torch.Tensor], /):
    if x is None:
        return None
    return x.detach()

def rnnForward(
    rnn: PredRNN, 
    z_transed, untrans: Callable[[torch.Tensor], torch.Tensor], 
    batch_size, experiment, hParams: HyperParams, 
    epoch, batch_i, profiler, 
):
    SEQ_LEN = experiment.DATASET_INSTANCE.SEQ_LEN
    min_context = hParams.rnn_min_context
    if hParams.sched_sampling is None:
        teacher_rate = 0
    else:
        teacher_rate = hParams.sched_sampling.get(epoch, hParams)
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
    flat_z_hat_tut = untrans(flat_z_hat_transed)
    if hParams.vvrnn or hParams.vvrnn_static is not None:
        flat_log_var = log_var.view(-1, hParams.symm.latent_dim)
        reparamed = untrans(reparameterize(
            flat_z_hat_transed, flat_log_var, 
        ))
    else:
        reparamed = flat_z_hat_tut
    return flat_z_hat_tut, reparamed, log_var

@lru_cache(2)
def offDiagonalMask2d(size: int):
    x = torch.ones((size, size), device=DEVICE)
    x.fill_diagonal_(0)
    return x
