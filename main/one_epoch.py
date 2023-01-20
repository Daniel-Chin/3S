from os import path
from typing import Dict, List, Optional
import inspect

import torch
import torch.utils.data
from torchWork import Profiler, LossLogger, saveModels, HAS_CUDA
from torchWork.utils import getGradNorm, getParams

from shared import *
from forward_pass import forward
from vae import VAE
from rnn import PredRNN, EnergyRNN
from load_dataset import Dataset, dataLoader
from video_eval import videoEval

def oneEpoch(
    group_name: str, epoch: int, 
    experiment, hParams: HyperParams, 
    models: Dict[str, List[torch.nn.Module]], 
    optim: torch.optim.Optimizer, 
    trainSet: Dataset, validateSet: Dataset, 
    lossLogger: LossLogger, profiler: Profiler, 
    save_path: str, trainer_id: int, 
):
    profiler.active = False
    if epoch > hParams.max_epoch:
        return False
    with profiler(f'line {inspect.getframeinfo(inspect.currentframe()).lineno}'):
        vae: VAE = models['vae'][0]
        predRnns: List[PredRNN] = models['predRnn']
        energyRnns: List[EnergyRNN] = models['energyRnn']

        trainLoader    = dataLoader(
            trainSet,    hParams.batch_size, hParams.train_set_size, 
        )
        with hParams.eval():
            validateLoader = dataLoader(
                validateSet, hParams.batch_size, experiment.VALIDATE_SET_SIZE, 
            )

        vae.train()
        [x.train() for x in predRnns]
        [x.train() for x in energyRnns]
    for batch_i, (video_batch, traj_batch) in enumerate(
        trainLoader, 
    ):
        with profiler('train'):
            (
                lossTree, reconstructions, img_predictions, 
                z, z_hat, extra_logs, 
            ) = forward(
                epoch, batch_i, experiment, hParams, 
                video_batch, traj_batch, 
                vae, predRnns, energyRnns, profiler, 
                False, False, 
            )
        with profiler('sum loss'):
            total_loss = lossTree.sum(
                hParams.lossWeightTree, epoch, 
            )
            if hParams.lr_diminish is None:
                scaled_loss = total_loss
            else:
                scaled_loss = total_loss * hParams.lr_diminish(
                    epoch, hParams, 
                )
        with profiler('good', 'backward'):
            optim.zero_grad()
            scaled_loss.backward()
        with profiler('grad norm'):
            params = getParams(optim)
            grad_norm = getGradNorm(params)
            if hParams.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    params, hParams.grad_clip, 
                )
        with profiler('good', 'step'):
            optim.step()
        with profiler('log losses'):
            lossLogger.eat(
                epoch, batch_i, True, profiler, 
                lossTree, hParams.lossWeightTree, [
                    ('grad_norm', grad_norm), 
                    *extra_logs, 
                ], flush=False, 
            )

    with profiler(f'line {inspect.getframeinfo(inspect.currentframe()).lineno}'):
        vae.eval()
        [x.eval() for x in predRnns]
        [x.eval() for x in energyRnns]
    with torch.no_grad(), hParams.eval():
        for batch_i, (video_batch, traj_batch) in enumerate(
            validateLoader, 
        ):
            with profiler('validate'):
                (
                    lossTree, reconstructions, img_predictions, 
                    z, z_hat, extra_logs, 
                ) = forward(
                    epoch, 0, experiment, hParams, 
                    video_batch, traj_batch, 
                    vae, predRnns, energyRnns, profiler, 
                    False, True, 
                )
            with profiler('log losses'):
                lossLogger.eat(
                    epoch, batch_i, False, profiler, 
                    lossTree, hParams.lossWeightTree, [
                        ('grad_norm', torch.tensor(0)), 
                        *extra_logs, 
                    ], flush=False, 
                )
        with profiler('logs.flush'):
            if epoch % 8 == 0:
                lossLogger.compressor.flush()

        if epoch % experiment.SLOW_EVAL_EPOCH_INTERVAL == 0:
            with profiler('save checkpoints'):
                saveModels(models, epoch, save_path)
            if hParams.lossWeightTree['vicreg'].weight == 0:
                with profiler('video eval'):
                    for name, dataset in [
                        ('train', trainSet), 
                        ('validate', validateSet), 
                    ]:
                        loader = dataLoader(dataset, 12)
                        videoEval(
                            epoch, save_path, name, 
                            experiment, hParams, *next(loader), 
                            vae, predRnns, energyRnns, profiler, 
                        )
    
    if epoch % 32 == 0:
        print(group_name, 'epoch', epoch, 'finished.')
    with profiler('report'):
        profiler.report()
    return True
