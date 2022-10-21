from os import path
from typing import Dict
import inspect

import torch
import torch.utils.data
import cv2
from torchWork import Profiler, LossLogger, saveModels, HAS_CUDA
from torchWork.utils import getGradNorm, getParams
import numpy as np

from shared import *
from forward_pass import forward
from vae import VAE
from rnn import RNN
from load_dataset import Dataset

COL_PAD = 1
Z_BAR_HEIGHT = 4

def dataLoader(dataset, batch_size, set_size=None):
    n_batches = None
    if set_size is not None:
        if set_size % batch_size:
            assert set_size < batch_size
            batch_size = set_size
        n_batches = set_size // batch_size
    batch_i = 0
    for batch in torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=0, 
    ):
        yield batch
        batch_i += 1
        if n_batches is not None and batch_i >= n_batches:
            return

def oneEpoch(
    group_name: str, epoch: int, 
    experiment, hParams: HyperParams, 
    models: Dict[str, torch.nn.Module], 
    optim: torch.optim.Optimizer, 
    trainSet: Dataset, validateSet: Dataset, 
    lossLogger: LossLogger, profiler: Profiler, 
    save_path: str, trainer_id: int, 
):
    if epoch > hParams.max_epoch:
        return False
    with profiler(f'line {inspect.getframeinfo(inspect.currentframe()).lineno}'):
        vae: VAE = models['vae']
        rnn: RNN = models['rnn']

        trainLoader    = dataLoader(
            trainSet,    hParams.batch_size, hParams.train_set_size, 
        )
        validateLoader = dataLoader(
            validateSet, hParams.batch_size, experiment.VALIDATE_SET_SIZE, 
        )

        vae.train()
        rnn.train()
        if HAS_CUDA:
            torch.cuda.synchronize()    # just for profiling
    for batch_i, (video_batch, traj_batch) in enumerate(
        trainLoader, 
    ):
        with profiler('train'):
            (
                lossTree, reconstructions, img_predictions, 
                z, z_hat, extra_logs, 
            ) = forward(
                epoch, experiment, hParams, 
                video_batch, traj_batch, 
                vae, rnn, profiler, False, 
            )
            if HAS_CUDA:
                torch.cuda.synchronize()    # just for profiling
        with profiler('sum loss'):
            total_loss = lossTree.sum(
                hParams.lossWeightTree, epoch, 
            )
            if HAS_CUDA:
                torch.cuda.synchronize()    # just for profiling
        with profiler('good', 'backward'):
            optim.zero_grad()
            total_loss.backward()
            if HAS_CUDA:
                torch.cuda.synchronize()    # just for profiling
        with profiler('grad norm'):
            params = getParams(optim)
            grad_norm = getGradNorm(params)
            torch.nn.utils.clip_grad_norm_(
                params, hParams.grad_clip, 
            )
            if HAS_CUDA:
                torch.cuda.synchronize()    # just for profiling
        with profiler('good', 'step'):
            optim.step()
            if HAS_CUDA:
                torch.cuda.synchronize()    # just for profiling
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
        rnn.eval()
    with torch.no_grad(), hParams.eval():
        for batch_i, (video_batch, traj_batch) in enumerate(
            validateLoader, 
        ):
            with profiler('validate'):
                (
                    lossTree, reconstructions, img_predictions, 
                    z, z_hat, extra_logs, 
                ) = forward(
                    epoch, experiment, hParams, 
                    video_batch, traj_batch, 
                    vae, rnn, profiler, False, 
                )
                if HAS_CUDA:
                    torch.cuda.synchronize()    # just for profiling
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

        if epoch % SLOW_EVAL_EPOCH_INTERVAL == 0:
            with profiler('save checkpoints'):
                saveModels(models, epoch, save_path)
            with profiler('avi'):
                for name, dataset in [
                    ('train', trainSet), 
                    ('validate', validateSet), 
                ]:
                    loader = dataLoader(dataset, 24)
                    evalAVIs(
                        epoch, save_path, name, 
                        experiment, hParams, *next(loader), 
                        vae, rnn, profiler, 
                    )
    
    if epoch % 32 == 0:
        print(group_name, 'epoch', epoch, 'finished.')
    if trainer_id == 0:
        with profiler('report'):
            profiler.report()
    return True

def evalAVIs(
    epoch, save_path, set_name, 
    experiment, hParams: HyperParams, 
    video_batch: torch.Tensor, traj_batch, 
    vae, rnn, profiler, 
):
    n_datapoints = video_batch.shape[0]
    (
        lossTree, reconstructions, img_predictions, 
        z, z_hat, extra_logs, 
    ) = forward(
        epoch, experiment, hParams, video_batch, traj_batch, 
        vae, rnn, profiler, True, 
    )
    filename = path.join(
        save_path, f'visualize_{set_name}_epoch_{epoch}.avi', 
    )
    col_width = RESOLUTION + COL_PAD
    row_height = (
        RESOLUTION + Z_BAR_HEIGHT * hParams.symm.latent_dim
    )
    width = col_width * n_datapoints
    height = 3 * row_height
    out = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*'RGBA'), 
        4, (width, height), 
    )
    for t in range(SEQ_LEN):
        frame = np.zeros((width, height, 3), dtype=np.uint8)
        for col_i in range(n_datapoints):
            x = col_i * col_width
            frame[x + RESOLUTION : x + col_width, :, :] = 255
            for row_i, img in enumerate([
                video_batch[col_i, t, :, :, :], 
                reconstructions[col_i, t, :, :, :], 
                img_predictions[
                    col_i, t - hParams.rnn_min_context, :, :, :, 
                ], 
            ]):
                y = row_i * row_height
                if row_i < 2 or t >= hParams.rnn_min_context:
                    frame[
                        x : x + RESOLUTION, 
                        y : y + RESOLUTION, :, 
                    ] = torch2np(img)
                    if row_i == 0:
                        continue
                    elif row_i == 1:
                        _z: np.ndarray = z.numpy()
                        _t = t
                    elif row_i == 2:
                        _z: np.ndarray = z_hat.numpy()
                        _t = t - hParams.rnn_min_context
                    y += RESOLUTION
                    cursorX = (
                        (_z[col_i, _t, :] + 2) * .25 * RESOLUTION
                    ).round().astype(np.int16)
                    for z_dim in range(hParams.symm.latent_dim):
                        x0 = (cursorX[z_dim] - 1).clip(0, RESOLUTION)
                        x1 = (cursorX[z_dim] + 1).clip(0, RESOLUTION)
                        frame[
                            x + x0 : x + x1, 
                            y : y + Z_BAR_HEIGHT, :, 
                        ] = 255
                        y += Z_BAR_HEIGHT
                else:
                    frame[
                        x : x + RESOLUTION, 
                        y : y + row_height : 4, :, 
                    ] = 255
        out.write(frame.transpose([1, 0, 2]))
    out.release()

    # print(f'Saved `{filename}`.')
