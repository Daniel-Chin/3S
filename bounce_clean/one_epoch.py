from os import path
from typing import List, Dict
import inspect

import torch
import torch.utils.data
from PIL import Image, ImageDraw
from torchWork import Profiler, LossLogger
from torchWork.utils import getGradNorm, getParams

from shared import *
from forward_pass import forward
from vae import VAE
from rnn import RNN

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
    group_name: str, epoch: int, hParams: HyperParams, 
    models: Dict[str, torch.nn.Module], 
    optim: torch.optim.Optimizer, 
    trainSet, validateSet, 
    lossLogger: LossLogger, profiler: Profiler, 
    save_path: str, 
):
    with profiler('epochs'):
        with profiler(f'line {inspect.getframeinfo(inspect.currentframe()).lineno}'):
            vae: VAE = models['vae']
            rnn: RNN = models['rnn']

            trainLoader    = dataLoader(
                trainSet,    hParams.batch_size, hParams.train_set_size, 
            )
            validateLoader = dataLoader(
                validateSet, hParams.batch_size, VALIDATE_SET_SIZE, 
            )

            vae.train()
            rnn.train()
        for batch_i, (video_batch, traj_batch) in enumerate(
            trainLoader, 
        ):
            with profiler('train'):
                (
                    lossTree, reconstructions, img_predictions, 
                    extra_logs, 
                ) = forward(
                    epoch, hParams, video_batch, traj_batch, 
                    vae, rnn, profiler, False, 
                )
            with profiler('sum loss'):
                total_loss = lossTree.sum(
                    hParams.lossWeightTree, epoch, 
                )
            with profiler('good'):
                optim.zero_grad()
                total_loss.backward()
            with profiler('grad norm'):
                params = getParams(optim)
                grad_norm = getGradNorm(params)
                torch.nn.utils.clip_grad_norm_(
                    params, hParams.grad_clip, 
                )
            with profiler('good'):
                optim.step()
            with profiler('log losses'):
                lossLogger.eat(
                    epoch, batch_i, True, 
                    lossTree, hParams.lossWeightTree, [
                        ('grad_norm', grad_norm), 
                        *extra_logs, 
                    ], 
                    # profiler, 
                )

        with profiler(f'line {inspect.getframeinfo(inspect.currentframe()).lineno}'):
            vae.eval()
            rnn.eval()
        with torch.no_grad():
            for batch_i, (video_batch, traj_batch) in enumerate(
                validateLoader, 
            ):
                with profiler('validate'):
                    (
                        lossTree, reconstructions, img_predictions, 
                        extra_logs, 
                    ) = forward(
                        epoch, hParams, video_batch, traj_batch, 
                        vae, rnn, profiler, True, 
                    )
                with profiler('log losses'):
                    lossLogger.eat(
                        epoch, batch_i, False, 
                        lossTree, hParams.lossWeightTree, [
                            ('grad_norm', 0), 
                            *extra_logs, 
                        ], 
                        # profiler, 
                    )

            if epoch % EPOCH_INTERVAL == 0:
                with profiler('save checkpoints'):
                    for key, model in models.items():
                        torch.save(model.state_dict(), path.join(
                            save_path, f'{key}_epoch_{epoch}.pt', 
                        ))

                with profiler('gif'):
                    for name, dataset in [
                        ('train', trainSet), 
                        ('validate', validateSet), 
                    ]:
                        loader = dataLoader(dataset, 24)
                        evalGIFs(
                            epoch, next(loader)[0], save_path, 
                            hParams.rnn_min_context, name,
                            reconstructions, img_predictions, 
                        )
        
        print(group_name, 'epoch', epoch, 'finished.', flush=True)
    profiler.report()

def evalGIFs(
    epoch, video_set: torch.Tensor, save_path, 
    rnn_min_context, set_name, reconstructions, img_predictions, 
):
    n_datapoints = video_set.shape[0]
    frames: List[Image.Image] = []
    for t in range(SEQ_LEN):
        frame = Image.new('RGB', (
            RESOLUTION * n_datapoints, RESOLUTION * 3, 
        ))
        frames.append(frame)
        imDraw = ImageDraw.Draw(frame)
        for i in range(n_datapoints):
            frame.paste(
                torch2PIL(video_set[i, t, :, :, :]), 
                (i * RESOLUTION, 0 * RESOLUTION), 
            )
            frame.paste(
                torch2PIL(reconstructions[i, t, :, :, :]), 
                (i * RESOLUTION, 1 * RESOLUTION), 
            )
            if t >= rnn_min_context:
                frame.paste(
                    torch2PIL(img_predictions[
                        i, t - rnn_min_context, :, :, :, 
                    ]), 
                    (i * RESOLUTION, 2 * RESOLUTION), 
                )
            else:
                imDraw.line((
                    (i + .2) * RESOLUTION, 2.2 * RESOLUTION, 
                    (i + .8) * RESOLUTION, 2.8 * RESOLUTION, 
                ), fill='white')
                imDraw.line((
                    (i + .2) * RESOLUTION, 2.8 * RESOLUTION, 
                    (i + .8) * RESOLUTION, 2.2 * RESOLUTION, 
                ), fill='white')

    filename = path.join(
        save_path, f'visualize_{set_name}_epoch_{epoch}.gif', 
    )
    frames[0].save(
        filename, 
        save_all=True, append_images=frames[1:], 
        duration=300, loop=0, 
    )
    # print(f'Saved `{filename}`.')