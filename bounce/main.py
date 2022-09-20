import os
import shutil
from itertools import count
import torch
import numpy as np
from PIL import Image, ImageDraw

try:
    from myTorch import LossLogger
    from streamProfiler import StreamProfiler
    from roundRobinSched import roundRobinSched
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
from vae import VAE
from rnn import RNN
from loadDataset import loadDataset, TRAIN_PATH, VALIDATE_PATH
from train import (
    oneEpoch, HAS_CUDA, DEVICE, BATCH_SIZE, oneBatch, 
    Config, 
)
from experiments import RAND_INIT_TIMES, EXPERIMENTS
from loadModels import loadModels

EPOCH_INTERVAL = 2000

class Trainer:
    def __init__(
        self, vae, rnn, optim, 
        train_videos, validate_videos, 
        train_traj, validate_traj, 
        rand_init_i, config: Config, 
    ) -> None:
        self.vae = vae
        self.rnn = rnn
        self.optim = optim
        self.train_videos = train_videos
        self.train_traj = train_traj
        self.validate_videos = validate_videos
        self.validate_traj = validate_traj

        self.config = config
        self.experiment_path = renderExperimentPath(
            rand_init_i, config, 
        )
        os.makedirs(self.experiment_path, exist_ok=True)

        self.entered = False
        self.prev_cwd = None

        with self:
            self.lossLogger = LossLogger('losses.log')
        self.lossLogger.clearFile()

        self.epoch = 0
    
    def __enter__(self):
        assert not self.entered
        self.prev_cwd = os.getcwd()
        os.chdir(self.experiment_path)
        self.entered = True
    
    def __exit__(self, *_):
        os.chdir(self.prev_cwd)
        self.entered = False
        return False
    
    def oneEpoch(self, profiler):
        with self:
            oneEpoch(
                profiler, 
                self.epoch, self.vae, self.rnn, self.optim, 
                self.train_videos, self.validate_videos, 
                self.train_traj, self.validate_traj, 
                self.lossLogger, self.config, *self.config, 
            )
            if self.epoch % EPOCH_INTERVAL == 0:
                torch.save(
                    self.vae.state_dict(), 
                    f'{self.epoch}_vae.pt', 
                )
                torch.save(
                    self.rnn.state_dict(), 
                    f'{self.epoch}_rnn.pt', 
                )
            self.epoch += 1

def main():
    try:
        shutil.rmtree(EXPERIMENTS_PATH)
    except FileNotFoundError:
        pass
    train_videos   , train_traj    = loadDataset(
        TRAIN_PATH,       TRAIN_SET_SIZE, DEVICE, 
    )
    validate_videos, validate_traj = loadDataset(
        VALIDATE_PATH, VALIDATE_SET_SIZE, DEVICE, 
    )
    assert TRAIN_SET_SIZE % BATCH_SIZE == 0
    assert VALIDATE_SET_SIZE == BATCH_SIZE
    trainers = []
    for _, config in EXPERIMENTS:
        for rand_init_i in range(RAND_INIT_TIMES):
            vae, rnn = loadModels(config)
            optim = torch.optim.Adam(
                [
                    *vae.parameters(), *rnn.parameters(), 
                ], lr=config.lr, 
            )
            trainers.append(Trainer(
                vae, rnn, optim, 
                train_videos, validate_videos, 
                train_traj, validate_traj, 
                rand_init_i, config, 
            ))
    profiler = StreamProfiler(
        DO_PROFILE=False, filename='profiler.log', 
    )   # Obselete, since we are scheduling multiple trainers. 
    print('Training starts...', flush=True)
    for i in roundRobinSched(len(trainers)):
        trainer: Trainer = trainers[i]
        # print(trainer.config)
        trainer.oneEpoch(profiler)
        if trainer.epoch % EPOCH_INTERVAL == 0:
            if trainer.config.vae_loss_coef != 0:
                with torch.no_grad():
                    with trainer:
                        for label, videos in [
                            ('train', train_videos), 
                            ('validate', validate_videos), 
                        ]:
                            evalGIFs(
                                trainer.epoch, 
                                trainer.vae, 
                                trainer.rnn, 
                                videos[:24, :, :, :, :], 
                                trainer.config.rnn_min_context, 
                                label,
                            )

def evalGIFs(
    epoch, vae: VAE, rnn: RNN, dataset: torch.Tensor, 
    rnn_min_context, label, 
):
    n_datapoints = dataset.shape[0]
    vae.eval()
    rnn.eval()
    predictions, reconstructions = oneBatch(
        vae, rnn, dataset, None, 0, 0, 0, False, False, False, 
        -20, rnn_min_context, 0, 0, 0, 1, 0, F.mse_loss, 
        0, False, False, False, 
        visualize=True, batch_size=n_datapoints, 
    )
    frames = []
    for t in range(SEQ_LEN):
        frame = Image.new('RGB', (
            RESOLUTION * n_datapoints, RESOLUTION * 3, 
        ))
        frames.append(frame)
        imDraw = ImageDraw.Draw(frame)
        for i in range(n_datapoints):
            frame.paste(
                torch2PIL(dataset[i, t, :, :, :]), 
                (i * RESOLUTION, 0 * RESOLUTION), 
            )
            frame.paste(
                torch2PIL(reconstructions[i, t, :, :, :]), 
                (i * RESOLUTION, 1 * RESOLUTION), 
            )
            if t >= rnn_min_context:
                frame.paste(
                    torch2PIL(predictions[
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

    filename = f'pred_{label}_{epoch}.gif'
    frames[0].save(
        filename, 
        save_all=True, append_images=frames[1:], 
        duration=300, loop=0, 
    )
    # print(f'Saved `{filename}`.')

if __name__ == '__main__':
    main()
