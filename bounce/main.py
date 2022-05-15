import os
import shutil
from itertools import count
import torch
import numpy as np
from PIL import Image, ImageDraw

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
from vae import VAE
from rnn import RNN
from loadDataset import loadDataset, TRAIN_PATH, VALIDATE_PATH
from train import (
    oneEpoch, HAS_CUDA, DEVICE, BATCH_SIZE, oneBatch, 
    Config, 
)
from experiments import RAND_INIT_TIMES, EXPERIMENTS
from loadModels import loadModels

EPOCH_INTERVAL = 1000

class Trainer:
    def __init__(
        self, vae, rnn, optim, train_set, validate_set, 
        rand_init_i, config: Config, 
    ) -> None:
        self.vae = vae
        self.rnn = rnn
        self.optim = optim
        self.train_set = train_set
        self.validate_set = validate_set

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
    
    def __enter__(self):
        assert not self.entered
        self.prev_cwd = os.getcwd()
        os.chdir(self.experiment_path)
        self.entered = True
    
    def __exit__(self, *_):
        os.chdir(self.prev_cwd)
        self.entered = False
        return False
    
    def oneEpoch(self, epoch, profiler):
        with self:
            oneEpoch(
                profiler, 
                epoch, self.vae, self.rnn, self.optim, 
                self.train_set, self.validate_set, 
                self.lossLogger, *self.config, 
            )
            if epoch % EPOCH_INTERVAL == 0:
                torch.save(
                    self.vae.state_dict(), f'{epoch}_vae.pt', 
                )
                torch.save(
                    self.rnn.state_dict(), f'{epoch}_rnn.pt', 
                )

def main():
    try:
        shutil.rmtree(EXPERIMENTS_PATH)
    except FileNotFoundError:
        pass
    train_set    = loadDataset(
        TRAIN_PATH,       TRAIN_SET_SIZE, DEVICE, 
    )
    validate_set = loadDataset(
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
                ], lr=.001, 
            )
            trainers.append(Trainer(
                vae, rnn, optim, train_set, validate_set, 
                rand_init_i, config, 
            ))
    profiler = StreamProfiler(
        DO_PROFILE=False, filename='profiler.log', 
    )
    for epoch in count():
        for trainer in trainers:
            trainer: Trainer
            # print(trainer.config)
            trainer.oneEpoch(epoch, profiler)
        if epoch % EPOCH_INTERVAL == 0:
            # print('making GIFs.')
            with torch.no_grad():
                for trainer in trainers:
                    with trainer:
                        evalGIFs(
                            epoch, 
                            trainer.vae, 
                            trainer.rnn, 
                            validate_set[:24, :, :, :, :], 
                            trainer.config.rnn_min_context, 
                        )
            # print('GIFs made.')

def evalGIFs(
    epoch, vae: VAE, rnn: RNN, dataset: torch.Tensor, 
    rnn_min_context, 
):
    n_datapoints = dataset.shape[0]
    vae.eval()
    rnn.eval()
    predictions, reconstructions = oneBatch(
        vae, rnn, dataset, 0, 0, 0, False, False, False, 
        -20, rnn_min_context, 
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

    filename = f'pred_{epoch}.gif'
    frames[0].save(
        filename, 
        save_all=True, append_images=frames[1:], 
        duration=300, loop=0, 
    )
    # print(f'Saved `{filename}`.')

if __name__ == '__main__':
    main()
