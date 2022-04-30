import os
from itertools import count
import torch
from PIL import Image

try:
    from myTorch import LossLogger
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'myTorch', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

from vae import VAE
from rnn import RNN
from makeDataset import (
    TRAIN_SET_SIZE, VALIDATE_SET_SIZE, SEQ_LEN, RESOLUTION, 
)
from loadDataset import loadDataset, TRAIN_PATH, VALIDATE_PATH
from train import (
    oneEpoch, HAS_CUDA, DEVICE, BATCH_SIZE, oneBatch, 
    RNN_MIN_CONTEXT, Config, 
)

RAND_INIT_TIMES = 1
EXPERIMENTS = [
    Config(.001, 1, 1, 0, 0), 
]

EXPERIMENTS_PATH = './experiments'

def loadModel():
    # future: load model from disk
    vae = VAE()
    rnn = RNN()
    if HAS_CUDA:
        vae = vae.cuda()
        rnn = rnn.cuda()
    return vae, rnn

def renderExperimentPath(
    rand_init_i, config: Config, 
):
    return os.path.join(
        EXPERIMENTS_PATH, f'''{(
            rand_init_i, *config, 
        )}'''
    )

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
    
    def oneEpoch(self, epoch):
        with self:
            oneEpoch(
                epoch, self.vae, self.rnn, self.optim, 
                self.train_set, self.validate_set, 
                self.lossLogger, *self.config, 
            )
            if epoch % 1 == 0:
                torch.save(
                    self.vae.state_dict(), f'{epoch}_vae.pt', 
                )
                torch.save(
                    self.rnn.state_dict(), f'{epoch}_rnn.pt', 
                )

def main():
    train_set    = loadDataset(   TRAIN_PATH, DEVICE)
    validate_set = loadDataset(VALIDATE_PATH, DEVICE)
    assert TRAIN_SET_SIZE % BATCH_SIZE == 0
    assert VALIDATE_SET_SIZE == BATCH_SIZE
    trainers = []
    for config in EXPERIMENTS:
        for rand_init_i in range(RAND_INIT_TIMES):
            vae, rnn = loadModel()
            optim = torch.optim.Adam(
                [
                    *vae.parameters(), *rnn.parameters(), 
                ], lr=.001, 
            )
            trainers.append(Trainer(
                vae, rnn, optim, train_set, validate_set, 
                rand_init_i, config, 
            ))
    for epoch in count():
        for trainer in trainers:
            trainer: Trainer
            print(trainer.config)
            trainer.oneEpoch(epoch)
        if epoch % 20 == 0:
            print('making GIFs.')
            with torch.no_grad():
                for trainer in trainers:
                    with trainer:
                        evalGIFs(
                            trainer.vae.cpu(), 
                            trainer.rnn.cpu(), 
                            validate_set[:24, :, :, :, :], 
                        )
            print('GIFs made.')

def torch2PIL(torchImg: torch.Tensor):
    return Image.fromarray(torchImg.numpy() * 255).convert('L')

def evalGIFs(vae: VAE, rnn: RNN, dataset: torch.Tensor):
    n_datapoints = dataset.shape[0]
    vae.eval()
    rnn.eval()
    predictions, reconstructions = oneBatch(
        vae, rnn, dataset, 0, 0, 0, 
        True, batch_size=n_datapoints, 
    )
    frames = []
    for t in range(SEQ_LEN):
        frame = Image.new('L', (
            RESOLUTION * n_datapoints, RESOLUTION * 3, 
        ))
        frames.append(frame)
        for i in range(n_datapoints):
            frame.paste(
                torch2PIL(dataset[i, t, 0, :, :]), 
                (i * RESOLUTION, 0 * RESOLUTION), 
            )
            frame.paste(
                torch2PIL(reconstructions[i, t, 0, :, :]), 
                (i * RESOLUTION, 1 * RESOLUTION), 
            )
            if t >= RNN_MIN_CONTEXT:
                frame.paste(
                    torch2PIL(predictions[
                        i, t - RNN_MIN_CONTEXT, 0, :, :, 
                    ]), 
                    (i * RESOLUTION, 2 * RESOLUTION), 
                )
    filename = 'pred.gif'
    frames[0].save(
        filename, 
        save_all=True, append_images=frames[1:], 
        duration=300, loop=0, 
    )
    print(f'Saved `{filename}`.')

main()