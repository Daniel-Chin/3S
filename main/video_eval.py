from os import path
import sys
import subprocess as sp
from contextlib import contextmanager

import cv2
import numpy as np
import torch
try:
    from async_std import PopenAsyncStd
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'async_std', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

from shared import *
from hyper_params import *
from forward_pass import forward

COL_MARGIN = 1
Z_BAR_HEIGHT = 3

VIDEO_SCALE = 2
FPS = 4

class VideoWriter:
    def __init__(self, width, height) -> None:
        self.scaled_dims = (
            width * VIDEO_SCALE, height * VIDEO_SCALE, 
        )
        self.args = [
            'ffmpeg', 
            '-s', f'{self.scaled_dims[0]}x{self.scaled_dims[1]}', 
            '-pixel_format', 'bgr24', '-f', 'rawvideo', 
            '-r', str(FPS), '-i', 'pipe:', 
            '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', 
            '-crf', '24', 
            '-y', 
        ]
    
    @contextmanager
    def context(self, filename):
        with PopenAsyncStd(
            [*self.args, filename], stdin=sp.PIPE, 
        ) as self.ffmpeg:
            try:
                yield self
            finally:
                ffmpeg = self.ffmpeg
                del self.ffmpeg
                ffmpeg.stdin.close()
                ffmpeg.wait()
    
    def write(self, img):
        img = cv2.resize(
            img, self.scaled_dims, 
            interpolation=cv2.INTER_NEAREST, 
        )
        poll = self.ffmpeg.poll()
        try:
            if poll is not None:
                raise EOFError(f'ffmpeg exited with {poll}')
            self.ffmpeg.stdin.write(img.tobytes())
        except (EOFError, BrokenPipeError):
            print('\nReporting collected out and err.')
            print('==== BEGIN ====\n')
            self.ffmpeg.reportCollectedOutErr()
            print('\n==== END ====\n')
            sys.stdout.flush()
            raise

def videoEval(
    epoch, save_path, set_name, 
    experiment, hParams: HyperParams, 
    video_batch: torch.Tensor, traj_batch, 
    vae, predRnns, energyRnns, profiler, 
):
    assert hParams.datasetDef.img_resolution[0] == hParams.datasetDef.img_resolution[1]
    RESOLUTION = hParams.datasetDef.img_resolution[0]
    n_datapoints = video_batch.shape[0]
    (
        lossTree, reconstructions, img_predictions, 
        z, z_hat, extra_logs, 
    ) = forward(
        epoch, 0, experiment, hParams, video_batch, traj_batch, 
        vae, predRnns, energyRnns, profiler, True, True, 
    )
    filename = path.join(
        save_path, f'visualize_{set_name}_epoch_{epoch}.mp4', 
    )
    col_width = RESOLUTION + COL_MARGIN
    row_heights = (
        RESOLUTION + 1, 
        RESOLUTION + Z_BAR_HEIGHT * hParams.symm.latent_dim, 
        RESOLUTION + Z_BAR_HEIGHT * hParams.symm.latent_dim, 
    )
    width = col_width * n_datapoints
    height = sum(row_heights)
    vid = VideoWriter(width, height)
    with vid.context(filename):
        for t in range(hParams.datasetDef.seq_len):
            frame = np.zeros((width, height, 3), dtype=np.uint8)
            for col_i in range(n_datapoints):
                x = col_i * col_width
                frame[x + RESOLUTION : x + col_width, :, :] = 255
                y = 0
                for row_i, img in enumerate([
                    video_batch, 
                    reconstructions, 
                    img_predictions, 
                ]):
                    if row_i < 2 or t >= hParams.rnn_min_context:
                        if row_i == 0:
                            _t = t
                        elif row_i == 1:
                            _z: np.ndarray = z.cpu().numpy()
                            _t = t
                        elif row_i == 2:
                            _z: np.ndarray = z_hat.cpu().numpy()
                            _t = t - hParams.rnn_min_context
                        frame[
                            x : x + RESOLUTION, 
                            y : y + RESOLUTION, :, 
                        ] = torch2np(img[col_i, _t, :, :, :]).transpose((1, 0, 2))
                        if row_i != 0:
                            _y = y + RESOLUTION
                            cursorX = (
                                (_z[col_i, _t, :] + 2) * .25 * RESOLUTION
                            ).round().astype(np.int16)
                            for z_dim in range(hParams.symm.latent_dim):
                                x0 = (cursorX[z_dim] - 1).clip(0, RESOLUTION)
                                x1 = (cursorX[z_dim] + 1).clip(0, RESOLUTION)
                                frame[
                                    x + x0 : x + x1, 
                                    _y : _y + Z_BAR_HEIGHT, :, 
                                ] = 255
                                _y += Z_BAR_HEIGHT
                    else:
                        cv2.line(
                            frame, (y - 1, x - 1), 
                            (y + row_heights[row_i], x + RESOLUTION), 
                            (255, 255, 255), 1, 
                        )
                    y += row_heights[row_i]
            vid.write(frame.transpose((1, 0, 2)))

    # print(f'Saved `{filename}`.')
