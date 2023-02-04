from os import path

import numpy as np
import torch
from matplotlib import pyplot as plt

from shared import *
from forward_pass import forward

def musicEval(
    epoch, save_path, set_name, 
    experiment, hParams: HyperParams, 
    video_batch: torch.Tensor, traj_batch, 
    vae, predRnns, energyRnns, profiler, 
):
    (
        n_datapoints, n_notes_per_song, IMG_N_CHANNELS, N_BINS, ENCODE_STEP, 
    ) = video_batch.shape
    filename = path.join(
        save_path, f'visualize_{set_name}_epoch_{epoch}.pdf', 
    )
    (
        lossTree, reconstructions, img_predictions, 
        z, z_hat, extra_logs, 
    ) = forward(
        epoch, 0, experiment, hParams, video_batch, traj_batch, 
        vae, predRnns, energyRnns, profiler, True, True, 
    )
    vmin = min(video_batch.min(), reconstructions.min(), img_predictions.min())
    vmax = max(video_batch.max(), reconstructions.max(), img_predictions.max())

    fig, axeses = plt.subplots(
        4, n_datapoints, sharex=True, 
    )
    X = (np.arange(n_notes_per_song) + .5) * ENCODE_STEP
    for col_i in range(n_datapoints):
        col_axes = [axeses[i][col_i] for i in range(4)]
        col_axes[0].imshow(
            viewAsSpectrogram(video_batch    [col_i, ...]), 
            aspect='auto', extent=(
                0, 
                n_notes_per_song * ENCODE_STEP, N_BINS, 0, 
            ), vmin=vmin, vmax=vmax, 
        )
        col_axes[1].imshow(
            viewAsSpectrogram(reconstructions[col_i, ...]), 
            aspect='auto', extent=(
                0, 
                n_notes_per_song * ENCODE_STEP, N_BINS, 0, 
            ), vmin=vmin, vmax=vmax, 
        )
        col_axes[2].imshow(
            viewAsSpectrogram(img_predictions[col_i, ...]), 
            aspect='auto', extent=(
                hParams.rnn_min_context * ENCODE_STEP, 
                n_notes_per_song * ENCODE_STEP, N_BINS, 0, 
            ), vmin=vmin, vmax=vmax, 
        )
        col_axes[3].plot(
            X, 
            z    [col_i, :, 0], label='encoded', 
            linewidth=1, 
        )
        col_axes[3].plot(
            X[hParams.rnn_min_context:], 
            z_hat[col_i, :, 0], label='predicted', 
            linewidth=1, 
        )
        for ax in col_axes[:3]:
            ax.tick_params(
                bottom=False, 
                left=False, 
                labelbottom=False, 
                labelleft=False, 
            )
        if col_i == 0:
            col_axes[0].set_ylabel('Ground-truth')
            col_axes[1].set_ylabel('Self-recon')
            col_axes[2].set_ylabel('Prediction')
            col_axes[3].set_ylabel('z')
            col_axes[3].legend(
                loc='upper left', bbox_to_anchor=(0, 0), 
            )
    plt.savefig(filename)
    plt.close('all')

def viewAsSpectrogram(batch: torch.Tensor):
    n_notes_per_song, IMG_N_CHANNELS, N_BINS, ENCODE_STEP = batch.shape
    return batch.squeeze(1).permute(0, 2, 1).reshape(
        n_notes_per_song * ENCODE_STEP, N_BINS, 
    ).T.cpu().numpy()
