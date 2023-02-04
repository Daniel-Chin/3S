from __future__ import annotations

import os
from os import path
import json
from typing import *
from functools import lru_cache
from copy import copy
import random

import numpy as np
import torch
import torch.utils.data
from PIL import Image
import tqdm
import librosa
from scipy.signal import stft

from shared import *
from physics_shared import Body
from dataset_definitions import (
    VIDEO, MUSIC, 
    DatasetDefinition, SongBox, MusicDatasetConfig, 
)
from synth_music_datasets import INDEX_FILENAME

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, datasetDefinition: DatasetDefinition, 
        is_train_not_validate: bool, 
        size: Optional[int], device=None, 
        dont_load: bool = False, do_shuffle: bool = False, 
    ) -> None:
        super().__init__()

        self.datasetDefinition = datasetDefinition
        self.is_train_not_validate = is_train_not_validate
        self.size = size
        self.device = device

        self.video_set: Optional[torch.Tensor] = None
        self.label_set: Optional[torch.Tensor] = None

        if not dont_load:
            self.load(do_shuffle)

    def load(self, do_shuffle):
        dDef = self.datasetDefinition
        if self.is_train_not_validate:
            which_set = 'train'
        else:
            which_set = 'validate'
        dataset_path = path.join(dDef.dataset_path, which_set)
        prev_cwd = os.getcwd()
        os.chdir(dataset_path)
        if self.size is None:
            if self.is_train_not_validate:
                assert dDef.data_modality is MUSIC
                # truncating the music set -> instrument imbalanace
                self.size = len(self.getMusicIndex(dataset_path))
            else:
                self.size = dDef.validate_set_size
        self.video_set = torch.zeros((
            self.size, 
            dDef.seq_len,
            dDef.img_n_channels, 
            *dDef.img_resolution,
        ))
        self.label_set = torch.zeros((
            self.size, 
            dDef.seq_len,
            dDef.actual_dim,
        ))
        if dDef.data_modality is VIDEO:
            assert not do_shuffle
            index = range(self.size)
        else:
            index = self.getMusicIndex(dataset_path)
            if do_shuffle:
                index = random.sample(index, self.size)
            else:
                index = index[:self.size]
            self.map = {}
        for data_i in tqdm.trange(
            len(index), desc=f'load {which_set} set', 
        ):
            if dDef.data_modality is VIDEO:
                self.loadOneVideo(data_i)
            else:
                self.loadOneMusic(data_i, *index[data_i])
        os.chdir(prev_cwd)

        if self.device is not None:
            print(f'Moving dataset to {self.device}...', flush=True)
            self.video_set = self.video_set.to(self.device)
            self.label_set = self.label_set.to(self.device)
        
        if dDef.data_modality is MUSIC:
            self.video_set = self.video_set - self.video_set.mean()
            self.video_set = self.video_set / self.video_set.std()
    
    @lru_cache()
    def getMusicIndex(self, dataset_path):
        with open(path.join(
            dataset_path, INDEX_FILENAME, 
        ), 'r') as f:
            index: List[Tuple[str, int, Any]] = json.load(f)
        return index
    
    def loadOneVideo(self, data_i: int):
        dDef = self.datasetDefinition
        with open(os.path.join(
            str(data_i), TRAJ_FILENAME, 
        ), 'r') as f:
            trajectory: List[List[Body]] = []
            for bodies_json in json.load(f):
                bodies = []
                trajectory.append(bodies)
                for body_json in bodies_json:
                    bodies.append(Body().fromJSON(body_json))
        for t in range(dDef.seq_len):
            for body_i, body in enumerate(trajectory[t]):
                self.label_set[
                    data_i, t, 3 * body_i : 3 * (body_i + 1)
                ] = torch.from_numpy(body.position)
            img = Image.open(os.path.join(
                str(data_i), f'{t}.png', 
            ))
            torchImg = img2Tensor(img)
            for c in range(dDef.img_n_channels):
                self.video_set[
                    data_i, t, c, :, :
                ] = torchImg[:, :, c]
    
    def loadOneMusic(
        self, data_i: int, 
        instrument_name: str, song_id: int, song_json, 
    ):
        config  = self.datasetDefinition.musicDatasetConfig
        songBox = self.datasetDefinition.songBox
        wav_name = f'{instrument_name}-{song_id}.wav'
        audio, _ = librosa.load(wav_name, sr=config.sr)
        _, _, spectrogram = stft(
            audio, nperseg=config.win_len, 
            noverlap=config.win_len - config.hop_len, 
            nfft=config.win_len, 
        )
        log_mag = torch.Tensor(np.abs(spectrogram[:, 1:])).log()
        # print('log_mag', log_mag.mean(), '+-', log_mag.std())
        for note_i in range(songBox.n_notes_per_song):
            self.video_set[data_i, note_i, 0, :, :] = log_mag[
                :, 
                note_i * config.encode_step : (note_i + 1) * config.encode_step, 
            ]
        song = Song.fromJSON(song_json)
        self.label_set[data_i, :, :] = song.toTensor()
        self.map[wav_name] = (self.video_set[data_i, :, :, :, :], song)
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return (
            self.video_set[index, :, :, :, :], 
            self.label_set[index, :, :], 
        )
    
    def __copy__(self):
        other = __class__(
            self.datasetDefinition, 
            self.is_train_not_validate, 
            self.size, 
            self.device, 
            True, 
        )
        other.video_set = self.video_set
        other.label_set = self.label_set
        return other
    
    def truncate(self, new_size: int):
        assert self.datasetDefinition.data_modality is VIDEO
        if new_size == self.size:
            return self
        assert new_size < self.size
        other = copy(self)
        other.size = new_size
        other.video_set = other.video_set[:new_size, ...]
        other.label_set = other.label_set[:new_size, ...]
        return other

def img2Tensor(img):
    np_img = np.asarray(img)
    return (
        torch.from_numpy(np_img / 256).float()
    )

def dataLoader(dataset: Dataset, batch_size, set_size=None):
    # this drops the last non-full batch even for validation set. 
    batch_size = min(batch_size, dataset.size)
    if set_size is None or set_size == dataset.size:
        truncatedDataset = dataset
    else:
        truncatedDataset = dataset.truncate(set_size)
    for batch in torch.utils.data.DataLoader(
        truncatedDataset, batch_size, shuffle=True, 
        drop_last=True, 
        num_workers=0, 
    ):
        batch: Tuple[torch.Tensor, torch.Tensor]
        yield batch

@lru_cache()
def getImageSet(dataset: Dataset):
    # flatten the videos to images. 
    _shape = dataset.video_set.shape
    image_set = dataset.video_set.view(
        _shape[0] * _shape[1], *_shape[2:], 
    )
    _shape = dataset.label_set.shape
    traj_set = dataset.label_set.view(
        _shape[0] * _shape[1], _shape[2], 
    )
    return image_set, traj_set

def printStats(dataset: Dataset):
    _, traj_set = getImageSet(dataset)
    print('label stats:')
    print('  mean:', traj_set.mean(dim=0))
    print('  std: ', traj_set.std (dim=0))

if __name__ == '__main__':
    from dataset_definitions import *
    dataset = Dataset(bounceSingleColor, True, 16)
    loader = dataLoader(dataset, 16, 8)
    for i, (x, y) in enumerate(loader):
        print(i, x.shape, y.shape)

    printStats(dataset)
