from __future__ import annotations

import os
from os import path
import json
from typing import *
from functools import lru_cache
from abc import ABCMeta, abstractmethod
from copy import copy

import numpy as np
import torch
import torch.utils.data
from PIL import Image
import tqdm
import librosa
from scipy.signal import stft

from shared import *
from physics_shared import Body
from music_dataset_shared import (
    INDEX_FILENAME, 
    Song, Note, SongBox, Config as MusicDatasetConfig, 
)

class Dataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()

        self.size: int = None
        self.video_set: torch.Tensor = None
        self.label_set: torch.Tensor = None
    
    def __len__(self):
        return self.size

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented
    
    @abstractmethod
    def __copy__(self) -> Dataset:
        raise NotImplemented
    
    @abstractmethod
    def truncate(self, new_size: int) -> Dataset:
        raise NotImplemented

class VideoDataset(Dataset):
    def __init__(
        self, dataset_path, size, SEQ_LEN, 
        ACTUAL_DIM: int, RESOLUTION: int, IMG_N_CHANNELS, 
        device=None, 
    ) -> None:
        super().__init__()

        self.size = size
        self.SEQ_LEN = SEQ_LEN
        self.ACTUAL_DIM = ACTUAL_DIM
        self.device = device

        if dataset_path is not None:
            prev_cwd = os.getcwd()
            os.chdir(dataset_path)
            video_set = torch.zeros((
                size, 
                SEQ_LEN, 
                IMG_N_CHANNELS, 
                RESOLUTION, 
                RESOLUTION, 
            ))
            label_set = torch.zeros((
                size, 
                SEQ_LEN, 
                ACTUAL_DIM, 
            ))
            for data_i in tqdm.trange(size, desc='load dataset'):
                with open(os.path.join(
                    str(data_i), TRAJ_FILENAME, 
                ), 'r') as f:
                    trajectory: List[List[Body]] = []
                    for bodies_json in json.load(f):
                        bodies = []
                        trajectory.append(bodies)
                        for body_json in bodies_json:
                            bodies.append(Body().fromJSON(body_json))
                for t in range(SEQ_LEN):
                    for body_i, body in enumerate(trajectory[t]):
                        label_set[
                            data_i, t, 3 * body_i : 3 * (body_i + 1)
                        ] = torch.from_numpy(body.position)
                    img = Image.open(os.path.join(
                        str(data_i), f'{t}.png', 
                    ))
                    torchImg = img2Tensor(img)
                    for c in range(IMG_N_CHANNELS):
                        video_set[
                            data_i, t, c, :, :
                        ] = torchImg[:, :, c]
            os.chdir(prev_cwd)

            if device is not None:
                print(f'Moving dataset to {device}...', flush=True)
                video_set = video_set.to(device)
                label_set = label_set.to(device)

            self.video_set = video_set
            self.label_set = label_set
    
    def __getitem__(self, index):
        return (
            self.video_set[index, :, :, :, :], 
            self.label_set[index, :, :], 
        )
    
    def __copy__(self):
        other = __class__(
            None, self.size, self.SEQ_LEN, self.ACTUAL_DIM, 
            self.device, 
        )
        other.video_set = self.video_set
        other.label_set = self.label_set
        return other
    
    def truncate(self, new_size: int):
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

def PersistentLoader(dataset, batch_size):
    while True:
        loader = torch.utils.data.DataLoader(
            dataset, batch_size, shuffle=True, 
            num_workers=0, 
        )
        for video_batch, traj_batch in loader:
            if video_batch.shape[0] != batch_size:
                break
            yield video_batch, traj_batch

def dataLoader(dataset: Dataset, batch_size, set_size=None):
    batch_size = min(batch_size, dataset.size)
    if set_size is not None:
        if set_size % batch_size:
            assert set_size < batch_size
            batch_size = set_size
    if set_size is None or set_size == dataset.size:
        truncatedDataset = dataset
    else:
        truncatedDataset = dataset.truncate(set_size)
    for batch in torch.utils.data.DataLoader(
        truncatedDataset, batch_size, shuffle=True, 
        drop_last=True, 
        num_workers=0, 
    ):
        batch: List[torch.Tensor]
        yield batch

@lru_cache()
def getImageSet(dataset: Dataset):
    # flatten the videos to images. 
    _shape = dataset.video_set.shape
    image_set = dataset.video_set.view(
        _shape[0] * _shape[1], _shape[2], _shape[3], _shape[4], 
    )
    _shape = dataset.label_set.shape
    traj_set = dataset.label_set.view(
        _shape[0] * _shape[1], _shape[2], 
    )
    return image_set, traj_set

def printStats(*args):
    _, traj_set = getImageSet(*args)
    print('mean:', traj_set.mean(dim=0))
    print('std: ', traj_set.std (dim=0))

class MusicDataset(Dataset):
    def __init__(
        self, songBox: SongBox, config: MusicDatasetConfig, 
        is_train_not_validate: bool, 
        size: Optional[int] = None, device=None, 
    ) -> None:
        super().__init__()

        if is_train_not_validate:
            which = 'train'
        else:
            which = 'validate'
        self.dataset_path = path.join(config.DATASET_PATH, which)

        with open(path.join(
            self.dataset_path, INDEX_FILENAME, 
        ), 'r') as f:
            index: List[Tuple[str, int, Any]] = json.load(f)
        index = index[:size]
        video_set = []
        label_set = []
        self.map = {}
        for (
            instrument_name, song_id, song_json, 
        ) in tqdm.tqdm(index, desc=f'load {which}'):
            wav_name = f'{instrument_name}-{song_id}.wav'
            datapoint = self.loadOneFile(
                config, songBox, wav_name, 
            )
            song = Song.fromJSON(song_json)
            video_set.append(datapoint)
            label_set.append(song.toTensor())
            self.map[wav_name] = (datapoint, song)
        self.size = len(index)
        video_set = torch.stack(video_set, dim=0)
        label_set = torch.stack(label_set, dim=0)
        if device is not None:
            print(f'Moving dataset to {device}...', flush=True)
            video_set = video_set.to(device)
            label_set = label_set.to(device)
        video_set = video_set - video_set.mean()
        video_set = video_set / video_set.std()
        self.video_set = video_set
        self.label_set = label_set
    
    def __getitem__(self, index):
        return (
            self.video_set[index, :, :, :], 
            self.label_set[index, :, :], 
        )

    def loadOneFile(
        self, config: MusicDatasetConfig, songBox: SongBox, 
        wav_name, 
    ):
        filename = path.join(
            self.dataset_path, wav_name, 
        )
        audio, _ = librosa.load(filename, sr=config.SR)
        _, _, spectrogram = stft(
            audio, nperseg=config.WIN_LEN, 
            noverlap=config.WIN_LEN - config.HOP_LEN, 
            nfft=config.WIN_LEN, 
        )
        log_mag = torch.Tensor(np.abs(spectrogram[:, 1:])).log()
        # print('log_mag', log_mag.mean(), '+-', log_mag.std())
        datapoint = torch.zeros((
            songBox.n_notes_per_song, config.N_BINS, 
            config.ENCODE_STEP, 
        ))
        for note_i in range(songBox.n_notes_per_song):
            datapoint[note_i, :, :] = log_mag[
                :, 
                note_i * config.ENCODE_STEP : (note_i + 1) * config.ENCODE_STEP, 
            ]
        return datapoint.unsqueeze(1)   # channel dim
    
    def truncate(self, new_size: int) -> Dataset:
        raise NotImplemented
    
    def __copy__(self) -> Dataset:
        raise NotImplemented
    
if __name__ == '__main__':
    # dataset = Dataset('../datasets/bounce/train', 128, 20, 3)
    # loader = PersistentLoader(dataset, 32)

    # dataset = Dataset('../datasets/bounce/train', 16, 20, 3)
    # loader = dataLoader(dataset, 16, 8)
    # for i, (x, y) in enumerate(loader):
    #     print(i, x.shape, y.shape)

    printStats('../datasets/bounce/validate', 16, 20, 3)
