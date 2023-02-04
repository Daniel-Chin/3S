from __future__ import annotations

from os import path
from typing import *
from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np

from shared import *

class DataModality(EnumStr): pass
VIDEO = DataModality('VIDEO')
MUSIC = DataModality('MUSIC')

class DatasetDefinition:
    def __init__(self) -> None:
        self.data_modality: DataModality = None
        self.dataset_path: str = None
        self.validate_set_size: int = None
        self.seq_len: int = None
        self.actual_dim: int = None
        self.img_n_channels: int = None
        self.img_resolution: Tuple[int, int] = None

        self.songBox: SongBox = None
        self.musicDatasetConfig: MusicDatasetConfig = None

class SongBox(metaclass=ABCMeta):
    # Used as an input to MusicDataset. 
    # Defines how the songs are generated. 
    @abstractmethod
    def __init__(self) -> None:
        self.name: str = None
        self.n_notes_per_song: int = None

    @abstractmethod
    def GenSongs(self, pitch_range: range) -> Generator[
        Tuple[int, Song], None, None, 
    ]:
        raise NotImplemented

class MusicDatasetConfig:
    def __init__(self) -> None:
        self.sound_font_path: str = None

        self.sr: int = 16000
        self.hop_len: int = 512
        self.win_len: int = 1024
        self.n_hops_per_note: int = 4
        self.n_hops_between_notes: int = 1
        self.fade_out_n_samples: int = 512

    def ready(self):
        assert self.win_len == 2 * self.hop_len
        self.encode_step = self.n_hops_per_note + self.n_hops_between_notes
        self.n_samples_per_note = self.hop_len * self.n_hops_per_note
        self.n_samples_between_notes = self.hop_len * self.n_hops_between_notes
        self.note_duration = self.n_samples_per_note / self.sr
        self.note_interval = (self.n_samples_per_note + self.n_samples_between_notes) / self.sr
        self.fade_out_filter = np.linspace(1, 0, self.fade_out_n_samples)
        self.n_bins = self.win_len // 2 + 1

defaultMusicDatasetConfig = MusicDatasetConfig()
defaultMusicDatasetConfig.sr = 16000
defaultMusicDatasetConfig.hop_len = 512
defaultMusicDatasetConfig.win_len = 1024
defaultMusicDatasetConfig.n_hops_per_note = 4
defaultMusicDatasetConfig.n_hops_between_notes = 1
defaultMusicDatasetConfig.fade_out_n_samples = 512
defaultMusicDatasetConfig.ready()

class IonianScalesSongBox(SongBox):
    IONIAN_SCALE = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]

    def __init__(self) -> None:
        self.name = 'ionian_scales'
        self.n_notes_per_song = 15

        self.scale_max = max(self.IONIAN_SCALE)
        assert min(self.IONIAN_SCALE) == self.IONIAN_SCALE[0] == 0

    def GenSongs(self, pitch_range: range) -> Generator[
        Tuple[int, Song], None, None, 
    ]:
        start_pitch = pitch_range.start
        while start_pitch + self.scale_max in pitch_range:
            notes = []
            for d_pitch in self.IONIAN_SCALE:
                pitch = start_pitch + d_pitch
                notes.append(Note(False, pitch))
            yield start_pitch, Song(notes)
            start_pitch += 1

dF = DatasetDefinition()
dF.data_modality = MUSIC
dF.dataset_path = path.abspath(
    '../datasets/ionian_scales_fr3gm', 
)
dF.songBox = IonianScalesSongBox()
dF.musicDatasetConfig = deepcopy(defaultMusicDatasetConfig)
dF.musicDatasetConfig.sound_font_path = path.abspath(
    '../sound_fonts/FluidR3_GM/FluidR3_GM.sf2', 
)
dF.validate_set_size = 476
dF.seq_len = dF.songBox.n_notes_per_song
dF.actual_dim = 2
dF.img_n_channels = 1
dF.img_resolution = (
    dF.musicDatasetConfig.n_bins, 
    dF.musicDatasetConfig.encode_step, 
)

ionianScales_fr3gm = dF
del dF

dF = DatasetDefinition()
dF.data_modality = VIDEO
dF.dataset_path = path.abspath(
    '../datasets/bounce'
)
dF.validate_set_size = 64
dF.seq_len = 20
dF.actual_dim = 3
dF.img_n_channels = 3
dF.img_resolution = (32, 32)

bounceSingleColor = dF
del dF


dF = DatasetDefinition()
dF.data_modality = VIDEO
dF.dataset_path = path.abspath(
    '../datasets/two_body'
)
dF.validate_set_size = 64
dF.seq_len = 25
dF.actual_dim = 6
dF.img_n_channels = 3
dF.img_resolution = (32, 32)

twoBody = dF
del dF
