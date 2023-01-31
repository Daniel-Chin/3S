__all__ = [
    'Note', 'Song', 'SongBox', 'Config', 'synthDataset', 
]

import os
from os import path
import shutil
import json
from typing import *
from abc import ABCMeta, abstractmethod
from functools import lru_cache
from itertools import combinations_with_replacement
from time import time

import pretty_midi as pm
from music21.instrument import Instrument, Piano
try:
    # my fork
    from midi2audio_fork.midi2audio import FluidSynth
except ImportError:
    print('Warning: falling back to non-forked midi2audio.')
    print('You can get the fork at')
    print('https://github.com/daniel-chin/midi2audio')
    print('and place it as', path.abspath('midi2audio_fork/'))
    # fallback
    from midi2audio import FluidSynth
import librosa
import soundfile
import numpy as np
from tqdm import tqdm

from intruments_and_ranges import intruments_ranges

BEND_RATIO = .5  # When MIDI specification is incomplete...
BEND_MAX = 8191
GRACE_END = .1
TEMP_MIDI_FILE = path.abspath('./temp/temp.mid')
TEMP_WAV_FILE  = path.abspath('./temp/temp.wav')
os.makedirs('temp', exist_ok=True)

class Note:
    def __init__(self, is_rest: bool, pitch: Optional[int] = None):
        assert is_rest == (pitch is None)
        self.is_rest = is_rest
        if not is_rest:
            self.pitch = pitch
    
    def toJSON(self):
        if self.is_rest:
            return -1
        else:
            return self.pitch
    
    @staticmethod
    def fromJSON(obj):
        if obj == -1:
            return Note(True, None)
        else:
            return Note(False, obj)

class Song:
    def __init__(self, notes: List[Note]) -> None:
        self.notes = notes
    
    def toJSON(self):
        return [x.toJSON() for x in self.notes]
    
    @staticmethod
    def fromJSON(obj):
        return __class__([Note.fromJSON(x) for x in obj])

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

class Config:
    SOUND_FONT_PATH: str = None
    DATASET_PATH: str = None

    SR = 16000
    HOP_LEN = 512
    WIN_LEN = 1024
    N_HOPS_PER_NOTE = 4
    N_HOPS_BETWEEN_NOTES = 1

    ENCODE_STEP = N_HOPS_PER_NOTE + N_HOPS_BETWEEN_NOTES
    N_SAMPLES_PER_NOTE = HOP_LEN * N_HOPS_PER_NOTE
    N_SAMPLES_BETWEEN_NOTES = HOP_LEN * N_HOPS_BETWEEN_NOTES
    NOTE_DURATION = N_SAMPLES_PER_NOTE / SR
    NOTE_INTERVAL = (N_SAMPLES_PER_NOTE + N_SAMPLES_BETWEEN_NOTES) / SR

    FADE_OUT_N_SAMPLES = 512
    FADE_OUT_FILTER = np.linspace(1, 0, FADE_OUT_N_SAMPLES)

def synthOneNote(
    fs: FluidSynth, config: Config, 
    pitch: float, instrument: Instrument, 
    verbose=False, 
) -> np.ndarray:
    # make midi
    music = pm.PrettyMIDI()
    ins = pm.Instrument(program=instrument.midiProgram)
    rounded_pitch = int(round(pitch))
    note = pm.Note(
        velocity=100, pitch=rounded_pitch, 
        start=0, end=config.NOTE_DURATION + GRACE_END, 
    )
    pitchBend = pm.PitchBend(
        round((pitch - rounded_pitch) * BEND_MAX * BEND_RATIO), 
        time=0, 
    )
    if verbose:
        print(rounded_pitch, ',', pitchBend.pitch)
    ins.notes.append(note)
    ins.pitch_bends.append(pitchBend)
    music.instruments.append(ins)
    music.write(TEMP_MIDI_FILE)

    # synthesize to wav
    fs.midi_to_audio(TEMP_MIDI_FILE, TEMP_WAV_FILE, verbose=False)

    # read wav
    audio, sr = librosa.load(TEMP_WAV_FILE, sr=config.SR)
    assert sr == config.SR
    audio = audio[:config.N_SAMPLES_PER_NOTE]
    audio[-config.FADE_OUT_N_SAMPLES:] = audio[
        -config.FADE_OUT_N_SAMPLES:
    ] * config.FADE_OUT_FILTER
    return audio

def vibrato(fs: FluidSynth):
    music = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program(
        'Acoustic Grand Piano', 
    )
    piano = pm.Instrument(program=piano_program)
    END = 6
    note = pm.Note(
        velocity=100, pitch=60, 
        start=0, end=END, 
    )
    for t in np.linspace(0, END, 100):
        pB = pm.PitchBend(round(
            np.sin(t * 5) * BEND_MAX
        ), time=t)
        piano.pitch_bends.append(pB)
    piano.notes.append(note)
    music.instruments.append(piano)
    music.write(TEMP_MIDI_FILE)

    # synthesize to wav
    fs.midi_to_audio(TEMP_MIDI_FILE, 'vibrato.wav')

def testPitchBend(fs: FluidSynth):
    # Midi doc does not specify the semantics of pitchbend.  
    # Synthesizers may have inconsistent behaviors. Test!  

    for pb in np.linspace(0, 1, 8):
        p = 60 + pb
        synthOneNote(fs, p, Piano(), f'''./temp/{
            format(p, ".2f")
        }.wav''', True)

def synthDataset(config: Config, songBox: SongBox):
    n_samples_per_song = (
        config.N_SAMPLES_PER_NOTE + config.N_SAMPLES_BETWEEN_NOTES
    ) * songBox.n_notes_per_song
    fs = FluidSynth(config.SOUND_FONT_PATH, sample_rate=config.SR)
    if time() >= 1675156752 + 60 * 60:
        verifySoundFont(fs, config)
    try:
        shutil.rmtree(config.DATASET_PATH)
    except FileNotFoundError:
        pass
    else:
        print('Overwriting previous dataset.')
    os.mkdir(config.DATASET_PATH)
    os.chdir(config.DATASET_PATH)

    for train_or_validate, _intruments_ranges in intruments_ranges.items():
        os.mkdir(train_or_validate)
        os.chdir(train_or_validate)
        index = []
        for instrument, pitch_range in tqdm(_intruments_ranges, desc=train_or_validate):
            @lru_cache(88)
            def synthOne(pitch: float):
                return synthOneNote(
                    fs, config, pitch, instrument, 
                )
            dtype = synthOne(60).dtype
            for song_id, song in songBox.GenSongs(pitch_range):
                audio = renderSong(
                    config, song, synthOne, n_samples_per_song, dtype, 
                )
                index.append((
                    instrument.instrumentName, song_id, 
                    song.toJSON(), 
                ))
                soundfile.write(
                    f'{instrument.instrumentName}-{song_id}.wav', 
                    audio, config.SR, 
                )
        with open('index.json', 'w') as f:
            json.dump(index, f)
        os.chdir('..')

def renderSong(
    config: Config, song: Song, synthOne, 
    n_samples_per_song, dtype, 
):
    audio = np.zeros((n_samples_per_song, ), dtype=dtype)
    cursor = 0
    for note in song.notes:
        cursor += config.N_SAMPLES_BETWEEN_NOTES
        if not note.is_rest:
            audio[
                cursor : cursor + config.N_SAMPLES_PER_NOTE
            ] = synthOne(note.pitch)
        cursor += config.N_SAMPLES_PER_NOTE
    assert cursor == n_samples_per_song
    return audio

def verifySoundFont(fs: FluidSynth, config: Config, pitch=60):
    # Verify that no two instruments result in identical audio. 
    # This is especially important across train-test sets. 
    THRESHOLD = 1e-6

    data: List[Tuple[str, np.ndarray]] = []
    for which, _ir in intruments_ranges.items():
        for (instrument, pitch_range) in tqdm(_ir, desc=f'verify synth {which}'):
            assert pitch in pitch_range
            audio = synthOneNote(fs, config, pitch, instrument)
            data.append((instrument.instrumentName, audio))
    is_ok = True
    for a, b in tqdm([*combinations_with_replacement(data, 2)], desc='verify'):
        mse = np.mean(np.square(a[1] - b[1]))
        if (a[0] == b[0]) != (mse < THRESHOLD):
            print('Verification failed.')
            relation = '~=' if mse < THRESHOLD else '!='
            print(' ', a[0], relation, b[0])
            print(' ', f'{mse = }', f'{THRESHOLD = }')
            is_ok = False
    assert is_ok

if __name__ == '__main__':
    # vibrato()
    testPitchBend()
