import os
from os import path
import shutil
import json
from typing import *
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

from shared import *
from intruments_and_ranges import intruments_ranges
from dataset_definitions import (
    MUSIC, 
    MusicDatasetConfig as Config, DatasetDefinition, 
)

BEND_RATIO = .5  # When MIDI specification is incomplete...
BEND_MAX = 8191
GRACE_END = .1

INDEX_FILENAME = 'index.json'
TEMP_MIDI_FILE = path.abspath('./temp/temp.mid')
TEMP_WAV_FILE  = path.abspath('./temp/temp.wav')
os.makedirs('temp', exist_ok=True)

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
        start=0, end=config.note_duration + GRACE_END, 
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
    audio, sr = librosa.load(TEMP_WAV_FILE, sr=config.sr)
    assert sr == config.sr
    audio = audio[:config.n_samples_per_note]
    audio[-config.fade_out_n_samples:] = audio[
        -config.fade_out_n_samples:
    ] * config.fade_out_filter
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

def synthDataset(datasetDefinition: DatasetDefinition):
    assert datasetDefinition.data_modality is MUSIC
    config = datasetDefinition.musicDatasetConfig
    songBox = datasetDefinition.songBox
    n_samples_per_song = (
        config.n_samples_per_note + config.n_samples_between_notes
    ) * songBox.n_notes_per_song
    fs = FluidSynth(config.sound_font_path, sample_rate=config.sr)
    if time() >= 1675156752 + 60 * 60:
        verifySoundFont(fs, config)
    try:
        shutil.rmtree(datasetDefinition.dataset_path)
    except FileNotFoundError:
        pass
    else:
        print('Overwriting previous dataset.')
    os.mkdir(datasetDefinition.dataset_path)
    os.chdir(datasetDefinition.dataset_path)

    for train_or_validate, _intruments_ranges in intruments_ranges.items():
        os.mkdir(train_or_validate)
        os.chdir(train_or_validate)
        index: List[Tuple[str, int, Any]] = []
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
                    instrument.instrumentName, 
                    song_id, 
                    song.toJSON(), 
                ))
                soundfile.write(
                    f'{instrument.instrumentName}-{song_id}.wav', 
                    audio, config.sr, 
                )
        with open(INDEX_FILENAME, 'w') as f:
            json.dump(index, f)
        os.chdir('..')

def renderSong(
    config: Config, song: Song, synthOne, 
    n_samples_per_song, dtype, 
):
    audio = np.zeros((n_samples_per_song, ), dtype=dtype)
    cursor = 0
    for note in song.notes:
        cursor += config.n_samples_between_notes
        if not note.is_rest:
            audio[
                cursor : cursor + config.n_samples_per_note
            ] = synthOne(note.pitch)
        cursor += config.n_samples_per_note
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
    from dataset_definitions import ionianScales_fr3gm
    # vibrato()
    # testPitchBend()
    synthDataset(ionianScales_fr3gm)
