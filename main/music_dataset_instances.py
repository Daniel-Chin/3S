from os import path
from typing import *

from music_dataset_shared import *

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

class IonianScales_fr3gm:
    _SongBox = IonianScalesSongBox
    config = Config()
    config.DATASET_PATH = path.abspath(
        '../datasets/ionian_scales_fr3gm', 
    )
    config.SOUND_FONT_PATH = path.abspath(
        '../sound_fonts/FluidR3_GM/FluidR3_GM.sf2', 
    )

if __name__ == '__main__':
    x = IonianScales_fr3gm
    synthDataset(x.config, x._SongBox())
