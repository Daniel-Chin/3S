from typing import *

from music_dataset_shared import SongBox, Note, Config

class IonianScalesSongBox(SongBox):
    IONIAN_SCALE = [0, 2, 4, 5, 7, 9, 11, 12, 11, 9, 7, 5, 4, 2, 0]

    def __init__(self) -> None:
        self.name = 'ionian_scales'
        self.n_notes_per_song = 15

        self.scale_max = max(self.IONIAN_SCALE)
        assert min(self.IONIAN_SCALE) == self.IONIAN_SCALE[0] == 0

    def GenSongs(self, pitch_range: range) -> Generator[List[Tuple[int, Note]]]:
        start_pitch = pitch_range.start
        while start_pitch + self.scale_max in pitch_range:
            song = []
            for d_pitch in self.IONIAN_SCALE:
                pitch = start_pitch + d_pitch
                song.append(Note(False, pitch))
            yield start_pitch, song
            start_pitch += 1

class IonianScales:
    _SongBox = IonianScalesSongBox
    config = Config()
