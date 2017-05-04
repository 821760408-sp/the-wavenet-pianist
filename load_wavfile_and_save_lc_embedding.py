import argparse
import os

import librosa
import numpy as np

from wavenet.audio_reader import AudioReader

def load_wav(wavfile, sr, mono=True):
    audio, _ = librosa.load(wavfile, sr=sr, mono=mono)
    # Normalize audio
    audio = librosa.util.normalize(audio) * 0.8
    lc = AudioReader.midi_notes_encoding(audio)

    fn = os.path.abspath(wavfile).strip('.wav')
    fn = "{}_lc_embedding.npy".format(fn)
    with open(fn, 'w') as f:
        np.save(f, lc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='save embedding for wav..')
    parser.add_argument('--wavfile', type=str, default=None, required=True)
    parser.add_argument('--sr', type=int, default=22050)
    args = parser.parse_args()
    load_wav(args.wavfile, args.sr)
