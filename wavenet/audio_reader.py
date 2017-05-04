import fnmatch
import os
import random
import re

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]
        pianist_id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or pianist_id < min_id:
            min_id = pianist_id
        if max_id is None or pianist_id > max_id:
            max_id = pianist_id

    return min_id, max_id


def find_files(directory, pattern='*.wav'):
    """Recursively finds all files matching the pattern."""
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    """Generator that yields audio waveforms from the directory."""

    def randomize_files(fns):
        for _ in fns:
            file_index = random.randint(0, len(fns) - 1)
            yield fns[file_index]

    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        # Normalize audio
        audio = librosa.util.normalize(audio) * 0.8
        # Trim the last 5 seconds to account for music rollout
        audio = audio[:-5 * sample_rate]
        audio = np.reshape(audio, (-1, 1))
        yield audio, filename, category_id


def not_all_have_id(files):
    """ Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id."""
    id_reg_exp = re.compile(FILE_PATTERN)
    for f in files:
        ids = id_reg_exp.findall(f)
        if not ids:
            return True
    return False


class AudioReader(object):
    """Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue."""

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 sample_size=10240,
                 queue_size=500,
                 gc_channels=None,
                 lc_channels=None):
        """
        :param audio_dir: The directory containing WAV files
        :param coord: tf.train.Coordinator 
        :param sample_rate: Sample rate of the audio files
        :param sample_size: Number of timesteps of a cropped sample
        :param queue_size: Size of input pipeline
        :param gc_channels: Global conditioning channels
        :param lc_channels: Local conditioning channels
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.gc_channels = gc_channels
        self.lc_channels = lc_channels
        self.gc_enabled = self.gc_channels or None
        self.lc_enabled = self.lc_channels or None
        self.threads = []
        # Run the Reader on CPU
        with tf.device('/cpu:0'):
            self.queue_audio = tf.placeholder(dtype=tf.float32,
                                              shape=[self.sample_size, 1])

            self.queue_gc = tf.placeholder(dtype=tf.int32,
                                           shape=[])

            self.queue_lc = tf.placeholder(
                dtype=tf.float32,
                # Correspond to librosa.piptrack
                shape=[np.ceil(self.sample_size / 512 + 1).astype(np.int32),
                       self.lc_channels])

            self.queue = tf.RandomShuffleQueue(
                capacity=queue_size,
                min_after_dequeue=5,
                dtypes=[tf.float32, tf.int32, tf.float32],
                shapes=[[self.sample_size, 1],
                        [],
                        [np.ceil(self.sample_size / 512 + 1).astype(np.int32),
                         self.lc_channels]])

            self.enqueue_op = self.queue.enqueue([self.queue_audio,
                                                  self.queue_gc,
                                                  self.queue_lc])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_cardinality += 1
            print("Detected --gc_cardinality={}".format(self.gc_cardinality))
        else:
            self.gc_cardinality = None

    def get_batch(self, batch_size):
        audio_batch, gc_batch, lc_batch = self.queue.dequeue_many(batch_size)
        return {'audio_batch': audio_batch,
                'gc_batch': gc_batch,
                'lc_batch': lc_batch}

    def enqueue(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate)
            for audio, filename, category_id in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                offset = np.random.randint(0, audio.shape[0] - self.sample_size)
                crop = audio[offset:offset + self.sample_size, :]
                assert crop.shape[0] == self.sample_size
                assert crop.shape[1] == audio.shape[1]

                if not self.gc_enabled:
                    category_id = None

                if self.lc_enabled:
                    # Reshape piece into 1-D audio signal
                    crop = np.reshape(crop, (-1,))
                    lc = self.midi_notes_encoding(crop)
                    crop = np.reshape(crop, (-1, 1))
                else:
                    lc = np.empty([1, self.lc_channels], dtype=np.float32)

                sess.run(self.enqueue_op, feed_dict={self.queue_audio: crop,
                                                     self.queue_gc: category_id,
                                                     self.queue_lc: lc})

    @staticmethod
    def midi_notes_encoding(audio):
        """
        Compute frame-based midi encoding of audio
        :param audio: 1-D array of audio time series 
        """
        pitches, magnitudes = librosa.piptrack(audio)
        pitches = np.transpose(pitches)
        magnitudes = np.transpose(magnitudes)
        lc = np.zeros((pitches.shape[0], 88), dtype=np.float32)
        for i in range(pitches.shape[0]):
            # Count non-zero entries of pitches
            nz_count = len(np.nonzero(pitches[i])[0])
            # Keep a maximum of 6 detected pitches
            num_ind_to_keep = min(nz_count, 6)
            ind_of_largest_pitches = np.argpartition(
                magnitudes[i], -num_ind_to_keep)[-num_ind_to_keep:] \
                if num_ind_to_keep != 0 else []
            # Convert the largest pitches to midi notes
            overtone_limit = librosa.midi_to_hz(96)[0]
            ind_of_largest_pitches = filter(
                lambda x: pitches[i, x] <= overtone_limit,
                ind_of_largest_pitches)
            midi_notes = librosa.hz_to_midi(pitches[i, ind_of_largest_pitches])
            midi_notes = midi_notes.round()
            # Normalize magnitudes of pitches
            midi_mags = magnitudes[i, ind_of_largest_pitches] / \
                        np.linalg.norm(magnitudes[i, ind_of_largest_pitches], 1)
            np.put(lc[i], midi_notes.astype(np.int64) - [9], midi_mags)
        return lc
