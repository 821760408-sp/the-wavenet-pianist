from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import json
import os

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode

SECS = 10
N_SAMPLES = 22050 * SECS
TEMPERATURE = 1.0
LOGDIR = './logdir'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Which model checkpoint to generate from')
    parser.add_argument(
        '--n_samples',
        type=int,
        default=N_SAMPLES,
        help='How many waveform samples to generate')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
             'information for TensorBoard.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='JSON file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--gc_channels',
        type=int,
        default=None,
        help='Number of global condition embedding channels. Omit if no '
             'global conditioning.')
    parser.add_argument(
        '--gc_cardinality',
        type=int,
        default=None,
        help='Number of categories upon which we globally condition.')
    parser.add_argument(
        '--gc_id',
        type=int,
        default=None,
        help='ID of category to generate, if globally conditioned.')
    parser.add_argument(
        '--lc_channels',
        type=int,
        default=None,
        help='Number of local condition embedding channels. Omit if no '
             'lc_embedding.')
    parser.add_argument(
        '--lc_embedding',
        type=str,
        default=None,
        help='The .npy file of pre-saved local condition embedding.')
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not "
                             "specified. Use --gc_cardinality=377 for full "
                             "VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was "
                             "not specified. Use --gc_id to specify global "
                             "condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]


def load_lc_embedding(lc_embedding):
    with open(lc_embedding, 'r') as f:
        return np.load(f)


def main():
    args = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(args.logdir, 'generate', started_datestring)
    with open(args.wavenet_params, 'r') as config_file:
        wavenet_params = json.load(config_file)

    sess = tf.Session(
        # config=tf.ConfigProto(device_count={'GPU': 0})
                      )

    # Build the WaveNet model
    net = WaveNetModel(
        batch_size=1,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        gc_channels=args.gc_channels,
        gc_cardinality=args.gc_cardinality,
        lc_channels=args.lc_channels)

    # Create placeholders
    # Default to fast generation
    samples = tf.placeholder(tf.int32)
    lc = tf.placeholder(tf.float32) if args.lc_embedding else None
    gc = args.gc_id or None

    # TODO: right now we pre-calculated lc embeddings of the same length
    # as the audio we'd like to generate so they're naturally algined.
    # Add function to load a length of `args.n_samples` of embeddings
    # from pre-calculated (full-length) embeddings.
    if args.lc_embedding is not None:
        lc_embedding = load_lc_embedding(args.lc_embedding)
        lc_embedding = tf.convert_to_tensor(lc_embedding)
        lc_embedding = tf.reshape(lc_embedding, [1, -1, args.lc_channels])
        lc_embedding = net._enc_upsampling_conv(lc_embedding, args.n_samples)
        lc_embedding = tf.reshape(lc_embedding, [-1, args.lc_channels])

    next_sample = net.predict_proba_incremental(samples, gc, lc)

    sess.run(tf.global_variables_initializer())
    sess.run(net.init_ops)
    # Group the ops we need to run
    output_ops = [next_sample]
    output_ops.extend(net.push_ops)
    # Convert mu-law encoded samples back to (-1, 1) of R
    QUANTIZATION_CHANNELS = wavenet_params['quantization_channels']
    decode = mu_law_decode(samples, QUANTIZATION_CHANNELS)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    ckpt = tf.train.get_checkpoint_state(args.checkpoint)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restoring model from {}'.format(ckpt.model_checkpoint_path))

    if args.wav_seed:
        seed = create_seed(args.wav_seed,
                           wavenet_params['sample_rate'],
                           QUANTIZATION_CHANNELS,
                           net.receptive_field)
        waveform = sess.run(seed).tolist()
    else:
        # Silence with a single random sample at the end.
        waveform = [0] * (net.receptive_field - 1)
        waveform.append(np.random.randint(-QUANTIZATION_CHANNELS // 2,
                                          QUANTIZATION_CHANNELS // 2))

    if args.lc_embedding is not None:
        lc_embedding = sess.run(lc_embedding)

    if args.wav_seed:
        # When using the incremental generation, we need to
        # feed in all priming samples one by one before starting the
        # actual generation.
        # TODO This could be done much more efficiently by passing the waveform
        # to the incremental generator as an optional argument, which would be
        # used to fill the queues initially.

        print('Priming generation...')
        for i, x in enumerate(waveform[-net.receptive_field: -1]):
            if i % 100 == 0:
                print('Priming sample {}'.format(i))
            lc_ = lc_embedding[i, :]
            sess.run(output_ops, feed_dict={samples: x, lc: lc_})
        print('Done.')

    last_sample_timestamp = datetime.now()
    lc_ = None
    import sys
    for step in range(args.n_samples):
        if step % 1000 == 0:
            print("Generating {} of {}.".format(step, args.n_samples))
            sys.stdout.flush()

        window = waveform[-1]

        if args.lc_embedding is not None:
            lc_ = lc_embedding[step, :]

        # Run the WaveNet to predict the next sample.
        feed_dict = {samples: window}
        if lc_ is not None:
            feed_dict[lc] = lc_
        results = sess.run(output_ops, feed_dict=feed_dict)

        pred = results[0]

        # Scale prediction distribution using temperature.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(pred) / args.temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')

        # Prediction distribution at temperature=1.0 should be unchanged
        # after scaling.
        if args.temperature == 1.0:
            np.testing.assert_allclose(
                pred, scaled_prediction, atol=1e-5,
                err_msg='Prediction scaling at temperature=1.0 '
                        'is not working as intended.')

        sample = np.random.choice(
            np.arange(-QUANTIZATION_CHANNELS // 2, QUANTIZATION_CHANNELS // 2),
            p=scaled_prediction)
        waveform.append(sample)

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                (step + 1) % args.save_every == 0):
            out = sess.run(decode, feed_dict={samples: waveform})
            write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as an audio summary.
    datestring = str(datetime.now()).replace(' ', 'T')
    writer = tf.summary.FileWriter(logdir)
    tf.summary.audio('generated', decode, wavenet_params['sample_rate'])
    summaries = tf.summary.merge_all()
    summary_out = sess.run(summaries,
                           feed_dict={samples: np.reshape(waveform, [-1, 1])})
    writer.add_summary(summary_out)

    # Save the result as a wav file.
    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_wav(out, wavenet_params['sample_rate'], args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()
