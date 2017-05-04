"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time
import threading

import tensorflow as tf

from wavenet import WaveNetModel, AudioReader, optimizer_factory

BATCH_SIZE = 1
DATA_DIRECTORY = '/scratch/yg1349/solo-piano-classical-corpus'
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 250
NUM_STEPS = int(3e5)
LEARNING_RATE_TRANSITION_STEPS = [
    0,
    10000,
    30000,
    60000,
    90000,
    120000,
    150000,
    180000,
    210000,
    240000
]
LEARNING_RATE_SCHEDULE = [
    1e-3,
    6e-4,
    4e-4,
    1e-4,
    8e-5,
    5e-5,
    2e-5,
    9e-6,
    6e-6,
    3e-6
]
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
MOMENTUM = 0.9


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: '
                             + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the training corpus.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                             'information for TensorBoard. '
                             'If the model already exists, it will restore '
                             'the state and will continue training. '
                             'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                             'output and generated model. These are stored '
                             'under the dated subdirectory of --logdir_root. '
                             'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                             'This creates the new model under the dated directory '
                             'in --logdir_root. '
                             'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. '
                             'Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: '
                             + str(NUM_STEPS) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: '
                             + WAVENET_PARAMS + '.')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. '
                             'Default: adam.')
    parser.add_argument('--momentum', type=float, default=MOMENTUM,
                        help='Specify the momentum to be '
                             'used by sgd or rmsprop optimizer. Ignored by the '
                             'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--gc_channels', type=int, default=None,
                        help='Number of global condition channels. '
                             'Default: None. Expecting: Int')
    parser.add_argument('--lc_channels', type=int, default=None,
                        help='Number of local condition channels. '
                             'Default: None. Expecting: Int')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Saving checkpoint done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Restoring checkpoint done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))
    else:
        logdir = os.path.join(LOGDIR_ROOT, logdir, STARTED_DATESTRING)

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Set up session
    sess = tf.Session()

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw waveform from corpus.
    with tf.name_scope('create_inputs'):
        reader = AudioReader(
            args.data_dir,
            coord,
            sample_rate=wavenet_params['sample_rate'],
            gc_channels=args.gc_channels,
            lc_channels=args.lc_channels)
        inputs_dict = reader.get_batch(args.batch_size)

    # Create network.
    audio_batch = inputs_dict['audio_batch']
    gc_batch = inputs_dict['gc_batch']
    lc_batch = inputs_dict['lc_batch']
    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        skip_channels=wavenet_params["skip_channels"],
        input_channels=audio_batch.get_shape().as_list()[2],
        quantization_channels=wavenet_params["quantization_channels"],
        gc_channels=args.gc_channels,
        gc_cardinality=reader.gc_cardinality,
        lc_channels=args.lc_channels)

    output_dict = net.loss(input_batch=audio_batch,
                           gc_batch=gc_batch \
                               if args.gc_channels is not None else None,
                           lc_batch=lc_batch \
                               if args.lc_channels is not None else None)

    loss = output_dict['loss']
    tf.summary.scalar('train_loss', loss)

    global_step = tf.get_variable(
        "global_step", [],
        tf.int32,
        initializer=tf.constant_initializer(0),
        trainable=False)

    assert len(LEARNING_RATE_SCHEDULE) == len(LEARNING_RATE_TRANSITION_STEPS)
    lr = tf.constant(LEARNING_RATE_SCHEDULE[0])
    for s, v in zip(LEARNING_RATE_TRANSITION_STEPS, LEARNING_RATE_SCHEDULE):
        lr = tf.cond(
            tf.less(global_step, s), lambda: lr, lambda: tf.constant(v))
    tf.summary.scalar("learning_rate", lr)

    optimizer = optimizer_factory[args.optimizer](learning_rate=lr,
                                                  momentum=args.momentum)
    train_op = optimizer.minimize(loss,
                                  global_step=global_step,
                                  name='train')

    # Set up logging for TensorBoard.
    writer = tf.summary.FileWriter(logdir)
    writer.add_graph(tf.get_default_graph())
    summary_op = tf.summary.merge_all()

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.trainable_variables())

    try:
        saved_global_step = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # Start enqueue op
    enqueue_thread = threading.Thread(target=reader.enqueue, args=[sess])
    enqueue_thread.daemon = True
    enqueue_thread.start()

    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, args.num_steps):
            start_time = time.time()

            summary, loss_value, _ = sess.run([summary_op, loss, train_op])
            writer.add_summary(summary, step)

            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(step, loss_value, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
