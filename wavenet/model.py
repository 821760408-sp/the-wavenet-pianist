from __future__ import division

import numpy as np
import tensorflow as tf

from .ops import mu_law_encode, conv1d, shift_right


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        # Make a one-hot encoding as the initial value.
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.Variable(initial_val, name=name)
    else:
        initializer = tf.contrib.layers.xavier_initializer_conv2d()
        return tf.Variable(initializer(shape=shape), name=name)


class WaveNetModel(object):
    """Implements the WaveNet network for generative audio.

    Usage (with the architecture as in the DeepMind paper):
        dilations = [2**i for i in range(N)] * M
        filter_width = 2  # Convolutions just use 2 samples.
        residual_channels = 128
        skip_channels = 256
        net = WaveNetModel(batch_size, dilations, filter_width,
                           residual_channels, dilation_channels,
                           skip_channels)
        loss = net.loss(input_batch)
    """

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 skip_channels,
                 input_channels=1,  # scalar input
                 quantization_channels=2 ** 8,
                 gc_channels=None,
                 gc_cardinality=None,
                 lc_channels=None):
        """Initializes the WaveNet model.

        Args:
            batch_size: How many audio files are supplied per batch
                (recommended: 1).
            dilations: A list with the dilation factor for each layer.
            filter_width: The samples that are included in each convolution,
                after dilating.
            residual_channels: How many filters to learn for the residual.
            skip_channels: How many filters to learn that contribute to the
                quantized softmax output.
            input_channels: How many channels for each input audio sample,
                i.e. the last dimension of input tensor.
            quantization_channels: How many amplitude values to use for audio
                quantization and the corresponding one-hot encoding.
                Default: 256 (8-bit quantization).
            gc_channels: Number of channels in (embedding
                size) of global conditioning vector. None indicates there is
                no global conditioning.
            gc_cardinality: Number of mutually exclusive
                categories to be embedded in global condition embedding. If
                not None, then this implies that global_condition tensor
                specifies an integer selecting which of the N global condition
                categories, where N = global_condition_cardinality. If None,
                then the global_condition tensor is regarded as a vector which
                must have dimension global_condition_channels.
            lc_channels: See audio_reader.py for details.
        """
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.quantization_channels = quantization_channels
        self.skip_channels = skip_channels
        self.gc_channels = gc_channels
        self.gc_cardinality = gc_cardinality
        self.lc_channels = lc_channels

        self.receptive_field = self.calculate_receptive_field(self.filter_width,
                                                              self.dilations)
        self.variables = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        # TODO: why add another (filter_width - 1)?
        receptive_field += filter_width - 1
        return receptive_field

    @staticmethod
    def _enc_upsampling_conv(encoding,
                             audio_length,
                             filter_length=1024,
                             time_stride=512):
        """Upsample local conditioning encoding to match time dim. of audio  
        :param encoding: [mb, timeframe, channels] Local conditionining encoding
        :param audio_length: Length of time dimension of audio 
        :param filter_length: transpose conv. filter length
        :param time_stride: stride along time dimension (upsamp. factor)
        :return: upsampled local conditioning encoding
        """
        with tf.variable_scope('upsampling_conv'):
            batch_size, _, enc_channels = encoding.get_shape().as_list()
            shape = tf.shape(encoding)
            strides = [1, 1, time_stride, 1]
            output_length = (shape[1] - 1) * time_stride + filter_length
            output_shape = tf.stack(
                [batch_size, 1, output_length, enc_channels])

            kernel_shape = [1, filter_length, enc_channels, enc_channels]
            biases_shape = [enc_channels]

            upsamp_weights = tf.get_variable(
                'weights',
                kernel_shape,
                initializer=tf.uniform_unit_scaling_initializer(1.0))
            upsamp_biases = tf.get_variable(
                'biases',
                biases_shape,
                initializer=tf.constant_initializer(0.0))

            encoding = tf.reshape(encoding,
                                  [batch_size, 1, shape[1], enc_channels])
            upsamp_conv = tf.nn.conv2d_transpose(
                encoding,
                upsamp_weights, output_shape, strides, padding='VALID')
            output = tf.nn.bias_add(upsamp_conv, upsamp_biases)

            output = tf.reshape(output,
                                [batch_size, output_length, enc_channels])
            output_sliced = tf.slice(
                output, [0, 0, 0],
                tf.stack([-1, audio_length, -1]))
            output_sliced.set_shape([batch_size, audio_length, enc_channels])
            return output_sliced

    # especially for global conditioning coz it doesn't algin with audio input
    # on the time dimension, and needs broadcasting its value to input;
    # for local conditioning, we've already match their size
    @staticmethod
    def _condition(input_batch, encoding, conditioning='global'):
        """Condition the input on the encoding.
          :param input_batch: [mb, length, channels] float tensor input
          :param encoding: [mb, encoding_length, channels] float tensor encoding
          :param conditioning: 'global' or 'local'
          :return: output after broadcasting the encoding to 
                   x's shape and adding them. 
        """
        mb, length, channels = input_batch.get_shape().as_list()
        enc_mb, enc_length, enc_channels = encoding.get_shape().as_list()
        assert enc_mb == mb
        assert enc_channels == channels
        if conditioning == 'local':
            assert enc_length == length

        if conditioning == 'global':
            encoding = tf.reshape(encoding, [mb, enc_length, 1, channels])
            input_batch = tf.reshape(input_batch,
                                     [mb, enc_length, -1, channels])
            input_batch += encoding
            input_batch = tf.reshape(input_batch, [mb, length, channels])
            input_batch.set_shape([mb, length, channels])
        elif conditioning == 'local':
            input_batch += encoding
        else:
            raise ValueError('`conditioning` must be "global" or "local"')
        return input_batch

    @staticmethod
    def _create_conv_layer(fitler_width, in_channels, out_channels):
        kernel_shape = [fitler_width,
                        in_channels,
                        out_channels]
        biases_shape = [out_channels]
        return {
            'weights': tf.get_variable(
                'weights',
                kernel_shape,
                initializer=tf.uniform_unit_scaling_initializer(1.0)),
            'biases': tf.get_variable(
                'biases',
                biases_shape,
                initializer=tf.constant_initializer(0.0))
        }

    def _create_variables(self):
        """This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function."""

        var = dict()

        with tf.variable_scope('wavenet'):
            if self.gc_cardinality is not None:
                # We only look up the embedding if we are conditioning on a
                # set of mutually-exclusive categories. We can also condition
                # on an already-embedded dense vector, in which case it's
                # given to us and we don't need to do the embedding lookup.
                # Still another alternative is no global condition at all, in
                # which case we also don't do a tf.nn.embedding_lookup.
                with tf.variable_scope('embeddings'):
                    layer = dict()
                    layer['gc_embedding'] = create_embedding_table(
                        'gc_embedding',
                        [self.gc_cardinality, self.gc_channels])
                    var['embeddings'] = layer

            # The first causal layer
            with tf.variable_scope('start_conv'):
                var['start_conv'] = self._create_conv_layer(
                    self.filter_width,
                    self.input_channels,
                    self.residual_channels)

            # Create first skip connections
            with tf.variable_scope('skip_start'):
                var['skip_start'] = self._create_conv_layer(
                    1,
                    self.input_channels,
                    self.skip_channels)

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i + 1)):
                        current = dict()
                        with tf.variable_scope('dilated_conv{}'.format(i + 1)):
                            # combines `filter` and `gate`
                            current['dilated_conv'] = self._create_conv_layer(
                                self.filter_width,
                                self.residual_channels,
                                2 * self.residual_channels)

                        with tf.variable_scope('residual_conv{}'.format(i + 1)):
                            # 1x1 residual conv
                            current['residual_conv'] = self._create_conv_layer(
                                1,
                                self.residual_channels,
                                self.residual_channels)

                        with tf.variable_scope('skip_conn{}'.format(i + 1)):
                            current['skip_conn'] = self._create_conv_layer(
                                1,
                                self.residual_channels,
                                self.skip_channels)

                        if self.gc_channels is not None:
                            with tf.variable_scope('gc_conv{}'.format(i + 1)):
                                # 1x1 learnable linear projection
                                current['gc_conv'] = self._create_conv_layer(
                                    1,
                                    self.gc_channels,
                                    2 * self.residual_channels)

                        if self.lc_channels is not None:
                            with tf.variable_scope('lc_conv{}'.format(i + 1)):
                                # 1x1 conv
                                current['lc_conv'] = self._create_conv_layer(
                                    1,
                                    self.lc_channels,
                                    2 * self.residual_channels)

                        var['dilated_stack'].append(current)

            with tf.variable_scope('postprocessing'):
                # 1x1 conv
                var['postprocessing'] = self._create_conv_layer(
                    1,
                    self.skip_channels,
                    self.skip_channels)

            if self.gc_channels is not None:
                with tf.variable_scope('gc_conv_out'):
                    var['gc_conv_out'] = self._create_conv_layer(
                        1,
                        self.gc_channels,
                        self.skip_channels)

            if self.lc_channels is not None:
                with tf.variable_scope('lc_conv_out'):
                    var['lc_conv_out'] = self._create_conv_layer(
                        1,
                        self.lc_channels,
                        self.skip_channels)

            with tf.variable_scope('logit'):
                # Create weights and biases for computing logits
                var['logit'] = self._create_conv_layer(
                    1,
                    self.skip_channels,
                    self.quantization_channels)

        return var

    def _create_start_layer(self, input_batch):
        """Creates the first causal convolution layer and skip connection.
        :param input_batch: [bs, time, input_channels] tensor input. 
        :return: the output of the causal convolution and skip connection.
        """
        with tf.name_scope('start_conv'):
            weights = self.variables['start_conv']['weights']
            biases = self.variables['start_conv']['biases']
        with tf.name_scope('skip_start'):
            skip_weights = self.variables['skip_start']['weights']
            skip_biases = self.variables['skip_start']['biases']
        return (conv1d(input_batch, weights, biases),
                conv1d(input_batch, skip_weights, skip_biases))

    def _create_dilation_layer(self,
                               input_batch,
                               skip_connection,
                               layer_index,
                               dilation,
                               gc_batch=None,
                               lc_batch=None):
        """Creates a single causal dilated convolution layer.

        Args:
             input_batch: Input to the dilation layer.
             skip_connection: The collection of skip connections so far.
             layer_index: Integer indicating which layer this is.
             dilation: Integer specifying the dilation size.
             gc_batch: Tensor containing the global data upon
                 which the output is to be conditioned upon. Shape:
                 [batch size, 1, channels]. The 1 is for the axis
                 corresponding to time so that the result is broadcast to
                 all time steps.
             lc_batch:

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output. Biases and global conditioning
        are omitted due to the limits of ASCII art.

        """
        variables = self.variables['dilated_stack'][layer_index]

        weights = variables['dilated_conv']['weights']
        biases = variables['dilated_conv']['biases']

        dilation_out = conv1d(input_batch, weights, biases, dilation)

        if lc_batch is not None:
            lc_weights = variables['lc_conv']['weights']
            lc_biases = variables['lc_conv']['biases']

            dilation_out = self._condition(dilation_out,
                                           conv1d(lc_batch, lc_weights,
                                                  lc_biases),
                                           conditioning='local')

        if gc_batch is not None:
            gc_weights = variables['gc_conv']['weights']
            gc_biases = variables['gc_conv']['biases']

            dilation_out = self._condition(dilation_out,
                                           conv1d(gc_batch, gc_weights,
                                                  gc_biases),
                                           conditioning='global')

        assert dilation_out.get_shape().as_list()[2] % 2 == 0
        m = dilation_out.get_shape().as_list()[2] // 2
        do_sigmoid = tf.sigmoid(dilation_out[:, :, :m])  # sigmoid is for gate
        do_tanh = tf.tanh(dilation_out[:, :, m:])  # tanh is for filter
        dilation_out = do_sigmoid * do_tanh

        # The 1x1 conv to produce the residual output
        res_weights = variables['residual_conv']['weights']
        res_biases = variables['residual_conv']['biases']
        input_batch += conv1d(dilation_out, res_weights, res_biases)

        # The 1x1 conv to produce the skip output
        skip_weights = variables['skip_conn']['weights']
        skip_biases = variables['skip_conn']['biases']
        skip_connection += conv1d(dilation_out, skip_weights, skip_biases)

        return input_batch, skip_connection

    def _embed_gc(self, gc):
        """Returns embedding for global condition.
        :param gc: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding.
        :return: Embedding or None
        """
        embedding = None

        if self.gc_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table, gc)

        if embedding is not None:
            embedding = tf.reshape(embedding,
                                   [self.batch_size, 1, self.gc_channels])

        return embedding

    def _create_network(self, input_batch, gc_batch=None, lc_batch=None):
        """Construct the WaveNet network.
        :param input_batch: The [nb, time, channels] right-shifted input tensor.
        :param gc_batch: 
        :param lc_batch: 
        :return: The [time, quantizations] logits computed by the network.
        """
        input_batch, skip_connection = self._create_start_layer(input_batch)

        if gc_batch is not None:
            gc_batch = self._embed_gc(gc_batch)

        if lc_batch is not None:
            # upsample local conditioning
            _, input_length, _ = input_batch.get_shape().as_list()
            lc_batch = self._enc_upsampling_conv(lc_batch, input_length)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index + 1)):
                    (input_batch, skip_connection) = self._create_dilation_layer(
                        input_batch,
                        skip_connection,
                        layer_index,
                        dilation,
                        gc_batch,
                        lc_batch)

        # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
        # postprocess the output.
        with tf.name_scope('postprocessing'):
            skip_connection = tf.nn.relu(skip_connection)

            pp_weights = self.variables['postprocessing']['weights']
            pp_biases = self.variables['postprocessing']['biases']
            skip_connection = conv1d(skip_connection, pp_weights, pp_biases)

        if lc_batch is not None:
            lc_weights = self.variables['lc_conv_out']['weights']
            lc_biases = self.variables['lc_conv_out']['biases']
            skip_connection = self._condition(
                skip_connection,
                conv1d(lc_batch, lc_weights, lc_biases),
                conditioning='local')

        if gc_batch is not None:
            gc_weights = self.variables['gc_conv_out']['weights']
            gc_biases = self.variables['gc_conv_out']['biases']
            skip_connection = self._condition(
                skip_connection,
                conv1d(gc_batch, gc_weights, gc_biases),
                conditioning='global')

        with tf.name_scope('logit'):
            skip_connection = tf.nn.relu(skip_connection)

            logit_weights = self.variables['logit']['weights']
            logit_biases = self.variables['logit']['biases']
            logits = conv1d(skip_connection, logit_weights, logit_biases)
            logits = tf.reshape(logits, [-1, 256])

        return logits

    def loss(self, input_batch, gc_batch=None, lc_batch=None, name='wavenet'):
        """Creates a WaveNet network and returns the autoencoding loss.
        :param input_batch: The [nb, time, 1] audio input tensor 
        :param gc_batch: 
        :param lc_batch: 
        :param name:
        :return: Prediction, loss, and quantized input
        """
        with tf.name_scope(name):
            input_quantized = mu_law_encode(input_batch,
                                            self.quantization_channels)
            # values of input_quantized in [-128., 128.]
            input_scaled = tf.cast(input_quantized, tf.float32) / 128.0
            assert len(input_scaled.get_shape()) == 3

            input_batch = shift_right(input_scaled)

            logits = self._create_network(input_batch, gc_batch, lc_batch)
            probs = tf.nn.softmax(logits, name='softmax')
            input_indices = tf.cast(tf.reshape(input_quantized, [-1]),
                                    tf.int32) + 128

            with tf.name_scope('loss'):
                loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=input_indices, name='nll'),
                    0,
                    name='loss')
                tf.summary.scalar('train_loss', loss)

            return {
                'predictions': probs,
                'loss': loss,
                'eval': {
                    'nll': loss
                },
                'quantized_input': input_quantized
            }

    @staticmethod
    def _generator_conv(input_batch, state_batch, weights, biases):
        """Perform convolution for a single convolutional processing step."""
        # TODO generalize to filter_width > 2
        past_weights = weights[0, :, :]  # now a rank 2 tensor
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + \
                 tf.matmul(input_batch, curr_weights)
        output = tf.nn.bias_add(output, biases)
        return output

    def _generator_start_layer(self, input_batch, state_batch):
        with tf.name_scope('start_conv'):
            weights = self.variables['start_conv']['weights']
            biases = self.variables['start_conv']['biases']
        with tf.name_scope('skip_start'):
            skip_weights = self.variables['skip_start']['weights']
            skip_biases = self.variables['skip_start']['biases']
            skip_weights = skip_weights[0, :, :]
            skip_conn = tf.matmul(input_batch, skip_weights)
            skip_conn = tf.nn.bias_add(skip_conn, skip_biases)
        return (self._generator_conv(input_batch, state_batch, weights, biases),
                skip_conn)

    def _generator_dilation_layer(self,
                                  input_batch,
                                  state_batch,
                                  layer_index,
                                  dilation,
                                  gc_batch,
                                  lc_batch):
        variables = self.variables['dilated_stack'][layer_index]

        weights = variables['dilated_conv']['weights']
        biases = variables['dilated_conv']['biases']
        dilation_out = self._generator_conv(input_batch, state_batch,
                                            weights, biases)

        if lc_batch is not None:
            lc_weights = variables['lc_conv']['weights']
            lc_biases = variables['lc_conv']['biases']
            lc_weights = lc_weights[0, :, :]
            dilation_out += tf.matmul(lc_batch, lc_weights)
            dilation_out = tf.nn.bias_add(dilation_out, lc_biases)

        if gc_batch is not None:
            gc_weights = variables['gc_conv']['weights']
            gc_biases = variables['gc_conv']['biases']
            gc_weights = gc_weights[0, :, :]

            dilation_out += tf.matmul(gc_batch, gc_weights)
            dilation_out = tf.nn.bias_add(dilation_out, gc_biases)

        assert dilation_out.get_shape().as_list()[1] % 2 == 0
        m = dilation_out.get_shape().as_list()[1] // 2
        do_sigmoid = tf.sigmoid(dilation_out[:, :m])  # sigmoid is for gate
        do_tanh = tf.tanh(dilation_out[:, m:])  # tanh is for filter
        dilation_out = do_sigmoid * do_tanh

        # The 1x1 conv to produce the residual output
        res_weights = variables['residual_conv']['weights']
        res_biases = variables['residual_conv']['biases']
        res_weights = res_weights[0, :, :]

        input_batch += tf.matmul(dilation_out, res_weights)
        input_batch = tf.nn.bias_add(input_batch, res_biases)

        # The 1x1 conv to produce the skip output
        skip_weights = variables['skip_conn']['weights']
        skip_biases = variables['skip_conn']['biases']
        skip_weights = skip_weights[0, :, :]
        # skip_biases = skip_biases[0, :, :]
        skip_connection = tf.matmul(dilation_out, skip_weights)
        skip_connection = tf.nn.bias_add(skip_connection, skip_biases)

        return skip_connection, input_batch

    def _create_generator(self, audio_batch, gc_batch=None, lc_batch=None):
        """Construct an efficient incremental generator."""
        init_ops = []
        push_ops = []
        skip_connection = []
        current_layer = audio_batch

        if gc_batch is not None:
            gc_batch = self._embed_gc(gc_batch)
            gc_batch = tf.reshape(gc_batch, shape=(-1, self.gc_channels))

        if lc_batch is not None:
            lc_batch = tf.reshape(lc_batch, shape=(-1, self.lc_channels))

        q = tf.FIFOQueue(
            capacity=1,
            dtypes=[tf.float32],
            shapes=[(self.batch_size, self.input_channels)])

        init = q.enqueue_many(
            tf.zeros([1, self.batch_size, self.input_channels], tf.float32))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

        current_layer, skip_conn = self._generator_start_layer(current_layer,
                                                               current_state)

        skip_connection.append(skip_conn)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index + 1)):
                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=[tf.float32],
                        shapes=[(self.batch_size, self.residual_channels)])
                    init = q.enqueue_many(tf.zeros([dilation,
                                                    self.batch_size,
                                                    self.residual_channels]))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    skip_conn, current_layer = self._generator_dilation_layer(
                        current_layer, current_state,
                        layer_index, dilation,
                        gc_batch, lc_batch)
                    skip_connection.append(skip_conn)

        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            skip_connection = tf.add_n(skip_connection)
            skip_connection = tf.nn.relu(skip_connection)
            pp_weights = self.variables['postprocessing']['weights']
            pp_weights = pp_weights[0, :, :]
            pp_biases = self.variables['postprocessing']['biases']
            skip_connection = tf.matmul(skip_connection, pp_weights)
            skip_connection = tf.nn.bias_add(skip_connection, pp_biases)

            if lc_batch is not None:
                lc_weights = self.variables['lc_conv_out']['weights']
                lc_biases = self.variables['lc_conv_out']['biases']
                lc_weights = lc_weights[0, :, :]
                skip_connection += tf.matmul(lc_batch, lc_weights)
                skip_connection = tf.nn.bias_add(skip_connection, lc_biases)

            if gc_batch is not None:
                gc_weights = self.variables['gc_conv_out']['weights']
                gc_biases = self.variables['gc_conv_out']['biases']
                gc_weights = gc_weights[0, :, :]
                skip_connection += tf.matmul(gc_batch, gc_weights)
                skip_connection = tf.nn.bias_add(skip_connection, gc_biases)

        with tf.name_scope('logit'):
            skip_connection = tf.nn.relu(skip_connection)

            logit_weights = self.variables['logit']['weights']
            logit_biases = self.variables['logit']['biases']
            logit_weights = logit_weights[0, :, :]
            logits = tf.matmul(skip_connection, logit_weights)
            logits = tf.nn.bias_add(logits, logit_biases)

        return logits

    def predict_proba_incremental(self,
                                  waveform,
                                  global_condition=None,
                                  local_condition=None,
                                  name='wavenet_predict'):
        """Computes the probability distribution of the next sample incrementally,
        based on a single sample and all previously passed samples."""
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        with tf.name_scope(name):
            waveform = tf.reshape(waveform, [-1, self.input_channels])
            # `waveform` now [-128, 128), needs to be [-1, 1)
            waveform = tf.cast(waveform, tf.float32) / 128.0

            raw_output = self._create_generator(waveform,
                                                global_condition,
                                                local_condition)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            return tf.reshape(last, [-1])
