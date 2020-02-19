# Created by ay27 at 2017/10/6
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

from .BT_mul_Keras import *


class BT_RNN(Recurrent):
    def __init__(self,
                 bt_input_shape, bt_output_shape, core_ranks, block_ranks,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 debug=False,
                 init_seed=11111986,
                 **kwargs):
        super(BT_RNN, self).__init__(**kwargs)

        self.bt_input_shape = np.array(bt_input_shape)
        self.bt_output_shape = np.array(bt_output_shape)
        self.core_ranks = np.array(core_ranks)
        self.block_ranks = int(block_ranks)
        self.debug = debug

        self.units = np.prod(self.bt_output_shape)
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = InputSpec(shape=(None, self.units))
        self.states = None
        self.kernel = None
        self.recurrent_kernel = None
        self.cores = [None]
        self.factors = [None]
        self.bias = None

        self.debug = debug
        self.init_seed = init_seed

        self.input_dim = np.prod(self.bt_input_shape)
        self.params_original = np.prod(self.bt_input_shape) * np.prod(self.bt_output_shape)
        self.params_bt = self.block_ranks * \
                         (np.sum(self.bt_input_shape * self.bt_output_shape * self.core_ranks) + np.prod(
                             self.core_ranks))
        self.batch_size = None

        # reported compress ratio in input->hidden weight
        self.compress_ratio = self.params_original / self.params_bt

        if self.debug:
            print('bt_input_shape = ' + str(self.bt_input_shape))
            print('bt_output_shape = ' + str(self.bt_output_shape))
            print('core_ranks = ' + str(self.core_ranks))
            print('block_ranks = ' + str(self.block_ranks))
            print('compress_ratio = ' + str(self.compress_ratio))
        assert len(self.core_ranks.shape) == len(self.bt_input_shape.shape) == len(self.bt_output_shape.shape)

    def build(self, input_shape):
        # input shape: `(batch, time (padded with zeros), input_dim)`
        # input_shape is a tuple
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        assert len(input_shape) == 3
        assert input_shape[2] == self.input_dim

        self.batch_size = input_shape[0] if self.stateful else None
        self.input_spec[0] = InputSpec(shape=(self.batch_size, None, self.input_dim))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        ################################################################################################################
        # input -> hidden state
        # the kernel layout is : [[core, factor0, factor1, factor2, ...],
        #                         [core, factor0, factor1, factor2, ...],
        #                         ...]
        self.kernel = self.add_weight((self.params_bt,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.cores, self.factors = split_kernel_into_core_and_factors(self.kernel,
                                                                      self.bt_input_shape, self.bt_output_shape,
                                                                      self.core_ranks, self.block_ranks)

        ################################################################################################################

        # hidden -> hidden
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def preprocess_input(self, inputs, training=None):
        # input shape: `(batch, time (padded with zeros), input_dim)`
        return inputs

    def step(self, inputs, states):
        # inputs shape: [batch, input_dim]
        if 0. < self.dropout < 1.:
            inputs = inputs * states[1]

        ################################################################################################################
        # NOTE: we now just substitute the `W_{xh}`
        if len(self.core_ranks) == 2:
            h = BT_mul2(inputs, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 3:
            h = BT_mul3(inputs, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 4:
            h = BT_mul4(inputs, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 5:
            h = BT_mul5(inputs, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        else:
            h = None
            raise ValueError('error in len(core_ranks)')

        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        ################################################################################################################

        prev_output = states[0]
        if 0. < self.recurrent_dropout < 1.:
            prev_output *= states[2]
        output = h + K.dot(prev_output, self.recurrent_kernel)
        if self.activation is not None:
            output = self.activation(output)

        # Properly set learning phase on output tensor.
        if 0. < self.dropout + self.recurrent_dropout:
            output._uses_learning_phase = True
        return output, [output]

    def get_constants(self, inputs, training=None):
        # this is totally same as the Keras API
        constants = []
        if self.implementation != 0 and 0. < self.dropout < 1.:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = K.in_train_phase(dropped_inputs,
                                       ones,
                                       training=training)
            constants.append(dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))

        if 0. < self.recurrent_dropout < 1.:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = K.in_train_phase(dropped_inputs,
                                           ones,
                                           training=training)
            constants.append(rec_dp_mask)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(BT_RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BT_GRU(Recurrent):
    def __init__(self,
                 bt_input_shape, bt_output_shape, core_ranks, block_ranks,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 debug=False,
                 init_seed=11111986,
                 **kwargs):
        super(BT_GRU, self).__init__(**kwargs)

        self.bt_input_shape = np.array(bt_input_shape)

        self.bt_output_shape = np.array(bt_output_shape)

        self.core_ranks = np.array(core_ranks)
        self.block_ranks = int(block_ranks)
        self.debug = debug

        self.units = np.prod(self.bt_output_shape)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = InputSpec(shape=(None, self.units))
        self.states = None
        self.kernel = None
        self.recurrent_kernel = None
        self.cores = [None]
        self.factors = [None]
        self.bias = None

        self.debug = debug
        self.init_seed = init_seed

        # store r, z, h
        self.bt_output_shape[0] *= 3

        self.input_dim = np.prod(self.bt_input_shape)
        self.params_original = np.prod(self.bt_input_shape) * np.prod(self.bt_output_shape)
        self.params_bt = self.block_ranks * \
                         (np.sum(self.bt_input_shape * self.bt_output_shape * self.core_ranks) + np.prod(
                             self.core_ranks))
        self.batch_size = None

        # reported compress ratio in input->hidden weight
        self.compress_ratio = self.params_original / self.params_bt

        if self.debug:
            print('bt_input_shape = ' + str(self.bt_input_shape))
            print('bt_output_shape = ' + str(self.bt_output_shape))
            print('core_ranks = ' + str(self.core_ranks))
            print('block_ranks = ' + str(self.block_ranks))
            print('compress_ratio = ' + str(self.compress_ratio))
        assert len(self.core_ranks.shape) == len(self.bt_input_shape.shape) == len(self.bt_output_shape.shape)

    def build(self, input_shape):
        # input shape: `(batch, time (padded with zeros), input_dim)`
        # input_shape is a tuple
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        assert len(input_shape) == 3
        assert input_shape[2] == self.input_dim

        self.batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(self.batch_size, None, self.input_dim))

        self.states = [None]
        if self.stateful:
            self.reset_states()

        ################################################################################################################
        # input -> hidden state
        self.kernel = self.add_weight((self.params_bt,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.cores, self.factors = split_kernel_into_core_and_factors(self.kernel,
                                                                      self.bt_input_shape, self.bt_output_shape,
                                                                      self.core_ranks, self.block_ranks)
        ################################################################################################################

        # hidden -> hidden
        # store r, z, h
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight((np.prod(self.bt_output_shape),),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def preprocess_input(self, x, training=None):
        return x

    def get_constants(self, inputs, training=None):
        # this is totally same as the Keras API
        constants = [[K.cast_to_floatx(1.) for _ in range(3)]]

        if 0. < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(3)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])
        return constants

    def step(self, x, states):
        h_tm1 = states[0]  # previous memory
        dp_mask = states[1]  # dropout matrices for recurrent units
        rec_dp_mask = states[2]

        x1 = x * dp_mask[0]

        ################################################################################################################
        # NOTE: we now just substitute the `W_{xh}`
        if len(self.core_ranks) == 2:
            matrix_x = BT_mul2(x1, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 3:
            matrix_x = BT_mul3(x1, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 4:
            matrix_x = BT_mul4(x1, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 5:
            matrix_x = BT_mul5(x1, self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        else:
            matrix_x = None
            raise ValueError('error in len(core_ranks)')

        # following is same as Keras API
        if self.use_bias:
            matrix_x = K.bias_add(matrix_x, self.bias)
        matrix_inner = K.dot(h_tm1 * rec_dp_mask[0],
                             self.recurrent_kernel[:, :2 * self.units])
        x_z = matrix_x[:, :self.units]
        x_r = matrix_x[:, self.units: 2 * self.units]
        recurrent_z = matrix_inner[:, :self.units]
        recurrent_r = matrix_inner[:, self.units: 2 * self.units]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        x_h = matrix_x[:, 2 * self.units:]
        recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
                            self.recurrent_kernel[:, 2 * self.units:])
        hh = self.activation(x_h + recurrent_h)

        h = z * h_tm1 + (1 - z) * hh
        if 0. < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(BT_GRU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BT_LSTM(Recurrent):
    def __init__(self,
                 bt_input_shape, bt_output_shape, core_ranks, block_ranks,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 debug=False,
                 init_seed=11111986,
                 **kwargs):
        super(BT_LSTM, self).__init__(**kwargs)

        self.bt_input_shape = np.array(bt_input_shape)

        self.bt_output_shape = np.array(bt_output_shape)

        self.core_ranks = np.array(core_ranks)
        self.block_ranks = int(block_ranks)
        self.debug = debug

        self.units = np.prod(self.bt_output_shape)
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_spec = [InputSpec(shape=(None, self.units)),
                           InputSpec(shape=(None, self.units))]
        self.states = None
        self.kernel = None
        self.recurrent_kernel = None
        self.cores = [None]
        self.factors = [None]
        self.bias = None

        self.debug = debug
        self.init_seed = init_seed

        # store i, f, c, o
        if not self.go_backwards:
            self.bt_output_shape[0] *= 4
        else:
            self.units = int(self.units / 4)

        self.input_dim = np.prod(self.bt_input_shape)
        self.params_original = np.prod(self.bt_input_shape) * np.prod(self.bt_output_shape)
        self.params_bt = self.block_ranks * \
                         (np.sum(self.bt_input_shape * self.bt_output_shape * self.core_ranks) + np.prod(
                             self.core_ranks))
        self.batch_size = None

        # reported compress ratio in input->hidden weight
        self.compress_ratio = self.params_original / self.params_bt

        if self.debug:
            print('bt_input_shape = ' + str(self.bt_input_shape))
            print('bt_output_shape = ' + str(self.bt_output_shape))
            print('core_ranks = ' + str(self.core_ranks))
            print('block_ranks = ' + str(self.block_ranks))
            print('compress_ratio = ' + str(self.compress_ratio))
        assert len(self.core_ranks.shape) == len(self.bt_input_shape.shape) == len(self.bt_output_shape.shape)

    def build(self, input_shape):
        print('BT-LSTM input shape = ' + str(input_shape))

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        self.batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec[0] = InputSpec(shape=(self.batch_size, None, self.input_dim))

        self.states = [None, None]
        if self.stateful:
            self.reset_states()

        ################################################################################################################
        # input -> hidden state
        self.kernel = self.add_weight((self.params_bt,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.cores, self.factors = split_kernel_into_core_and_factors(self.kernel,
                                                                      self.bt_input_shape, self.bt_output_shape,
                                                                      self.core_ranks, self.block_ranks)

        ################################################################################################################

        # hidden -> hidden
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def preprocess_input(self, x, training=None):
        return x

    def get_constants(self, inputs, training=None):
        # this is totally same as the Keras API
        constants = []
        if self.implementation != 0 and 0. < self.dropout < 1:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0. < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def step(self, inputs, states):
        h_tm1 = states[0]
        c_tm1 = states[1]
        dp_mask = states[2]
        rec_dp_mask = states[3]

        if len(self.core_ranks) == 2:
            z = BT_mul2(inputs * dp_mask[0],
                        self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 3:
            z = BT_mul3(inputs * dp_mask[0],
                        self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 4:
            z = BT_mul4(inputs * dp_mask[0],
                        self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        elif len(self.core_ranks) == 5:
            z = BT_mul5(inputs * dp_mask[0],
                        self.cores, self.factors, self.bt_input_shape, self.bt_output_shape, self.core_ranks)
        else:
            raise ValueError('error in len(core_ranks)')

        z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]

        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)

        h = o * self.activation(c)
        if 0. < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c]

    def get_config(self):
        config = {'bt_input_shape': self.bt_input_shape,
                  'bt_output_shape': self.bt_output_shape,
                  'core_ranks': self.core_ranks,
                  'block_ranks': self.block_ranks,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(BT_LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
