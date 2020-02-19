# Created by ay27 at 08/10/2017
from keras.optimizers import Adam

import utils
from dataset.UCF11 import *
from BT_RNN.BTRNN import *
from TT_RNN.TTRNN import *
import sys
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Masking
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
import time
from utils import MetricsHistory
import tensorflow as tf
import keras

model_save_path = '/home/ay27/hdd/BT_RNN_dump_model/ucf_tune3/'
data_path = '/home/ay27/hdd/dataset/UCF11_6/'
metrics_save_path = '/home/ay27/hdd/BT_RNN_metric/ucf_tune3/'


net_names = ['rnn', 'gru', 'lstm',
             'ttrnn', 'ttgru', 'ttlstm',
             'btrnn', 'btgru', 'btlstm']

networks = {net_names[0]: SimpleRNN, net_names[1]: GRU, net_names[2]: LSTM,
            net_names[3]: TT_RNN, net_names[4]: TT_GRU, net_names[5]: TT_LSTM,
            net_names[6]: BT_RNN, net_names[7]: BT_GRU, net_names[8]: BT_LSTM}

params2 = dict(
    max_time_len=6,
    input_shape=[320, 180],
    output_shape=[16, 16],
    core_ranks=[4, 4],
    tt_ranks=[1, 4, 1],
    blocks=2,
    batch_size=16,
    alpha=1e-2,
    epochs=500,
    net_type=-1
)

params3 = dict(
    max_time_len=6,
    input_shape=[36, 40, 40],
    output_shape=[8, 4, 8],
    core_ranks=[4, 4, 4],
    tt_ranks=[1, 4, 4, 1],
    blocks=2,
    batch_size=16,
    alpha=1e-2,
    epochs=500,
    net_type=-1
)

params4 = dict(
    max_time_len=6,
    input_shape=[8, 20, 20, 18],
    output_shape=[4, 4, 4, 4],
    core_ranks=[4, 4, 4, 4],
    tt_ranks=[1, 4, 4, 4, 1],
    blocks=2,
    batch_size=16,
    alpha=1e-2,
    epochs=500,
    net_type=-1
)

params5 = dict(
    max_time_len=6,
    input_shape=[8, 10, 10, 9, 8],
    output_shape=[4, 2, 2, 4, 4],
    core_ranks=[4, 4, 4, 4, 4],
    tt_ranks=[1, 4, 4, 4, 1],
    blocks=2,
    batch_size=16,
    alpha=1e-2,
    epochs=500,
    net_type=-1
)

pp = [params2, params3, params4, params5]


def network_architecture(params):
    input = Input(shape=(params['max_time_len'], 120 * 160 * 3))
    masked_input = Masking(mask_value=0, input_shape=(params4['max_time_len'], 120 * 160 * 3))(input)
    net_type = params['net_type']

    if net_type in net_names[:3]:
        # rnn, gru, lstm
        rnn_layer = networks[net_type](units=np.array(params['output_shape']).prod(),
                                       return_sequences=False,
                                       dropout=0.25, recurrent_dropout=0.25, activation='tanh',
                                       kernel_initializer=keras.initializers.TruncatedNormal(0., 0.1),
                                       implementation=2)
    elif net_type in net_names[3:6]:
        # ttrnn, ttgru, ttlstm
        rnn_layer = networks[net_type](tt_input_shape=params['input_shape'], tt_output_shape=params['output_shape'],
                                       tt_ranks=params['tt_ranks'],
                                       return_sequences=False,
                                       dropout=0.25, recurrent_dropout=0.25, activation='tanh',
                                       kernel_initializer=keras.initializers.TruncatedNormal(0., 0.08 * (
                                               5. - params['core_ranks'][0])))
    else:
        # btrnn, btgru, btlstm
        rnn_layer = networks[net_type](params['input_shape'], params['output_shape'],
                                       params['core_ranks'], params['blocks'], return_sequences=False,
                                       dropout=0.25, recurrent_dropout=0.25, activation='tanh',
                                       kernel_initializer=keras.initializers.TruncatedNormal(0., 0.05 * (
                                               5. - params['core_ranks'][0])))
    h = rnn_layer(masked_input)
    output = Dense(units=11, activation='softmax', kernel_regularizer=l2(params['alpha']))(h)
    return Model(input, output)


def main(params):
    dataset = UCF11DataSet(data_path, params['max_time_len'], [160, 120, 3], 0.2, params['batch_size'], True)
    train_generator = dataset.generate_train()
    test_generator = dataset.generate_test()

    model = network_architecture(params)
    model.compile(optimizer=Adam(1e-4, decay=0.00016667), loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(model_save_path, save_best_only=True)
    metrics_callback = MetricsHistory(metrics_save_path, params)

    start = time.time()
    model.fit_generator(generator=train_generator, steps_per_epoch=len(dataset.data_train) // params['batch_size'],
                        epochs=params['epochs'], verbose=1,
                        validation_data=test_generator,
                        validation_steps=len(dataset.data_test) // params['batch_size'],
                        workers=4, callbacks=[checkpoint_callback, metrics_callback])
    stop = time.time()
    print('total time : %f' % (stop - start))


def print_help():
    helps = """Usage:
        python main.py [net-type] args
        net-type must be [rnn, gru, lstm, ttrnn, ttgru, ttlstm, btrnn, btgru, btlstm]
        when net-type in [rnn, gru, lstm]:
            args: None
        when net-type in [ttrnn, ttgru, ttlstm]:
            args: d rank
        when net-type in [btrnn, btgru, btlstm]:
            args: d rank blocks
        """
    print(helps)


def print_params(params):
    for k, v in params.items():
        print(k, ':', v)


if __name__ == '__main__':
    np.random.seed(11111986)
    tf.set_random_seed(11111986)

    print(sys.argv)
    if len(sys.argv) < 2:
        print_help()
        exit(-1)

    net_type = sys.argv[1]
    if net_type not in net_names:
        print('net-type must in [rnn, gru, lstm, ttrnn, ttgru, ttlstm, btrnn, btgru, btlstm]')
        exit(-1)

    os.makedirs(metrics_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    if net_type in net_names[:3]:
        assert len(sys.argv) == 2
        # rnn, gru, lstm
        params = pp[0]

        metrics_save_path += '%s.csv' % net_type
        model_save_path += '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % net_type
    elif net_type in net_names[3:6]:
        assert len(sys.argv) == 4
        # tt networks
        d, rank = sys.argv[2:]
        d, rank = int(d), int(rank)
        params = pp[d - 2]

        metrics_save_path += '%s_d%d_rank%d.csv' % (net_type, d, rank)
        model_save_path += '%s_d%d_rank%d-{epoch:02d}-{val_loss:.2f}.hdf5' % (net_type, d, rank)
    else:
        assert len(sys.argv) == 5
        # bt networks
        d, rank, block = sys.argv[2:]
        d, rank, block = int(d), int(rank), int(block)
        params = pp[d - 2]
        params['core_ranks'] = [rank] * d
        params['blocks'] = block

        metrics_save_path += '%s_d%d_block%d_rank%d.csv' % (net_type, d, params['blocks'], rank)
        model_save_path += '%s_d%d_block%d_rank%d-{epoch:02d}-{val_loss:.2f}.hdf5' % (
            net_type, d, params['blocks'], rank)

    params['net_type'] = net_type

    print_params(params)
    print('Model save in: %s' % model_save_path)
    print('Metrics save in: %s' % metrics_save_path)

    main(params)
