import mxnet as mx
import numpy as np
import logging

data_path = "/home/chendi/data/"
batch_size = 1
I1, I2, I3, I4, J1, J2, J3, J4 = 8, 8, 8, 8, 8, 8, 8, 8
Rank = 3
stddev = 0.5

train_iter = mx.io.ImageRecordIter(
    path_imgrec='/hdd/chendi/cifar10/cifar10_train.rec',
    data_shape=(3, 32, 32),
    # pad=4,
    # rand_crop=True,
    # max_crop_size=32,
    # min_crop_size=32,
    # mirror=True,
    # random_l=63,
    # max_random_contrast=50,
    batch_size = batch_size,
)

val_iter = mx.io.ImageRecordIter(
    path_imgrec='/hdd/chendi/cifar10/cifar10_val.rec',
    data_shape=(3, 32, 32),
    batch_size=batch_size,
    # pad=4,
    # rand_crop=True,
    # max_crop_size=32,
    # min_crop_size=32,
    # mirror=True,
    # random_l=63,
    # max_random_contrast=50,
)

data = mx.symbol.Variable('data')
conv1 = mx.symbol.Convolution(name='conv1', data=data, kernel=(5, 5), stride=(1, 1), num_filter=64, pad=(3, 3))
norm1 = mx.symbol.BatchNorm(data=conv1)

relu1 = mx.symbol.Activation(data=norm1, act_type="relu")
pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), stride=(2, 2))

conv2 = mx.symbol.Convolution(name='conv2', data=pool1, kernel=(5, 5), stride=(1, 1), num_filter=64, pad=(3, 3))
norm2 = mx.symbol.BatchNorm(data=conv2)
relu2 = mx.symbol.Activation(data=norm2, act_type="relu")
pool2 = mx.symbol.Pooling(data=relu2, pool_type="max", kernel=(3, 3), stride=(2, 2))
resh = mx.symbol.Flatten(data=pool2)

W1 = mx.symbol.Variable('W1', shape=(I1 * J1 * Rank, 1), init=mx.init.Normal(stddev))
W2 = mx.symbol.Variable('W2', shape=(1, I2 * J2 * Rank), init=mx.init.Normal(stddev))
W3 = mx.symbol.Variable('W3', shape=(Rank * I3 * J3, 1), init=mx.init.Normal(stddev))
W4 = mx.symbol.Variable('W4', shape=(1, Rank * I4 * J4), init=mx.init.Normal(stddev))
W_core = mx.symbol.Variable('W_core', shape=(Rank ** 2, Rank ** 2), init=mx.init.Normal(stddev))
# multiply W1 and W2 and W_core
s1_tmp = mx.symbol.dot(W1, W2)
s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1, J1 * Rank, I2, J2 * Rank))
s1_tmp = mx.symbol.transpose(s1_tmp, axes=(0, 2, 1, 3))
s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1 * I2 * J1, Rank, J2, Rank))
s1_tmp = mx.symbol.transpose(s1_tmp, axes=(0, 2, 1, 3))
s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1 * I2 * J1 * J2, Rank * Rank))
s1_tmp = mx.symbol.dot(s1_tmp, W_core)
s1_tmp = mx.symbol.reshape(s1_tmp, shape=(I1 * I2, J1 * J2 * Rank * Rank))
# multiply InData
s2_tmp = mx.symbol.reshape(resh, shape=(batch_size, I1 * I2, I3 * I4))
s2_tmp = mx.symbol.transpose(s2_tmp, axes=(0, 2, 1))
s2_tmp = mx.symbol.reshape(s2_tmp, shape=(batch_size * I3 * I4, I1 * I2))
s2_tmp = mx.symbol.dot(s2_tmp, s1_tmp)
s2_tmp = mx.symbol.reshape(s2_tmp, shape=(batch_size, I3 * I4, J1 * J2 * Rank * Rank))
s2_tmp = mx.symbol.transpose(s2_tmp, axes=(0, 2, 1))
s2_tmp = mx.symbol.reshape(s2_tmp, shape=(batch_size * J1 * J2, Rank * Rank * I3 * I4))
# multiply W3 and W4
s3_tmp = mx.symbol.dot(W3, W4)
s3_tmp = mx.symbol.reshape(s3_tmp, shape=(Rank, I3 * J3, Rank, I4 * J4))
s3_tmp = mx.symbol.transpose(s3_tmp, axes=(0, 2, 1, 3))
s3_tmp = mx.symbol.reshape(s3_tmp, shape=(Rank * Rank * I3, J3, I4, J4))
s3_tmp = mx.symbol.transpose(s3_tmp, axes=(0, 2, 1, 3))
s3_tmp = mx.symbol.reshape(s3_tmp, shape=(Rank * Rank * I3 * I4, J3 * J4))
# multiply InData and W34
OutData = mx.symbol.reshape(mx.symbol.dot(s2_tmp, s3_tmp), shape=(batch_size, J1 * J2 * J3 * J4))

# W31 = mx.symbol.Variable('W31', shape=(3136, 384), init=mx.init.Normal(0.1))
# b3 = mx.symbol.Variable('b3', shape=(50, 384), init=mx.init.One())
#
# fc2 = mx.symbol.dot(resh, W31) + b3 / 10

# fc2 = mx.symbol.FullyConnected(name='fc2', data=resh, num_hidden=1024)

# norm3 = mx.symbol.BatchNorm(data=fc2)
# relu3 = mx.symbol.Activation(data=norm3, act_type="relu")
#
# W41 = mx.symbol.Variable('W41', shape=(384, 192), init=mx.init.Normal(0.1))
# b4 = mx.symbol.Variable('b4', shape=(50, 192), init=mx.init.One())
#
# fc3 = mx.symbol.dot(relu3, W41) + b4 / 10
# fc2 = mx.symbol.FullyConnected(name='fc2', data=resh, num_hidden=1024)


norm4 = mx.symbol.BatchNorm(data=OutData)
relu4 = mx.symbol.Activation(data=norm4, act_type="relu")
dp2 = mx.symbol.Dropout(data=relu4,name="dp1")

fc4 = mx.symbol.FullyConnected(name='fc4', data=dp2, num_hidden=10)
softmax = mx.symbol.SoftmaxOutput(data=fc4, name='softmax')

logging.getLogger().setLevel(logging.DEBUG)

# model = mx.mod.Module(symbol=softmax, context=mx.gpu(0), fixed_param_names=('W31',))
model = mx.mod.Module(symbol=softmax, context=mx.gpu(0))

# model.bind(data_shapes=[('data', [128, 3, 32, 32]), ])

#  w31 = mx.ndarray.array(np.random.normal(size=(3136, 1024)))


# init = mx.init.One()
# model.init_params(init)
# model.set_params({'W31': mx.ndarray.array(np.zeros(shape=(4096,1024)))}, {},allow_missing=True, allow_extra=False)


# for d in model.get_params():
#      for key in d:
#          print key
#          print d[key].asnumpy()
# loss_m = mx.metric.Loss()
# metric = mx.metric.create(['loss','acc'])

model.fit(train_iter,  # train data
          eval_data=val_iter,  # validation data
          optimizer='adam',  # use SGD to train
          optimizer_params={'learning_rate': 0.001},  # use fixed learning rate
          # eval_metric='acc',  # report accuracy during training
          batch_end_callback=mx.callback.Speedometer(batch_size, 100),  # output progress for each 100 data batches
          num_epoch=100)  # train for at most 10 dataset passes
