import tensorflow as tf
import numpy as np
import time
import cifar10


# batch_size = 50
# CP_r = 4
# Tuk_r = 2
# d_in = 2
# d_out = 2
#
# Input = tf.truncated_normal([batch_size, 4, 4, 4, 4, 4, 4], stddev=5e-2)


def BT_Layer12(Input):
    # Input: batch_size*64*64
    # W1-12 = tf.truncated_normal([d_in, Tuk_r, d_out], stddev=5e-2)
    # W_core = tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r, Tuk_r ** 3, Tuk_r ** 6], stddev=0.1)
    d_in = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    d_out = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    Tuk_r = 2
    W1 = tf.Variable(tf.truncated_normal([d_in[0], Tuk_r, d_out[0]], stddev=0.1))
    W2 = tf.Variable(tf.truncated_normal([d_in[1], Tuk_r, d_out[1]], stddev=0.1))
    W3 = tf.Variable(tf.truncated_normal([d_in[2], Tuk_r, d_out[2]], stddev=0.1))
    W4 = tf.Variable(tf.truncated_normal([d_in[3], Tuk_r, d_out[3]], stddev=0.1))
    W5 = tf.Variable(tf.truncated_normal([d_in[4], Tuk_r, d_out[4]], stddev=0.1))
    W6 = tf.Variable(tf.truncated_normal([d_in[5], Tuk_r, d_out[5]], stddev=0.1))
    W7 = tf.Variable(tf.truncated_normal([d_in[6], Tuk_r, d_out[6]], stddev=0.1))
    W8 = tf.Variable(tf.truncated_normal([d_in[7], Tuk_r, d_out[7]], stddev=0.1))
    W9 = tf.Variable(tf.truncated_normal([d_in[8], Tuk_r, d_out[8]], stddev=0.1))
    W10 = tf.Variable(tf.truncated_normal([d_in[9], Tuk_r, d_out[9]], stddev=0.1))
    W11 = tf.Variable(tf.truncated_normal([d_in[10], Tuk_r, d_out[10]], stddev=0.1))
    W12 = tf.Variable(tf.truncated_normal([d_in[11], Tuk_r, d_out[11]], stddev=0.1))
    W_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r, Tuk_r ** 3, Tuk_r ** 6], stddev=0.1))

    op = 'abc,behxy,def,ghi->adgxycfi'
    W123 = tf.reshape(tf.einsum(op, W1, W_core, W2, W3), [d_in[0] ** 3, Tuk_r ** 3, Tuk_r ** 6, d_out[0] ** 3])
    op = 'abc,def,ghi->adgbehcfi'
    W456 = tf.reshape(tf.einsum(op, W4, W5, W6), [d_in[3] ** 3, Tuk_r ** 3, d_out[3] ** 3])
    op = 'abcd,ebf->aecdf'
    W123456 = tf.reshape(tf.einsum(op, W123, W456), [(d_in[0] * d_in[3]) ** 3, Tuk_r ** 6, (d_out[0] * d_out[3]) ** 3])

    op = 'abc,def,ghi->adgbehcfi'
    W789 = tf.reshape(tf.einsum(op, W7, W8, W9), [d_in[6] ** 3, Tuk_r ** 3, d_out[6] ** 3])
    op = 'abc,def,ghi->adgbehcfi'
    W101112 = tf.reshape(tf.einsum(op, W10, W11, W12), [d_in[9] ** 3, Tuk_r ** 3, d_out[9] ** 3])
    op = 'abc,def->adbecf'
    W789_12 = tf.reshape(tf.einsum(op, W789, W101112),
                         [(d_in[6] * d_in[9]) ** 3, Tuk_r ** 6, (d_out[6] * d_out[9]) ** 3])

    op = 'abc,dae,ebf->dcf'
    return tf.reshape(tf.einsum(op, W123456, Input, W789_12), [-1, (d_out[0] * d_out[3] * d_out[6] * d_out[9]) ** 3])


def BT_Layer6(Input, num_layer):
    d_in = [4, 3, 3, 4, 4, 4]
    d_out = [4, 4, 3, 3, 3, 3]
    d_in_half1, d_in_half2 = d_in[0] * d_in[1] * d_in[2], d_in[3] * d_in[4] * d_in[5]
    d_out_half1, d_out_half2 = d_out[0] * d_out[1] * d_out[2], d_out[3] * d_out[4] * d_out[5]
    Tuk_r = 1
    W1 = cifar10._variable_with_weight_decay('W1%s' % num_layer, shape=[d_in[0], Tuk_r, d_out[0]], stddev=0.4, wd=0.04)
    W2 = cifar10._variable_with_weight_decay('W2%s' % num_layer, shape=[d_in[1], Tuk_r, d_out[1]], stddev=0.4, wd=0.04)
    W3 = cifar10._variable_with_weight_decay('W3%s' % num_layer, shape=[d_in[2], Tuk_r, d_out[2]], stddev=0.4, wd=0.04)
    W4 = cifar10._variable_with_weight_decay('W4%s' % num_layer, shape=[d_in[3], Tuk_r, d_out[3]], stddev=0.4, wd=0.04)
    W5 = cifar10._variable_with_weight_decay('W5%s' % num_layer, shape=[d_in[4], Tuk_r, d_out[4]], stddev=0.4, wd=0.04)
    W6 = cifar10._variable_with_weight_decay('W6%s' % num_layer, shape=[d_in[5], Tuk_r, d_out[5]], stddev=0.4, wd=0.04)
    W_core = cifar10._variable_with_weight_decay('W_core%s' % num_layer, shape=[Tuk_r, Tuk_r, Tuk_r, Tuk_r ** 3],
                                                 stddev=0.4, wd=0.04)

    # W1 = tf.Variable(tf.truncated_normal([d_in[0], Tuk_r, d_out[0]], stddev=0.35))
    # W2 = tf.Variable(tf.truncated_normal([d_in[1], Tuk_r, d_out[1]], stddev=0.35))
    # W3 = tf.Variable(tf.truncated_normal([d_in[2], Tuk_r, d_out[2]], stddev=0.35))
    # W4 = tf.Variable(tf.truncated_normal([d_in[3], Tuk_r, d_out[3]], stddev=0.35))
    # W5 = tf.Variable(tf.truncated_normal([d_in[4], Tuk_r, d_out[4]], stddev=0.35))
    # W6 = tf.Variable(tf.truncated_normal([d_in[5], Tuk_r, d_out[5]], stddev=0.35))
    # W_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r, Tuk_r ** 3], stddev=0.35))

    op = 'abc,behx,def,ghi->adgxcfi'
    W123 = tf.reshape(tf.einsum(op, W1, W_core, W2, W3), [d_in_half1, Tuk_r ** 3, d_out_half1])
    op = 'abc,def,ghi->adgbehcfi'
    W456 = tf.reshape(tf.einsum(op, W4, W5, W6), [d_in_half2, Tuk_r ** 3, d_out_half2])
    op = 'abc,dae,ebf->dcf'
    return tf.reshape(tf.einsum(op, W123, Input, W456), [-1, d_out_half1 * d_out_half2])
    # op = 'debc,ebf->dcf'
    # return tf.reshape(tf.einsum(op, tf.einsum('dae,abc->debc', Input, W123), W456), [-1, d_out_half1 * d_out_half2])


def BT_Layer4(Input, num_layer,stddev=0.4):
    print('using BT_layer4...')
    d_in = [6, 6, 6, 6]
    d_out = [4, 4, 4, 6]
    Tuk_r = 3
    W1 = cifar10._variable_with_weight_decay('W1%s' % num_layer, shape=[d_in[0], Tuk_r, d_out[0]], stddev=stddev, wd=0.04)
    W2 = cifar10._variable_with_weight_decay('W2%s' % num_layer, shape=[d_in[1], Tuk_r, d_out[1]], stddev=stddev, wd=0.04)
    W3 = cifar10._variable_with_weight_decay('W3%s' % num_layer, shape=[d_in[2], Tuk_r, d_out[2]], stddev=stddev, wd=0.04)
    W4 = cifar10._variable_with_weight_decay('W4%s' % num_layer, shape=[d_in[3], Tuk_r, d_out[3]], stddev=stddev, wd=0.04)
    W_core = cifar10._variable_with_weight_decay('W_core%s' % num_layer, shape=[Tuk_r, Tuk_r, Tuk_r ** 2], stddev=stddev,
                                                 wd=0.04)

    # W1 = tf.Variable(tf.truncated_normal([d_in[0], Tuk_r, d_out[0]], stddev=stddev))
    # W2 = tf.Variable(tf.truncated_normal([d_in[1], Tuk_r, d_out[1]], stddev=stddev))
    # W3 = tf.Variable(tf.truncated_normal([d_in[2], Tuk_r, d_out[2]], stddev=stddev))
    # W4 = tf.Variable(tf.truncated_normal([d_in[3], Tuk_r, d_out[3]], stddev=stddev))
    # W_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r ** 2], stddev=stddev))

    op = 'abc,bex,def->adxcf'
    W12 = tf.reshape(tf.einsum(op, W1, W_core, W2), [d_in[0] * d_in[1], Tuk_r ** 2, d_out[0] * d_out[1]])
    op = 'abc,def->adbecf'
    W34 = tf.reshape(tf.einsum(op, W3, W4), [d_in[2] * d_in[3], Tuk_r ** 2, d_out[2] * d_out[3]])
    op = 'abc,dae,ebf->dcf'
    return tf.reshape(tf.einsum(op, W12, Input, W34), [-1, d_out[0] * d_out[1] * d_out[2] * d_out[3]])


def BT_Layer4_add(Input, num_layer):
    d_in = [8, 8, 8, 8]
    d_out = [4, 4, 4, 4]
    Tuk_r = 3
    W1 = cifar10._variable_with_weight_decay('W1%s' % num_layer, shape=[d_in[0], Tuk_r, d_out[0]], stddev=0.4, wd=0.04)
    W2 = cifar10._variable_with_weight_decay('W2%s' % num_layer, shape=[d_in[1], Tuk_r, d_out[1]], stddev=0.4, wd=0.04)
    W3 = cifar10._variable_with_weight_decay('W3%s' % num_layer, shape=[d_in[2], Tuk_r, d_out[2]], stddev=0.4, wd=0.04)
    W4 = cifar10._variable_with_weight_decay('W4%s' % num_layer, shape=[d_in[3], Tuk_r, d_out[3]], stddev=0.4, wd=0.04)
    W_core = cifar10._variable_with_weight_decay('W_core%s' % num_layer, shape=[Tuk_r, Tuk_r, Tuk_r ** 2], stddev=0.4,
                                                 wd=0.04)

    # W1 = tf.Variable(tf.truncated_normal([d_in[0], Tuk_r, d_out[0]], stddev=0.3))
    # W2 = tf.Variable(tf.truncated_normal([d_in[1], Tuk_r, d_out[1]], stddev=0.3))
    # W3 = tf.Variable(tf.truncated_normal([d_in[2], Tuk_r, d_out[2]], stddev=0.3))
    # W4 = tf.Variable(tf.truncated_normal([d_in[3], Tuk_r, d_out[3]], stddev=0.3))
    # W_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r ** 2], stddev=0.3))

    op = 'abc,bex,def->adxcf'
    W12 = tf.reshape(tf.einsum(op, W1, W_core, W2), [d_in[0] * d_in[1], Tuk_r ** 2, d_out[0] * d_out[1]])
    op = 'abc,def->adbecf'
    W34 = tf.reshape(tf.einsum(op, W3, W4), [d_in[2] * d_in[3], Tuk_r ** 2, d_out[2] * d_out[3]])
    op = 'abc,dae,ebf->dcf'
    return tf.reshape(tf.einsum(op, W12, Input, W34), [-1, d_out[0] * d_out[1] * d_out[2] * d_out[3]])


def BT_Layer3(Input, num_layer):
    d_in = [12, 12, 16]
    d_out = [12, 12, 9]
    Tuk_r = 3
    W1 = cifar10._variable_with_weight_decay('W1%s' % num_layer, shape=[d_in[0], Tuk_r, d_out[0]], stddev=0.4, wd=0.04)
    W3 = cifar10._variable_with_weight_decay('W3%s' % num_layer, shape=[d_in[1], Tuk_r, d_out[1]], stddev=0.4, wd=0.04)
    W4 = cifar10._variable_with_weight_decay('W4%s' % num_layer, shape=[d_in[2], Tuk_r, d_out[2]], stddev=0.4, wd=0.04)
    W_core = cifar10._variable_with_weight_decay('W_core%s' % num_layer, shape=[Tuk_r, Tuk_r ** 2], stddev=0.4,
                                                 wd=0.04)

    # W1 = tf.Variable(tf.truncated_normal([d_in[0], Tuk_r, d_out[0]], stddev=0.3))
    # W2 = tf.Variable(tf.truncated_normal([d_in[1], Tuk_r, d_out[1]], stddev=0.3))
    # W3 = tf.Variable(tf.truncated_normal([d_in[2], Tuk_r, d_out[2]], stddev=0.3))
    # W4 = tf.Variable(tf.truncated_normal([d_in[3], Tuk_r, d_out[3]], stddev=0.3))
    # W_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r ** 2], stddev=0.3))
    op = 'abc,be->aec'
    W1core = tf.reshape(tf.einsum(op, W1, W_core), [d_in[0], Tuk_r ** 2, d_out[0]])
    op = 'abc,def->adbecf'
    W34 = tf.reshape(tf.einsum(op, W3, W4), [d_in[1] * d_in[2], Tuk_r ** 2, d_out[1] * d_out[2]])
    op = 'abc,dae,ebf->dcf'
    return tf.reshape(tf.einsum(op, W1core, Input, W34), [-1, d_out[0] * d_out[1] * d_out[2]])


def BT_Layer2(Input, num_layer):
    d_in = [36, 64]
    d_out = [36, 36]
    Tuk_r = 1

    W1 = cifar10._variable_with_weight_decay('W1%s' % num_layer, shape=[d_in[0], Tuk_r, d_out[0]], stddev=0.4, wd=0.04)
    W2 = cifar10._variable_with_weight_decay('W2%s' % num_layer, shape=[d_in[1], Tuk_r, d_out[1]], stddev=0.4, wd=0.04)
    W_core = cifar10._variable_with_weight_decay('W_core%s' % num_layer, shape=[Tuk_r, Tuk_r], stddev=0.4, wd=0.04)

    # W1 = tf.Variable(tf.truncated_normal([d_in[0], Tuk_r, d_out[0]], stddev=0.22))
    # W2 = tf.Variable(tf.truncated_normal([d_in[1], Tuk_r, d_out[1]], stddev=0.22))
    # W_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r], stddev=0.22))
    op = 'abc,be->aec'
    W1core = tf.reshape(tf.einsum(op, W1, W_core), [d_in[0], Tuk_r, d_out[0]])
    op = 'abc,dae,ebf->dcf'
    return tf.reshape(tf.einsum(op, W1core, Input, W2), [-1, d_out[0] * d_out[1]])

def BT_Layer2_add(Input, num_layer):
    d_in = [36, 36]
    d_out = [16, 12]
    Tuk_r = 3

    W1 = cifar10._variable_with_weight_decay('W1%s' % num_layer, shape=[d_in[0], Tuk_r, d_out[0]], stddev=0.4, wd=0.04)
    W2 = cifar10._variable_with_weight_decay('W2%s' % num_layer, shape=[d_in[1], Tuk_r, d_out[1]], stddev=0.4, wd=0.04)
    W_core = cifar10._variable_with_weight_decay('W_core%s' % num_layer, shape=[Tuk_r, Tuk_r], stddev=0.4, wd=0.04)

    # W1 = tf.Variable(tf.truncated_normal([d_in[0], Tuk_r, d_out[0]], stddev=0.22))
    # W2 = tf.Variable(tf.truncated_normal([d_in[1], Tuk_r, d_out[1]], stddev=0.22))
    # W_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r], stddev=0.22))
    op = 'abc,be->aec'
    W1core = tf.reshape(tf.einsum(op, W1, W_core), [d_in[0], Tuk_r, d_out[0]])
    op = 'abc,dae,ebf->dcf'
    return tf.reshape(tf.einsum(op, W1core, Input, W2), [-1, d_out[0] * d_out[1]])



# Y2 = tf.reshape(Input, [-1, d_in ** 6, d_in ** 6])
# # Y3 = BT_Layer12(Y2, batch_size)
# # Y3 = BT_Layer6(Y2, batch_size)
# Y3 = BT_Layer4(Y2, batch_size)
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# with sess:
#     sess.run(init)
#     time_st = time.time()
#     for i in range(90):
#         sess.run(Y3)
#     print(time.time() - time_st)
