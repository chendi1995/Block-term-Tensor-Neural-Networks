# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

def bt_weight_variable(shape,Tuk_r,stddev=0.1):
    print('using bt...')
    print(stddev)
    print(shape)##[3,3,256,256]
    _W21 = tf.Variable(tf.truncated_normal([shape[0], Tuk_r, 1], stddev=stddev))
    _W22 = tf.Variable(tf.truncated_normal([shape[1], Tuk_r, 1], stddev=stddev))
    _W23 = tf.Variable(tf.truncated_normal([shape[2], Tuk_r, shape[3]], stddev=stddev))
    _W2_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r], stddev=stddev))
    op = 'beh,abc,def,ghi->adgcfi'
    kernel1 = tf.reshape(tf.einsum(op, _W2_core, _W21, _W22, _W23), [shape[0], shape[1], shape[2], shape[3]])
    #
    # W21 = tf.Variable(tf.truncated_normal([shape[0], Tuk_r, 1], stddev=stddev))
    # W22 = tf.Variable(tf.truncated_normal([shape[1], Tuk_r, 1], stddev=stddev))
    # W23 = tf.Variable(tf.truncated_normal([shape[2], Tuk_r, shape[3]], stddev=stddev))
    # W2_core = tf.Variable(tf.truncated_normal([Tuk_r, Tuk_r, Tuk_r], stddev=stddev))
    # op = 'beh,abc,def,ghi->adgcfi'
    # kernel = kernel1 + tf.reshape(tf.einsum(op, W2_core, W21, W22, W23), [shape[0], shape[1], shape[2], shape[3]])

    return kernel1

def tt_weight_variable(shape,TT_r,stddev):
    print('using tt...')
    print(TT_r)
    W21 = tf.Variable(tf.truncated_normal([shape[0], TT_r, 1], stddev=stddev))
    W22 = tf.Variable(tf.truncated_normal([shape[1], TT_r, 1, TT_r], stddev=stddev))
    W23 = tf.Variable(tf.truncated_normal([shape[2], TT_r, shape[3]], stddev=stddev))
    op = 'abc,dbef,gfh->adgceh'
    kernel = tf.reshape(tf.einsum(op, W21, W22, W23), [shape[0], shape[1], shape[2], shape[3]])
    return kernel



def weight_variable(shape, stddev=0.1, name="weight"):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)

def deconv2d(x, W,stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='VALID', name="conv2d_transpose")

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='VALID')

def crop_and_concat(x1,x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)

def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keep_dims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keep_dims=True)
        return exponential_map / normalize

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")