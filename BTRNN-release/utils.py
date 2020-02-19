# Created by ay27 at 08/10/2017
import threading

import time
from keras.callbacks import Callback
import csv
import numpy as np

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as K
import keras
import tensorflow


def set_gpu(gpu_memory_ratio):
    tf.set_random_seed(11111986)

    def get_session(gpu_fraction):
        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    K.set_session(get_session(gpu_memory_ratio))


# from https://stanford.edu/~shervine/blog/keras-generator-multiprocessing.html
class threadsafe_iter(object):
    """
      Takes an iterator/generator and makes it thread-safe by
      serializing call to the `next` method of given iterator/generator.
      """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """
      A decorator that takes a generator function and makes it thread-safe.
      """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


class MetricsHistory(Callback):
    def __init__(self, saving_path, params):
        super().__init__()
        self.saving_path = saving_path
        self.epochs = []
        self.losses = []
        self.val_losses = []
        self.accs = []
        self.val_accs = []
        self.elapse_time = []
        self.start_time = time.time()
        self.params = params

        with open(self.saving_path, 'w') as params_file:
            params_file.write(str(self.params))

    def on_epoch_end(self, epoch, logs={}):
        # Store
        self.epochs.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('acc'))
        self.val_accs.append(logs.get('val_acc'))
        self.elapse_time.append(time.time() - self.start_time)

        header = ['ecochs', 'train_losses', 'val_losses', 'train_accs', 'val_accs', 'elapse_time']
        # Save to file
        res = np.array([self.epochs, self.losses, self.val_losses, self.accs, self.val_accs, self.elapse_time])

        with open(self.saving_path, 'w') as file:
            w = csv.writer(file)
            w.writerow(header)
            w.writerows(res.T)


def mean_average_precision(y_true, y_pred):
    return tensorflow.reduce_mean(
        tensorflow.metrics.sparse_average_precision_at_k(tensorflow.cast(y_true, tensorflow.int64), y_pred, 1)[0])
