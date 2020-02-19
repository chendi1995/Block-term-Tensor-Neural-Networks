# Created by ay27 at 08/10/2017
# from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import numpy as np
from utils import threadsafe_generator
import os
import imageio
import cv2
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle
import random
import sys

classes = ['basketball', 'biking', 'diving', 'golf_swing', 'horse_riding', 'soccer_juggling',
           'swing', 'tennis_swing', 'trampoline_jumping', 'volleyball_spiking', 'walking']

total_frames = 0


def __read_clip(path, img_shape, max_time_len):
    global total_frames
    vid = imageio.get_reader(path, 'ffmpeg')
    this_clip = []
    for v in vid:
        this_clip.append(cv2.resize(v, (img_shape[0], img_shape[1])))
    total_frames += len(this_clip)
    # this_clip = [cv2.resize(vid.get_data(_), (img_shape[0], img_shape[1])) for _ in range(len(vid))]
    this_clip = np.asanyarray(this_clip)
    this_clip = this_clip.reshape(this_clip.shape[0], -1)  # of shape (nb_frames, data_shape)
    this_clip = (this_clip - 128.).astype('int8')  # this_clip.mean()
    if len(this_clip) > max_time_len:
        this_clip = this_clip[sorted(random.sample(range(0, len(this_clip)), max_time_len))]
    return pad_sequences([this_clip], maxlen=max_time_len, truncating='post', dtype='int8')[0]


def preprocess_UCF11(data_path, write_out_path, max_time_len, img_shape):
    res = []
    for ii, class_name in enumerate(classes):
        files = os.listdir(os.path.join(data_path, class_name))
        for this_collection in files:
            if this_collection == 'Annotation':
                continue
            os.makedirs(os.path.join(write_out_path, class_name + '/' + this_collection), exist_ok=True)

            clips = os.listdir(os.path.join(data_path, class_name + '/' + this_collection))
            clips.sort()
            for this_clip in clips:
                if this_clip.endswith('xgtf') or this_clip.endswith('pkl'):
                    continue
                path = os.path.join(data_path, class_name + '/' + this_collection + '/' + this_clip)
                res.append(path)
    all_clips = np.asarray(res)
    print('scan all original clips, #total=', len(all_clips))
    for ii, clip in enumerate(all_clips):
        vid = __read_clip(clip, img_shape, max_time_len)
        if vid is None:
            continue
        dump_path = os.path.join(write_out_path, clip.split('/')[-3], clip.split('/')[-2], clip.split('/')[-1]) + '.pkl'
        with open(dump_path, 'wb') as file:
            pickle.dump(vid, file)
        print('%d/%d %d dump %s finish' % (ii, len(all_clips), total_frames, dump_path))
    print('avg frames : %f' % (total_frames / len(all_clips)))


class UCF11DataSet(object):
    def __init__(self, data_path, max_time_len, img_shape, test_rate, batch_size, shuffle=True):
        """

        Parameters
        ----------
        data_path
        max_time_len
        img_shape : tuple or list
            [width, height, channel]
        test_rate : float
            0~1
        batch_size
        shuffle : True or False
        """
        self.data_path = data_path
        self.max_time_len = int(max_time_len)
        self.img_shape = np.asarray(img_shape)
        assert len(self.img_shape) == 3

        self.test_rate = float(test_rate)
        self.batch_size = int(batch_size)
        self.shuffle = shuffle

        self.all_clips, self.all_labels = self.scan_clips()
        self.data_train, self.data_test, self.labels_train, self.labels_test = self.split()

    def split(self):
        indxs = pickle.load(open(os.path.join(self.data_path, 'indices.pkl'), 'rb'))
        clips = self.all_clips[indxs]
        labels = self.all_labels[indxs]

        train_cnt = int(len(clips) * (1 - self.test_rate))

        data_train, data_test, labels_train, labels_test = \
            clips[:train_cnt], clips[train_cnt:], labels[:train_cnt], labels[train_cnt:]
        return data_train, data_test, labels_train, labels_test

    def scan_clips(self):
        res = []
        labels = []
        for ii, class_name in enumerate(classes):
            files = os.listdir(os.path.join(self.data_path, class_name))
            for this_collection in files:
                if this_collection == 'Annotation':
                    continue
                clips = os.listdir(os.path.join(self.data_path, class_name + '/' + this_collection))
                clips.sort()
                for this_clip in clips:
                    if not this_clip.endswith('pkl'):
                        continue
                    path = os.path.join(self.data_path, class_name + '/' + this_collection + '/' + this_clip)
                    res.append(path)
                    labels.append(ii)
        return np.asarray(res), np.asarray(labels)

    @threadsafe_generator
    def generate_train(self):
        while True:
            indexes = np.arange(len(self.data_train))
            if self.shuffle:
                np.random.shuffle(indexes)

            steps = int(len(indexes) / self.batch_size)
            for i in range(steps):
                choose_idxs = indexes[i * self.batch_size: (i + 1) * self.batch_size]
                X, y = self.read_from_disk(self.data_train, self.labels_train, choose_idxs)
                yield X, y

    @threadsafe_generator
    def generate_test(self):
        while True:
            indexes = np.arange(len(self.data_test))
            if self.shuffle:
                np.random.shuffle(indexes)

            steps = int(len(indexes) / self.batch_size)
            for i in range(steps):
                choose_idxs = indexes[i * self.batch_size: (i + 1) * self.batch_size]
                X, y = self.read_from_disk(self.data_test, self.labels_test, choose_idxs)
                yield X, y

    def read_from_disk(self, dataset, labels, choose_idxs):
        X = np.zeros((self.batch_size, self.max_time_len, int(np.prod(self.img_shape))))
        y = np.zeros((self.batch_size, len(classes)), dtype=int)

        # Generate data
        for ii, idx in enumerate(choose_idxs):
            with open(dataset[idx], 'rb') as file:
                X[ii] = pickle.load(file)
            y[ii, labels[idx]] = 1

        return X, y


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: %s input-folder output-folder' % sys.argv[0])
        exit(-1)
    data_path = sys.argv[1]
    write_out_path = sys.argv[2]
    max_time_len = 6
    preprocess_UCF11(data_path, write_out_path, max_time_len, (160, 120, 3))

    indxs = random.sample(range(0, 1600), 1600)
    indxs = np.asarray(indxs)
    pickle.dump(indxs, open(os.path.join(write_out_path, 'indices.pkl'), 'wb'))
