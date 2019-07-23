from __future__ import print_function

from collections import defaultdict

import tensorflow as tf
import keras.backend as K
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils.generic_utils import Progbar
import numpy as np
import os
import random as rn
import argparse
import time
import math
import glob
import h5py

def build_decoder(input_shape):

    print("Decoder")

    decoder = Sequential()
    # (4, 32, 32)
    decoder.add(Conv2D(32, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 16, 16)
    decoder.add(Conv2D(64, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 8, 8)
    decoder.add(Conv2D(128, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 4, 4)
    decoder.add(Conv2D(256, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 2, 2)
    decoder.add(Conv2D(512, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 1, 1)
    decoder.add(Conv2D(1024, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))

    decoder.add(Flatten())

    decoder.add(Dense(1024))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(Dense(1024, activation='relu'))

    gasf_input = Input(4, 32, 32)
    features = decoder(gasf_input)

    output = Dense((4, 60), activation='linear', name='decode')(features)

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--clip_value", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--prefix", default='')
    parser.add_argument("--seed", type=int, default="123")
    args = parser.parse_args()

    print(args)
    epochs = args.epochs
    batch_size = args.batch_size

    # setting seed for reproducibility
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    rn.seed(args.seed)

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    adam_lr = args.lr
    adam_beta_1 = 0.5

    decoder = build_decoder(input_shape=(4, 32, 32))

    decoder.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1), loss='meab_squared_error')

    fin = h5py.File('./data/input.h5','r')
    fout = h5py.File('./data/output.h5', 'r')

    X_input = np.array(list(fin['input']))
    y = np.array(list(fout['output']))

    fin.close()
    fout.close()

    training_size = math.floor(0.75 * X_input.shape[0])

    X_train = X_input[:training_size]
    X_test = X_input[training_size:]

    y_train = y_input[:training_size]
    y_test = y_input[training_size:]

    num_train, num_test = X_train.shape[0], X_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    # Set session details and placeholders for privacy accountant
    sess = K.get_session()

    for epoch in range(epochs):
        print('Epoch {} of {}'.format(epoch, epochs))

        num_batches = num_train // batch_size
        progress_bar = Progbar(target=num_batches)

        epoch_decoder_loss = []

        # print(gamma)

        train_start_time = time.clock()
        for index in range(num_batches):
            progress_bar.update(index)

        epoch_decoder_loss.append(decoder.train_on_batch(X_train, y_train))

        print('\n Train time: ', time.clock() - train_start_time)

        decoder_test_loss = decoder.evaluate(X_test, y_test, verbose=False)

        decoder_train_loss = np.mean(np.array(epoch_decoder_loss), axis=0)

        train_history['decoder'].append(decoder_train_loss)

        test_history['decoder'].append(decoder_test_loss)

        print('{0:<22s} | {1:4s}'.format(
            'component', *decoder.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f}'
        print(ROW_FMT.format('decoder (train)',
                             *train_history['decoder'][-1]))
        print(ROW_FMT.format('decoder (test)',
                             *test_history['decoder'][-1]))
