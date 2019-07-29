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
    decoder.add(Conv2D(32, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 8, 8)
    decoder.add(Conv2D(64, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 4, 4)
    decoder.add(Conv2D(64, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 2, 2)
    decoder.add(Conv2D(128, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))
    decoder.add(Dropout(0.3))
    decoder.add(BatchNormalization())
    # (4, 1, 1)
    decoder.add(Conv2D(256, 3, padding='same', strides=1, input_shape=input_shape))
    decoder.add(LeakyReLU(0.2))

    decoder.add(Flatten())

    decoder.add(Dense(1024, activation='relu'))

    gasf_input = Input(shape=(4, 32, 32))
    features = decoder(gasf_input)

    dense1 = Dense(240, activation='linear', name='decode')(features)
    output = Reshape((4, 60))(dense1)

    return Model(gasf_input, output)

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

    decoder.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                    loss='categorical_crossentropy', metrics=['accuracy'])

    fin = h5py.File('./data/input.h5','r')
    fout = h5py.File('./data/output.h5', 'r')

    X_input = np.array(list(fin['input']))
    y = np.array(list(fout['output']))

    fin.close()
    fout.close()

    training_size = math.floor(0.75 * X_input.shape[0])

    X_train = X_input[:training_size]
    X_test = X_input[training_size:]

    y_train = y[:training_size]
    y_test = y[training_size:]

    num_train, num_test = X_train.shape[0], X_test.shape[0]

    epoch_decoder_loss = []


    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True


    num_batches = num_train // batch_size

    for epoch in range(epochs):
        for i in range(num_batches):
            input_batch = X_train[i * batch_size: i+1 * batch_size]
            output_batch = y_train[i * batch_size: i+1 * batch_size]
            epoch_decoder_loss.append(decoder.train_on_batch(input_batch, output_batch))
            output = decoder
            print(output)
        decoder_loss = np.mean(np.array(epoch_decoder_loss), axis=0)

        print("Train loss"):
        print(decoder_loss)

    #decoder.fit(X_train, y_train,
    #            batch_size=batch_size,
    #            epochs=epochs,
    #            verbose=1,
    #            validation_data=(X_test, y_test))

    score = decoder.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy", score[1])
