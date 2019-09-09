import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, ConvLSTM2D, LSTM, Reshape, TimeDistributed, RepeatVector
from keras.layers import AveragePooling3D
from keras.models import Sequential
import numpy as np
import pandas as pd
import math
import h5py
from functools import partial
from keras.utils.generic_utils import Progbar





# invert encoding
def invert(seq, alphabet):
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	strings = list()
	for pattern in seq:
		string = int_to_char[argmax(pattern)]
		strings.append(string)
	return ''.join(strings)

if __name__ == '__main__':


    # define the hyperparameters
    epochs = 200
    batch_size = 250
    output_shape = (1, 60)
    latent_dim = 150

    largest_input = 22
    largest_output = 3

    n_in = 1024
    n_out = 60


    # define the model
    decoder = Sequential()
    decoder.add(ConvLSTM2D(latent_dim, 2, input_shape=(1024, 1, 23, 14), data_format='channels_first', return_sequences=True))
    decoder.add(ConvLSTM2D(latent_dim, 2, return_sequences=True))
    decoder.add(AveragePooling3D((1, 3, 14)))
    decoder.add(Reshape((-1, 60)))

    decoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the data
    fin = h5py.File('./data/input.h5','r')
    fout = h5py.File('./data/output.h5', 'r')

    print("Loading data into array.")

    X_input = np.array(list(fin['input']))
    y = np.array(list(fout['output']))

    fin.close()
    fout.close()

    print("Data loaded.")

    X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 1, X_input.shape[2], X_input.shape[3]))
    y = y.reshape((y.shape[0], y.shape[1], 1, y.shape[2], y.shape[3]))

    training_size = math.floor(0.75 * X_input.shape[0])

    X_train = X_input[:training_size]
    X_test = X_input[training_size:]

    y_train = y[:training_size]
    y_test = y[training_size:]

    num_train, num_test = X_train.shape[0], X_test.shape[0]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    num_batches = num_train // batch_size
    epoch_decoder_loss = []




    decoder.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test))

    score = decoder.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy", score[1])
