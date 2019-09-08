import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, LSTM, Reshape, TimeDistributed, RepeatVector
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
    decoder.add(LSTM(latent_dim, input_shape=(1024, 23, 14)))
    decoder.add(RepeatVector(60))
    decoder.add(LSTM(latent_dim, return_sequences=True))
    decoder.add(TimeDistributed(Dense(14)))

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




    for epoch in range(epochs):
        decoder.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=1,
                verbose=1,
                validation_data=(X_test, y_test))
        decoder.reset_states()

    score = decoder.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy", score[1])
