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



def string_to_length(max_len, strin):
    strout = ''
    strin = str(strin)
    if len(strin) < max_len:
        strout = ''.join([' ' for _ in range(max_len - len(strin))]) + strin
    return strout

def convert_to_string(X, y, largest_in, largest_out):
    print("Input into string...")
    Xstr = [string_to_length(largest_in, number) for x in X for input in x for number in input]
    print("Example output:")
    print(Xstr[0])
    Ystr = list()
    for i, set in enumerate(y):
        progress_bar = Progbar(target=len(y))
        for values in set:
            patterns = list()
            for value in values:
                patterns.append(string_to_length(largest_out, value))
            Ystr.append(patterns)
        progress_bar.update(i + 1)


    return Xstr, Ystr

def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for i, pattern in enumerate(X):
        progress_bar = Progbar(target=len(Y))
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
        progress_bar.update(i+1)
    Yenc = list()
    for pattern in y:
        integer_encoded = [char_to_int[char] for char in pattern]

    return Xenc, Yenc

def one_hot_encode(X, y, max_int):
    Xenc = list()
    for seq in X:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Xenc.append(pattern)

    Yenc = list()
    for seq in y:
        pattern = list()
        for index in seq:
            vector = [0 for _ in range(max_int)]
            vector[index] = 1
            pattern.append(vector)
        Yenc.append(pattern)
    return Xenc, Yenc

# invert encoding
def invert(seq, alphabet):
	int_to_char = dict((i, c) for i, c in enumerate(alphabet))
	strings = list()
	for pattern in seq:
		string = int_to_char[argmax(pattern)]
		strings.append(string)
	return ''.join(strings)

if __name__ == '__main__':

    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', ' ', 'e']

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
    decoder.add(LSTM(latent_dim, input_shape=(22, 12)))
    decoder.add(RepeatVector(3))
    decoder.add(LSTM(latent_dim, return_sequences=True))
    decoder.add(TimeDistributed(Dense(12)))

    decoder.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load the data
    fin = h5py.File('./data/input.h5','r')
    fout = h5py.File('./data/output.h5', 'r')

    print("Loading data into array.")

    X_input = np.array(list(fin['input']))
    X_input = X_input.reshape(X_input.shape[0], 1, 1024)
    y = np.array(list(fout['output']))

    fin.close()
    fout.close()

    print("Data loaded.")

    print("Converting int to string.")
    X_input, y = convert_to_string(X_input, y, largest_input, largest_output)
    print("Converted, encoding to index and one_hot.")
    X_input, y = integer_encode(X_input, y, alphabet)
    X_input, y = one_hot_encode(X_input, y, len(alphabet))

    print("All data modification completed.")

    X_input, y = np.array(X_input), np.array(y)

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
