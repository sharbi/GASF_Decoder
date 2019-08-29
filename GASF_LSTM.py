import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, LSTM, Reshape, TimeDistributed, RepeatVector
from keras.models import Sequential
import numpy as np
import pandas as pd
import math
import h5py

def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)

def one_hot_decode(encoded_seq):
    return [np.argmax(vector) for vector in encoded_seq]

def to_supervised(seq, n_in, n_out):
    df = pd.DataFrame(seq)
    df = pd.concat([df.shift(n_in-i-1) for i in range(n_in)], axis=1)

    values = df.values
    width = seq.shape[1]
    X = values.reshape(len(values), n_in, width)
    y = values[:, 0:(n_out*width)].reshape(len(values), n_out, width)
    return X, y


if __name__ == '__main__':


    # define the hyperparameters
    epochs = 200
    batch_size = 256
    input_shape = (1, 1024)
    output_shape = (1, 60)
    latent_dim = 150

    n_in = 1024
    n_out = 60


    # define the model
    decoder = Sequential()
    decoder.add(LSTM(latent_dim, batch_input_shape=(batch_size, 1, n_in), stateful=True))
    decoder.add(RepeatVector(1))
    decoder.add(LSTM(latent_dim, return_sequences=True, stateful=True))
    decoder.add(TimeDistributed(Dense(60)))

    decoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Load the data
    fin = h5py.File('./data/input.h5','r')
    fout = h5py.File('./data/output.h5', 'r')

    X_input = np.array(list(fin['input']))
    X_input = X_input.reshape(X_input.shape[0], 1, 1024)
    y = np.array(list(fout['output']))

    fin.close()
    fout.close()

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
