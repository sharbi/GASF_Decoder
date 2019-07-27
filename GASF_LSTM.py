import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Dense, LSTM, Reshape, TimeDistributed
import numpy as np
import h5py

if __name__ == '__main__':


    # define the hyperparameters
    epochs = 200
    batch_size = 256
    input_shape = (4, 1024)
    output_shape = (4, 60)
    latent_dim = 256

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    num_batches = num_train // batch_size

    decoder = Sequential()
    decoder.add(LSTM(latent_dim, input_shape=(4, 1024), return_sequences=True))
    decoder.add(LSTM(latent_dim, return_sequences=True))
    decoder.add(TimeDistributed(Dense(60)))

    decoder.fit(X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(X_test, y_test))

    score = decoder.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy", score[1])
