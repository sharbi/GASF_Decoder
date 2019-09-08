import numpy as np
from pyts.image import GramianAngularField
import h5py
import os

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
    for set in y:
        for values in set:
            patterns = list()
            for value in values:
                patterns.append(string_to_length(largest_out, value))
            Ystr.append(patterns)


    return Xstr, Ystr

def integer_encode(X, y, alphabet):
    char_to_int = dict((c, i) for i, c in enumerate(alphabet))
    Xenc = list()
    for pattern in X:
        integer_encoded = [char_to_int[char] for char in pattern]
        Xenc.append(integer_encoded)
    Yenc = list()
    for patterns in y:
        for pattern in patterns:
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

def generate_samples(n):
    alphabet = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', ' ', 'e']

    for _ in range(n):

        X = list()
        y = list()

        print("generating random input")
        random_input = np.random.randint(1, 300, (1000 ,1, 60))
        print("Input generated")


        image_size = 32
        print("Performing GASF on input")
        gasf = GramianAngularField(image_size, method='summation')
        X_gasf = np.array([gasf.fit_transform(x) for x in random_input])

        print(X_gasf.shape)

        print("Finished GASF")

        X_gasf = X_gasf.reshape(X_gasf.shape[0], 1, 1024)

        X, y = convert_to_string(X_gasf, random_input, 22, 3)
        X, y = np.array(X), np.array(y)
        X, y = integer_encode(X, y, alphabet)
        X, y = np.array(X), np.array(y)
        X, y = one_hot_encode(X, y, len(alphabet))
        X, y = np.array(X), np.array(y)

        print(X.shape)
        print(y.shape)

        out_directory = './data/'
        output_file = 'output.h5'
        input_file = 'input.h5'

        if not os.path.exists(out_directory + output_file):
            fout = h5py.File(out_directory + output_file, 'w')
            dset_out = fout.create_dataset("output", data=y, chunks=True, maxshape=(None, 1, 60, 14))

        else:
            fout = h5py.File(out_directory + output_file, 'a')
            fout['output'].resize((fout['output'].shape[0] + y.shape[0]), axis=0)
            fout['output'][-random_input.shape[0]:] = y


        if not os.path.exists(out_directory + input_file):
            fin = h5py.File(out_directory + input_file, 'w')
            dset_in = fin.create_dataset("input", data=X, chunks=True, maxshape=(None, 1, 1024, 14))

        else:
            fin = h5py.File(out_directory + input_file, 'a')
            fin['input'].resize((fin['input'].shape[0] + X.shape[0]), axis=0)
            fin['input'][-X_gasf.shape[0]:] = X



        fout.close()
        fin.close()

if __name__ == '__main__':
    generate_samples(250)
