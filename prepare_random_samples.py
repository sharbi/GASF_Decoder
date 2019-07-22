import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from mpl_toolkits.axes_grid1 import ImageGrid
import h5py
import os

print("generating random input")
random_input = np.random.randint(1, 300, (1000 ,4, 60))
print("Input generated")


image_size = 32
print("Performing GASF on input")
gasf = GramianAngularField(image_size, method='summation')
X_gasf = np.array([gasf.fit_transform(x) for x in random_input])

print(X_gasf.shape)

print("Finished GASF")

out_directory = './data/'
output_file = 'output.h5py'
input_file = 'input.h5py'

if not os.path.exists(out_directory + output_file):
    fout = h5py.File(out_directory + output_file, 'w')
    dset_out = fout.create_dataset("output", data=random_input, maxshape=(None, 4, 60))

else:
    fout = h5py.File(out_directory + output_file, 'a')
    fout['output'].resize((fout['output'].shape[0] + random_input.shape[0]), axis=0)
    fout['output'][-random_input.shape[0]:] = random_input


if os.path.exists(out_directory + input_file):
    fin = h5py.File(out_directory + input_file, 'w')
    dset_in = fin.create_dataset("input", data=X_gasf[0], maxshape=(None, 4, 32, 32))

else:
    fin = h5py.File(out_directory + input_file, 'a')
    fin['input'].resize((fin['input'].shape[0] + X_gasf[0].shape[0]), axis=0)
    fin['input'][-X_gasf[0].shape[0]:] = X_gasf[0]



fout.close()
fin.close()
