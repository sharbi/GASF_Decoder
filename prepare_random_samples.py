import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from mpl_toolkits.axes_grid1 import ImageGrid
import h5py

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

fout = h5py.File(out_directory + output_file, 'a')
fin = h5py.File(out_directory + input_file, 'a')

dset_out = fout.create_dataset("output", data=random_input, maxshape=(None,))
dset_in = fin.create_dataset("input", data=X_gasf[0], maxshape=(None,))
