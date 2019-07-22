import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle as pkl

print("generating random input")
random_input = np.random.randint(1, 300, (100 ,4, 60))
print("Input generated")


image_size = 32
print("Performing GASF on input")
gasf = GramianAngularField(image_size, method='summation')
X_gasf = np.array([gasf.fit_transform(x) for x in random_input])

print(X_gasf.shape)

print("Finished GASF")

np.save('./data/output.npy', random_input)
np.save('./data/input_1.pkl', X_gasf[0])
