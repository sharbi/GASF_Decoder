import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle as pkl

print("generating random input")
random_input = np.random.randint(1, 300, (1000000 ,4, 60))
print("Input generated")


image_size = 32
print("Performing GASF on input")
gasf = GramianAngularField(image_size, method='summation')
X_gasf = np.array((1000000, (4, 32, 32)))
for i, x in enumerate(random_input):
    X_gasf[i] = gasf.fit_transform(x)

print("Finished GASF")

pkl.dump(random_input, open('./data/output_1.pkl', 'wb'))
pkl.dump(X_gasf[0], open('./data/input_1.pkl', 'wb'))
