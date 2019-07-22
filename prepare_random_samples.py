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
X_gasf = [gasf.fit_transform(x) for x in random_input]
print("Finished GASF")

pkl.dump(random_input, open('./data/output_1.pkl', 'wb'))
pkl.dump(X_gasf[0], open('./data/input_1.pkl', 'wb'))
