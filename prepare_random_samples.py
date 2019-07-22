import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from mpl_toolkits.axes_grid1 import ImageGrid
import pickle as pkl

random_input = np.random.randint(1, 300, (1000000 ,4, 60))

image_size = 32
gasf = GramianAngularField(image_size, method='summation')
X_gasf = [gasf.fit_transform(x) for x in random_input]

pkl.dump(random_input, open('./data/output_1.pkl', 'wb'))
pkl.dump(X_gasf[0], open('./data/input_1.pkl', 'wb'))
