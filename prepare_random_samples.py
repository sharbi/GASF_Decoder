import numpy as np
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

random_input = np.random.randint(1, 300, (1000000 ,4, 60))

image_size = 32
gasf = GramianAngularField(image_size, method='summation')
X_gasf = [gasf.fit_transform(x) for x in random_input]

pkl.dump(random_input, open('./data/output.pkl', 'wb'))
pkl.dump(X_gasf[0], open('./data/input.pkl', 'wb'))
