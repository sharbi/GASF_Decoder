import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField

# Parameters
n_samples, n_features = 60, 1

def make_sin_values(quantity):
    array = np.array((0., 15., 30., 45., 60., 90., 60., 45., 30., 15.))
    total_array = np.tile(array, int(quantity/len(array)))
    values = np.sin(total_array * np.pi / 180. )
    print(values)
    return values


input = make_sin_values(n_samples)

input = input.reshape(1, -1)

image_size = 32
gasf = GramianAngularField(image_size, method='summation')
X_gasf = gasf.fit_transform(input)
gadf = GramianAngularField(image_size, method='difference')
X_gadf = gadf.fit_transform(input)

print(X_gasf.shape)

# Show the images for the first time series
fig = plt.figure(figsize=(12, 7))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(1, 2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.3,
                 )
images = [X_gasf[0], X_gadf[0]]
titles = ['Gramian Angular Summation Field',
          'Gramian Angular Difference Field']
for image, title, ax in zip(images, titles, grid):
    im = ax.imshow(image, cmap='rainbow', origin='lower')
    ax.set_title(title)
ax.cax.colorbar(im)
ax.cax.toggle_label(True)
plt.show()
