import numpy as np
from pyts.image import GramianAngularField

random_input = np.random.randint(1, 300, (1000000 ,4, 60))

image_size = 32
gasf = GramianAngularField(image_size, method='summation')
X_gasf = [gasf.fit_transform(x) for x in random_input]
gadf = GramianAngularField(image_size, method='difference')
X_gadf = [gadf.fit_transform(x) for x in random_input]


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
