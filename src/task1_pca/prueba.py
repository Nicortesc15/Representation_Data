import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
import scipy.datasets
import scipy.misc
from skimage.transform import resize
import numpy.typing as npt
import utils

img = utils.load_resize_image()
img_t = img.T                    # Columns considered as data points
centered_img_t = utils.center_data(img_t)
U, S, Vt = utils.compute_svd(centered_img_t)
print(centered_img_t.shape)
print(Vt.shape)

# Display the image
plt.imshow(centered_img_t)
plt.title("Raccoon Face")
plt.axis("off")
plt.show()