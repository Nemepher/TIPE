import imageio
import numpy as np
import matplotlib.pyplot as plt
from CS_transform import *

image=imageio.imread("../assets/boeing-avion.jpg")
image2 = rgb_to_ycbcr(image)
#imageio.imwrite("test.png", image2[:,:,0])
plt.imshow(image2)
plt.axis('off')
plt.show()

image3 =ycbcr_to_rgb(image2)
#imageio.imwrite("test2.png", image3)
plt.imshow(image3)
plt.axis('off')
plt.show()
