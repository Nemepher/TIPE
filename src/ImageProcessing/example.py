import imageio
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from segmentation import *
from lumiDistrib import * 

image=imageio.imread("../../assets/foret.png")


image2 = rgb_to_ycbcr(image)[:,:,2]
#imageio.imwrite("gray.png", image2[:,:,0])
#plt.imshow(image2,cmap="gray")
h = luminance_distribution( image2 )
plt.bar(np.arange(0,256),h, width=2)
plt.show()

'''
image3 = grayscale_segmentation(image2,2)
plt.imshow(image3, cmap="gray")
plt.axis('off')
plt.show()
'''

'''
contours=outline(image2,8000)
plt.imshow(contours)
plt.axis('off')
plt.show()
'''

'''
ker = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
conv=convolution(image2,ker)
plt.imshow(conv)
plt.axis('off')
plt.show()
'''

'''
image3 =ycbcr_to_rgb(image2)
#imageio.imwrite("test2.png", image3)
plt.imshow(image3)
plt.axis('off')
plt.show()
'''
