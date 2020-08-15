import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks 
from utils import *
from color_conversion import *
from segmentation import *
from luminance import * 

image=imageio.imread("../../assets/tree4.jpg")

fig = plt.figure()
image=imageio.imread("../../assets/tree4.jpg")

image2 = rgb_to_ycbcr(image)[:,:,2]
ax = fig.add_subplot(2,2,1)
ax.imshow(image2,cmap="gray")

t= luminance_distribution(image2)
a,d = find_peaks(t, height=0.02, distance=5)
print(a,d)
ax = fig.add_subplot(2,2,2)
ax.plot(t)

image3 = threshold( image2, 127+1)
ax = fig.add_subplot(2,2,3)
ax.imshow(image3,cmap="gray")

image4 = dilation( erosion( image3 ) )
ax = fig.add_subplot(2,2,4)
ax.imshow(image4,cmap="gray")
plt.show()


'''
image2 = rgb_to_ycbcr(image)[:,:,2]
plt.imshow(image2,cmap="gray")
plt.show()
image3 = median_blur( image2)
plt.imshow(image3,cmap="gray")
plt.show()
'''

'''
image2 = rgb_to_ycbcr(image)[:,:,2]
#imageio.imwrite("gray.png", image2[:,:,0])
#plt.imshow(image2,cmap="gray")
h = luminance_distribution( image2 )
plt.bar(np.arange(0,256),h, width=2)
plt.show()
'''

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
