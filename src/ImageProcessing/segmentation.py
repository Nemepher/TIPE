import numpy as np
from utils import *

def outline ( im, threshold=5000 ) :

    sobel = np.array( [[-1,-2,-1],[0,0,0],[1,2,1]] )
    Y = convolution( im, sobel )
    X = convolution( im, sobel.T ) #Transpose
    G = X**2 + Y**2 #Gradient

    f = lambda x : True if x>threshold else False
    f = np.vectorize(f)
    return np.uint8( f(G) )


def grayscale_segmentation ( im, level=2 ):
    
    step = 255//level
    return step*(im//step)


def threshold( im, th ):

    f = np.vectorize( lambda x : 255 if x<th else 0 )
    return f(im)


def median_blur ( im, size=3, iterations=1 ):
    '''size must be odd '''

    pad = (size-1)//2
    temp = np.pad( im, pad_width=pad, mode='constant', constant_values=255 )
    #to do : set a t one if b&w image!!!!
    blur = np.zeros( im.shape )

    for i in range ( 0, im.shape[0] ):
        for j in range ( 0, im.shape[1] ):
            blur[i,j] = np.median( temp[ i:i+2*pad+1, j:j+2*pad+1] )
    
    if iterations == 1 : return blur
    return median_blur( blur, size, iterations-1 )



#Transformations morphologiques

def erosion ( im, size=3, iterations=1 ):
    '''size must be odd '''

    pad = (size-1)//2
    temp = np.pad( im, pad_width=pad, mode='constant', constant_values=255 )
    #to do : set a t one if b&w image!!!!
    erode = np.zeros( im.shape )

    for i in range ( 0, im.shape[0] ):
        for j in range ( 0, im.shape[1] ):
            erode[i,j] = np.min( temp[ i:i+2*pad+1, j:j+2*pad+1] )
    
    if iterations == 1 : return erode
    return erosion( erode, size, iterations-1 )


def dilation ( im, size=3, iterations=1 ):
    '''size must be odd '''

    pad = (size-1)//2
    temp = np.pad( im, pad_width=pad, mode='constant', constant_values=255 )
    dilate = np.zeros( im.shape )

    for i in range ( 0, im.shape[0] ):
        for j in range ( 0, im.shape[1] ):
            dilate[i,j] = np.max( temp[ i:i+2*pad+1, j:j+2*pad+1] )
    
    if iterations == 1 : return dilate
    return dilation( dilate, size, iterations-1 )

