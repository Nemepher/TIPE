import numpy as np
from utils import *

def convolution(im, ker, borders=0):

    padding = (np.array(ker.shape)-1)
    N,M = padding//2

    #smaller image to apply the kernel to (no copying)
    temp = np.zeros(im.shape)
    for i in range(N,im.shape[0]-N-1):
        for j in range(M,im.shape[1]-M-1):
                temp[i,j]=np.sum(ker*im[i:i+2*N+1,j:j+2*M+1])
    return temp

    #larger image to apply the kernel to
    '''
    temp = borders*np.ones(np.shape(im)+padding)
    temp[N:-N,M:-M]=im
    return np.array([ [ np.sum(ker*temp[ i:i+2*N+1, j:j+2*M+1]) for j  in range(np.shape(im)[1]) ] for i in range(np.shape(im)[0]) ])
    '''

def outline(im, threshold=5000) :

    sobel=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    Y=convolution(im,sobel)
    X=convolution(im,sobel.T)  #Transpose
    G = X**2 + Y**2 #Gradient

    f = lambda x : True if x>threshold else False
    f=np.vectorize(f)
    return np.uint8(f(G))

def grayscale_segmentation(im, level=2):
    m=mean(im)
    v=min(255-m,m)/level
    values=np.linspace(m-v,m+v,2*level)
    print(m,values) 
    f = np.vectorize(lambda x : dicho_search_nearest(values, x))
    return f(im)
