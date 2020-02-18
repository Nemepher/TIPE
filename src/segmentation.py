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

def grayscale_segmentation(im, prec):

    range = np.linespace(0,255,prec)
    m=mean(im)
    f= np.vectorize(lambda x : 255 if x>=m else 0)
    def f (x,v,p):
        if x>=v:
            return 
        else : 
            return 
    return f(im)
