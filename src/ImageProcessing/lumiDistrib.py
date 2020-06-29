import numpy as np

def luminance_distribution( im ) :
    '''Expect a b́&w image and return an histogram of luminance'''
    l,c = np.shape(im)
    hist = np.array([0]*256)
    for i in range(l) :
        for j in range(c) :
            hist[ im[i,j] ] += 1
    return hist/(l*c) 



