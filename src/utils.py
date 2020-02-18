import numpy as np

def rgb_to_ycbcr (im):

    input = np.float64(np.array(im)) #RGB

    #YCbCR #Can be done with a dot product
    output=np.empty_like(input)
    output[:,:,0] = 0.299*input[:,:,0] + 0.587*input[:,:,1] + 0.114*input[:,:,2] #Y
    output[:,:,1] = -0.1687*input[:,:,0] - 0.3313*input[:,:,1] + 0.5*input[:,:,2] +128 #Cb
    output[:,:,2] = 0.5*input[:,:,0] - 0.4187*input[:,:,1] - 0.0813*input[:,:,2] +128 #Cr
    return np.uint8(output)

def ycbcr_to_rgb (im):

    input = np.float64(np.array(im)) #YCbCr #the int to float conversion is essential !

    #RGB #Can be done with a dot product
    output=np.empty_like(input)
    output[:,:,0] = input[:,:,0] + 1.402*(input[:,:,2]-128) #R
    output[:,:,1] = input[:,:,0] - 0.34414*(input[:,:,1]-128) - 0.71414*(input[:,:,2]-128) #G
    output[:,:,2] = input[:,:,0] + 1.772*(input[:,:,1]-128) #B
    return np.uint8(output)

def mean (im):

    input = np.float64(np.array(im))
    m = np.sum(input)//np.size(im)
    return m
