import imageio
import numpy as np
import matplotlib.pyplot as plt
from color_conversion import *

'''Log ou DoG ? DoG, plus simple (difference, des gaussien successifs... et plus rapide, aproximation suffisante)
   To do:  
    - Définir un Gaussien 1D. (La convolution 2D) est séparable en 2 convolution 1D (séparation des variables ? A verifier)
    - Ecrire une convolution 1D appliquant le gaussien a diferentes échelles. (Faut il ensuite rééchantilloner les images avec une fréquence plus faible ? Possible ...)
    - Réaliser les différences des gaussiens (DoG)
    - En déduire des points caractérqtiques :
        - Chercher les points d'annulation (contours, méthode du "zero crossing") (Prouver que les contours sont fermés (fonction continue de N2?) (Attention aux bords des images!)
        - Prendre les maximas/minimum pour chaque image de la pyramide ( voir Lindeberg )
        !!!!! TRouevr les minimiums qui correspondent à la reponse en amplitude la plus inmportante caracteirstique d'une certaine distance entre les bords. he oui jammy!
    - En déduire les frontieres entourant les blob ou les zones des blobs eux memes 
    - Profit ???
'''

def gauss1D( a, x ):
    ''' a : standard deviation '''
    return 1/(np.sqrt(2*np.pi)*a)*np.exp(-x*x/(2*a*a)) 
    
def gauss2D( a, x, y ):
    ''' a : standard deviation '''
    return 1/(2*np.pi*a*a)*np.exp(-(x*x+y*y)/(2*a*a))

def gaussKernel1D( a, horizontal=True, normalized=True):
    ''' Discrete approximation of the Gaussian kernel. The Gaussian is effectively zero more than 3 standard devation from the mean.
    The gaussian needs to be normalized (sum of all coef equals to 1) If not the image is brightened.'''
    
    size = int(np.ceil(a))
    temp = np.array( [[gauss1D(a,i) for i in range(-size,size+1)]] ) 
    
    if normalized : temp = temp/np.sum(temp)
    if horizontal : return temp
    return temp.T

def gaussKernel2D( a, normalized=True ):
    ''' Discrete approximation of the Gaussian kernel. The Gaussian is effectively zero more than 3 standard devation from the mean. '''
    
    size = int(np.ceil(a))
    temp = np.array( [[gauss2D(a,i,j) for i in range(-size,size+1)] for j in range(-size,size+1)] ) 
    
    if normalized : temp = temp/np.sum(temp)
    return temp

def convolute( array, kernel ):

    padding = np.array( kernel.shape ) -1
    N,M = padding//2
    temp = np.zeros( array.shape )
    for i in range( N, array.shape[0] -2*N-1 ):
        for j in range( M, array.shape[1] -2*M-1 ):
            temp[i+N,j+M] = np.sum( kernel * array[ i:i+2*N+1, j:j+2*M+1 ] )
            
    return temp

def doG( array, threshold=1, ratio=1.6, a0=0.9 ):
    
    temp1 = convolute( array, gaussKernel2D(a0) )
    temp2 = convolute( array, gaussKernel2D(ratio*a0) )
    temp = temp1-temp2
    temp[temp <= threshold]=1
    temp[temp > threshold]=0 
    return temp

def pyramid( array, height, threshold=1, a0=1, ratio=1.4 ):
    ''' ratio of 1.6 to approximate a laplacian of gaussian '''
    
    temp = array
    p1 = [array]
    p2 = []
    p3 = []
    
    for i in range(height):

        temp2 = convolute( temp, gaussKernel2D(a0*(ratio)**i) )
        p1.append(temp2)

        diff = temp-temp2
        p2.append(diff)

        '''
        t=threshold*(1)
        diff[(-t<=diff) & (diff<=0)]=0
        diff[(-t>diff) | (diff>t) ]=1  
        '''
        #diff[ diff<=0 ] = 0
        diff[diff>0] = 1
        p3.append(diff)
        
        temp = temp2
 
    return p1,p2,p3

def maxima( p_array ):
    ''' p2 '''
    ''' attention aux bords trompeurs qui ne doivent pas etre pris en compte!! '''
    ''' comparaisons avec les voisins sur le meme plan et sur tous les plan précedents et suivants 
        ne le faire que pour les valeures négatives (on ne le fait que pour les points interieurs '''
    
    

print(gaussKernel2D(0.5),"\n", 1/16*np.array([[1,2,1],[2,4,2],[1,2,1]]))
image = imageio.imread("../../assets/tree1.jpg")
image = rgb_to_ycbcr(image)[:,:,2]

h=8
p1,p2,p3 = pyramid( image, h, a0=1.5)
fig = plt.figure()
for i in range(h):
    ax=fig.add_subplot(h,1,i+1)
    ax.imshow(p1[i],vmin=0,vmax=255, cmap="gray")
fig=plt.figure()
for i in range(h):
    ax=fig.add_subplot(h,1,i+1)
    ax.imshow(p2[i],vmin=-i,vmax=i,cmap="PiYG")
fig=plt.figure()
for i in range(h):
    ax=fig.add_subplot(h,1,i+1)
    ax.imshow(p3[i],vmin=0,vmax=1,cmap="gray")
plt.show()

'''
fig=plt.figure()
ax=fig.add_subplot(2,1,1)
ax.imshow(image, cmap="gray")
image2 = doG( image)
ax=fig.add_subplot(2,1,2)
ax.imshow( image2, cmap="gray")
plt.show()
'''

'''
fig = plt.figure()
ax = fig.add_subplot(2,2,1)
ax.imshow(image,cmap="gray")

image2 = convolute( image, gaussKernel2D(1) )
ax = fig.add_subplot(2,2,2)
ax.imshow(image2,cmap="gray")

image2 = convolute( image, gaussKernel2D(2) )
ax = fig.add_subplot(2,2,3)
ax.imshow(image2,cmap="gray")

image2 = convolute( image, gaussKernel2D(3) )
ax = fig.add_subplot(2,2,4)
ax.imshow(image2,cmap="gray")

plt.show()
'''


