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

    padding = np.array( kernel.shape )
    N,M = padding//2
    X,Y = array.shape
    temp = np.zeros( (X,Y) )

    for i in range( N+1, X-N ):
        for j in range( M+1, Y-M ):
            temp[i,j] = np.sum( kernel * array[ i-N:i+N+1, j-N:j+M+1 ] )
            
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

        temp2 = convolute( array, gaussKernel2D(a0*(ratio)**i) )
        p1.append(temp2)

        diff = temp-temp2
        p2.append(diff)

        diff[diff>0] = 1
        p3.append(diff)
        
        temp = temp2
 
    return p1,p2,p3



def maxima( array, height, threshold=1, a0=1, ratio=1.4 ):
    ''' p2 '''
    ''' attention aux bords trompeurs qui ne doivent pas etre pris en compte!! '''
    ''' comparaisons avec les voisins sur le meme plan et sur tous les plan précedents et suivants 
        ne le faire que pour les valeures négatives (on ne le fait que pour les points interieurs ''' 

    N = int( np.ceil(a0*(ratio)**(height-1)))
    X,Y = np.shape( array ) 
    p = np.empty((height,X,Y))
    temp = array 

    for i in range(height):

        temp2 = convolute( array, gaussKernel2D(a0*(ratio)**i) )
        p[i] = temp-temp2
        temp = temp2
    ''' 
    m = np.array( [ [ np.argmin(p[:,i,j]) for j in range(N+1,Y-N) ] for i in range(N+1,X-N) ] ) 
    n = np.empty((X-2*N,Y-2*N))
   
    for x in range(X-2*N-1):
        for y in range(Y-2*N-1):

            d = m[x,y]
            v = p[d,N+1+x,N+1+y]
            #if v == np.min(p[d,N+x:N+x+3,N+y:N+y+3]) : n[x,y] = d
            #else : n[x,y] = -1
            n[x,y] = v
    '''
    m = np.array( [ [ np.argmin(p[:,i,j]) for j in range(Y) ] for i in range(X) ] ) 
    n = np.empty((X,Y))
   
    for x in range(1,X-1):
        for y in range(1,Y-1):

            d = m[x,y]
            v = p[d,x,y]
            if ( v == np.min(p[:,x-1:x+1,y-1:y+1])) : n[x,y] = d
            else : n[x,y] = -1
            
    return p,m,n
     



image = imageio.imread("../../assets/testing2.png")
image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]) #Super duper sweet!
X,Y = np.shape( image  ) 
h=8
a=3
p,m,n = maxima(image,h, a0=a)


fig= plt.figure()
ax=fig.add_subplot(1,3,1)
ax.imshow(n,cmap="binary")
ax=fig.add_subplot(1,3,2)
ax.imshow(m,cmap="binary")
ax=fig.add_subplot(1,3,3)
ax.imshow(image, cmap="gray")

for x in range(1,X-1):
    for y in range(1,Y-1):
        a = n[x,y]
        if a > 1 and a<5 :
            c = plt.Circle((x,y),a,color='b',fill=False)
            ax.add_artist(c)

p1,p2,p3 = pyramid( image, h, a0=a)
fig = plt.figure()
for i in range(h):
    ax=fig.add_subplot(h,1,i+1)
    ax.imshow(p1[i],vmin=0,vmax=255, cmap="gray")
fig=plt.figure()
for i in range(h):
    ax=fig.add_subplot(h,1,i+1)
    ax.imshow(p2[i],cmap="PiYG")
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


