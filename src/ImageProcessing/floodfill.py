import numpy as np 
import imageio
import matplotlib.pyplot as plt
 

def floodfill(array, severity,  outcol, x, y) :
    incol = np.copy(array[x,y])
    shape = np.shape(array)
    margin = [severity]*3
    
    def aux(x,y): 
        col = array[x,y]
        if (np.abs((np.int16(col) - np.int16(incol))) <= margin ).all() : 
            array[x,y] = outcol
            x,y = min(max(1,x),shape[1]-1), min(max(1,y),shape[0]-1)
            aux(x,y+1); aux(x,y-1); aux(x+1,y); aux(x-1,y)
        return
    aux(x,y)

def middle(array, severity,  outcol, x, y) :
    incol = np.copy(array[x,y])
    margin= [severity]*3
    insideX = []
    insideY = []
    def aux(x,y): 
        col = array[x,y]
        if (np.abs((np.int16(col) - np.int16(incol))) <= margin ).all() : 
            insideX.apend(x)
            insideY.append(y)
            aux(x,y+1); aux(x,y-1); aux(x+1,y); aux(x-1,y)
    aux(x,y)
    return [np.mean(insideX), np.mean(insideY)]


image = (np.array(imageio.imread("assets/bw.png")))
plt.imshow(image)
plt.show()

floodfill(image, 0,  (0,0,0), 20,290)
plt.imshow(image)
plt.show()
