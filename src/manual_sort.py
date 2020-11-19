### A script to help manualy sorting a folder of images into categories ###

import os
import glob
import shutil 
import imageio
import getch
from matplotlib import pyplot as plt

folder = "../assets/processed_data/test1/"
subf = ['a','z'] 
done = False 

try :
    os.mkdir(folder+"sorted/")
    for sf in subf : 
        os.mkdir(folder+"sorted/"+sf)
except FileExistsError as exc:
    print("already done, go on ? y/n")
    if input()!="y" : done = True

if not done :
    i=0
    for fpath in glob.glob(folder+"*.png"):
        
        img = imageio.imread(fpath)
        plt.imshow(img)
        plt.show(block=False)
        plt.pause(0.1) 
        cat = getch.getch()
        plt.close()
        
        src = fpath
        dst = folder+"sorted/"+str(cat)+"/"+str(i)+".png"
        shutil.copy(src,dst)

        i+=1
