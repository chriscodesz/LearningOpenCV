# 2d Convolution to get average value of neigbours
#used for noise reduction and smoothing

#allows us to manually change bluring of an image based on a tracker bar
#slide bar increases nxn kernerl size increasing the amount of avergae blur
#looses edges, but later codes will blur and preserve edges>>>median filtering

    #Gaussian Filtering → Best for general noise reduction, smooth blur, edge-aware smoothing, natural-looking blur.
    #Average Filtering → Only good for quick & simple smoothing, but often inferior to Gaussian.
    #Median Filtering → Best for removing salt-and-pepper noise while keeping edges sharp.
    #General Convolution → Flexible framework (Gaussian is one special kernel).

import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np

def callback():
    pass

def avergaefiltering():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath) 
    
    winName='avg filter'     #creates a window name string
    cv.namedWindow(winName)
    cv.createTrackbar('n',winName,1,100,callback) # tracker bar named n, minium 1 -100, callback function needed but wont do anything here

    height,width,_=img.shape 
    scale=1/4
    width=int(width*scale)
    height=int(height*scale)
    img=cv.resize(img,(width,height)) #changes size of the image to a quater

    while True:
        if cv.waitKey(1)== ord('q'):
            break
            
        n = cv.getTrackbarPos('n',winName) #reads in track bar value
        imgFilter = cv.blur(img,(n,n)) #filters image, using sqyare kernel n by n
        cv.imshow(winName,imgFilter)
    cv.destroyAllWindows()




if __name__=='__main__':
    avergaefiltering()