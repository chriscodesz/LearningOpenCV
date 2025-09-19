import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pa


def readAndWriteSinglePixel(): #chnages the colour of a pixel on an image
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath)
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB) #breaks up image into pixel values, switches to RGB

    plt.figure()
    plt.imshow(imgRGB) #adds image to a plot
    plt.show()

    eyePixel = imgRGB[1021,1956] #collects RGB value at loctaion x=1956,y=1021, eye pixel locations, ROWS,COLUMNS, so [y,x]
    
    imgRGB[1021,1956]=(255,0,0) #change the pixel at x=1956,y=1021 to bright red RGB

    plt.figure() #before and after pixel edit chnage, need to zoom in to see
    plt.imshow(imgRGB) #adds image to a plot
    plt.show()



def readAndWritePixelRegion(): #selects an area of an image and writes it to another section
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath)
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB) #adds image to a plot
    plt.show()

    eyeRegion = imgRGB[979:1082,1847:2062]

    dx = 1082-979
    dy = 2062-1847
    
    startX= 746
    startY= 2068

    imgRGB[startX:startX+dx, startY:startY+dy] = eyeRegion

    plt.figure()
    plt.imshow(imgRGB) #adds image to a plot
    plt.show()
    



if __name__=='__main__':
    #readAndWriteSinglePixel()
    readAndWritePixelRegion()
