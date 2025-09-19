# A graph that shows the distribution of data fro 2 variables, brighter the colour the highier the distribution
# good fro data visualization, feature extraction, segmentation, image comparison




import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def histogram2D():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/cat.jpg')
    img = cv.imread(imgPath) 
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    hsv =cv.cvtColor(img,cv.COLOR_BGR2HSV)

    hist = cv.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256]) #max ranges 180 and 256 for hue and saturation

    plt.figure()
    plt.subplot(131)
    plt.imshow(imgRGB)
    plt.subplot(132)
    plt.imshow(hist)
    plt.xlabel('Hue')
    plt.ylabel('Saturation')

    lowerBound = np.array([15,80,0])
    upperBound = np.array([25,120,255])
    mask=cv.inRange(hsv, lowerBound, upperBound) # should mask out what values are within this HSV range weve determined
    
    plt.subplot(133)
    plt.imshow(mask,cmap='gray')


    plt.show()


if __name__=='__main__':
    histogram2D()