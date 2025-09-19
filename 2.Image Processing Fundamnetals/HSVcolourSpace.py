#HSV colour space is Hue (type of colour) between 0-180 for OpenCV but also 0-360, 
# saturation(concentration) between 0-255, white (diluted) or strong coloured?
# and Value (intensity of colour) between 0-255, light or dark?

# need for segmentation of images using colour, 
# more of an intuative way of describing colours also



import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np


def hsvColorSegmentation():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath) #gets image
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB) #seperates image to BGR values, but still as normal image
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV) #seperates to HSV

    lowerBound=np.array([0,0,50]) #HSV Value lower bound for what we want is 50 ish
    upperBound=np.array([20,120,100])
    mask = cv.inRange(hsv, lowerBound, upperBound) #creates a mask area that includes colors between our upper and lower bound HSV colors


    plt.figure()
    plt.imshow(imgRGB)

    plt.show()
    
    cv.imshow('mask',mask)
    cv.waitKey(0)



if __name__=='__main__':
    hsvColorSegmentation()