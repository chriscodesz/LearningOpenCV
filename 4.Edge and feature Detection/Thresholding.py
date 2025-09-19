# image processing that converts the image to binary (black and white)
# used for segmenation, feature extraction and object detection
# couple of types of thresholding, binary, trunc,toero and all have inverse
# need a threshold value, can get from a histogram, could be where peaks split


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def Thresholding():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath) 
    imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #creates gray scale image

    hist=cv.calcHist([imgGray],[0],None,[256],[0,256]) #create histogram to find a good threshold
    plt.figure()
    plt.subplot(121)
    plt.plot(hist)
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')
    plt.subplot(122)
    plt.imshow(imgGray) #plots gray scale iimage, but not a garyscale
    plt.show()

    thresOpt=[cv.THRESH_BINARY,cv.THRESH_BINARY_INV,cv.THRESH_TOZERO,cv.THRESH_TOZERO_INV,cv.THRESH_TRUNC] # list of thresholding options
    thresNames = ['Binary','Binary Inverted','ToZero','TpZero Inv', 'Trunc']

    plt.figure()
    plt.subplot(231)
    plt.imshow(imgGray,cmap = 'gray') #plots grayscale image as grayscale

    for i in range(len(thresOpt)):
        plt.subplot(2,3,i+2) # increments which subplot were plotting
        _, imgThres = cv.threshold(imgGray,140, 255,thresOpt[i]) # uses _, as were ignoring the return value, threshold between 70 and 255,create sthreshold images
        plt.imshow(imgThres,cmap='gray') #plots threshold images a grayscale
        plt.title(thresNames[i])
    plt.show()
        

if __name__=='__main__':
    Thresholding()