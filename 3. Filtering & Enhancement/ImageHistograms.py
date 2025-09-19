# Shows distribution of pixel intensities against number of pixels
# Needed for thresholding, equalisationa nd enhancements, color anaylsis and segmentation

import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np


def grayHistogram(): #creates a grayscale colour histogram
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE) #turn image grayscale

    plt.figure()
    plt.imshow(img,cmap='gray') #outputs a grayscale image

    hist = cv.calcHist([img],[0],None,[256], [0,256]) #using histogram function, pass image, channel 1 so [0], no mask, 256, range of values 0-256

    plt.figure()
    plt.plot(hist)
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')
    plt.show()

def colorHistogram(): #creates a histogram of RGB values of an image
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath) #dont want grayscale anymore, turn to RGB
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)

    colours = ['b','g', 'r']
    plt.figure()
    for i in range(len(colours)):
        hist=cv.calcHist([imgRGB],[i],None,[256], [0,256]) # calc hists for all colours, i index for each differnt channel
        plt.plot(hist,colours[i]) # plots the histograms based on the corresponding colours


    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')
    plt.show()

def histogramRegion():
    root = os.getcwd() #all same as colour histogram but trunkating the image
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath) 
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)


    imgRGB= imgRGB[600:700,600:700]#crops image into area of interest, from pixels 600-700

    plt.figure()
    plt.imshow(imgRGB)

    colours = ['b','g', 'r']
    plt.figure()
    for i in range(len(colours)):
        hist=cv.calcHist([imgRGB],[i],None,[256], [0,256])
        plt.plot(hist,colours[i]) 


    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')
    plt.show()



if __name__=='__main__':
    #grayHistogram()
    #colorHistogram()
    histogramRegion()
