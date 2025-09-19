'''What is RGB colour channels'''
#images have green, blue, and red channels, with opencv its BGR,
#we need RGB colour channels for features or segmentation of the images

import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pa

def pureColors():
    zeros=np.zeros((100,100))
    ones=np.ones((100,100))
    bImg=cv.merge((zeros,zeros,255*ones)) #removes all greens and red in BGR, but using matplotLib,Imshow is RGB
    GImg=cv.merge((zeros,255*ones,zeros))
    RImg=cv.merge((255*ones,zeros,zeros))
    blackImg = cv.merge((zeros,zeros,zeros))
    whiteimage=cv.merge((ones,ones,ones))

    plt.figure()
    plt.subplot(231) #creates a 2x3 subplot
    plt.imshow(bImg) #RGB
    plt.title('blue')
    plt.subplot(232) #selects 2nd channel
    plt.imshow(GImg) #RGB
    plt.title('Green')
    plt.subplot(233) #selects 3rd channel
    plt.imshow(RImg) #RGB
    plt.title('Red')

    plt.subplot(234) #selects 3rd channel
    plt.imshow(blackImg) #RGB
    plt.title('black')
    plt.subplot(235) #selects 3rd channel
    plt.imshow(whiteimage) #RGB
    plt.title('white')

    plt.show()


def bgrChannelGrayScale(): #shows image in seperate BGR values , normalised to a grey scale
    #this doesn not show only red only green only blue as we havent zeroed out the other colour channels
    
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath)
    b,g,r=cv.split(img) #passes in the BGR channels seperately for the image, each is a 2D array

    zeros = np.zeros_like(b) #creates a zerosmatrix the smae size as the b matrix(matches image size)
    ones = np.ones_like(b)

    plt.figure()
    plt.subplot(131)
    plt.imshow(b, cmap='gray') #shows the blue channel in gray scale
    plt.title('blue')
    plt.subplot(132)
    plt.imshow(g, cmap='gray') #shows the blue channel in gray scale
    plt.title('Green')
    plt.subplot(133)
    plt.imshow(r, cmap='gray') #shows the blue channel in gray scale
    plt.title('red')

    plt.show()



def bgrChannelColor():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath)
    b,g,r=cv.split(img)

    zeros = np.zeros_like(b) #create zeros matrix same size as image (b channel)(can be any)
    bImg=cv.merge((b,zeros,zeros)) #zeroing out the other colours to reveal just the blue,green,red
    GImg=cv.merge((zeros,g,zeros))
    RImg=cv.merge((zeros,zeros,r))

    plt.figure()
    plt.subplot(131)
    plt.imshow(bImg)
    plt.subplot(132)
    plt.imshow(GImg)
    plt.subplot(133)
    plt.imshow(RImg)
    plt.show() #plot individual channels

if __name__=='__main__':
    #pureColors()
    #bgrChannelGrayScale()
    bgrChannelColor()