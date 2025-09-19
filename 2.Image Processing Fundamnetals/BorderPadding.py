# Adds pixels to the boarder of the image
# used to preserve image size, (need for convolution), and handling boundary conditions


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def BorderPadding():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    img = cv.imread(imgPath) 
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)

    pad =200
    borderTypes=[cv.BORDER_CONSTANT,cv.BORDER_REFLECT, cv.BORDER_REPLICATE,cv.BORDER_WRAP]
    borderTitles = ['constant', 'reflect', ' replicate','wrap']

    plt.figure()
    plt.subplot(231)
    plt.imshow(imgRGB)
    plt.title('original')

    for i in range(len(borderTypes)):
        plt.subplot(2,3,i+2)
        plt.imshow(cv.copyMakeBorder(imgRGB,pad,pad,pad,pad,borderTypes[i]))
        plt.title(borderTitles[i])

    plt.show()


if __name__=='__main__':
    BorderPadding()


