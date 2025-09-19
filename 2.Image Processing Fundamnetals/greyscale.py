#single channel black and white pixels, used to reduce the amount of data.
# works by scaling BGR, for open CV: Greyscale Pixel= 0.299R+ 0.587G + 0.114B

import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np

def grayScale():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #converts it to gray

    cv.imshow('gray',imgGray)
    cv.waitKey(0)


def readAsGray():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE) #converts it to gray, different method, loses a line

    cv.imshow('gray',img)
    cv.waitKey(0)



if __name__=='__main__':
    #grayScale()
    readAsGray()