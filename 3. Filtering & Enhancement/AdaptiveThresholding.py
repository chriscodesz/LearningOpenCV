# A thresholding that finds a threshold value based on a local region of pixels (instead of normal global)
# allowing for thresholding in uneven lighting, varying contrast to obtain outline of image
# breaks image into section, finds threshold value, two methods,
    # Gaussian adaptive thresholding
    # Mean adaptive thresholding

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def AdaptiveThreshold():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE) 

    plt.figure()
    plt.subplot(141)
    plt.imshow(imgGray, cmap='gray')
    plt.title('gray')

    plt.subplot(142)
    _, imgThres=cv.threshold(imgGray,70,255,cv.THRESH_BINARY) #basic threshold, seen in other project
    plt.imshow(imgThres, cmap='grey')
    plt.title('global thresh')

    plt.subplot(143)
    maxValue=255
    blocksize=7 #pixels size 7x7
    offsetC =2
    imgMean = cv.adaptiveThreshold(imgGray,maxValue, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, blocksize,offsetC)
    plt.imshow(imgMean, cmap='gray')
    plt.title('mean thres')

    plt.subplot(144)
    imgGauss = cv.adaptiveThreshold(imgGray,maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY, blocksize,offsetC)
    plt.imshow(imgGauss, cmap='gray')
    plt.title('Gauss thres')







    plt.show()


if __name__=='__main__':
    AdaptiveThreshold()