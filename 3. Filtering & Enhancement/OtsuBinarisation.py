# Thresholding technique tries to seperate the foreground and background by trying to seperate the two histograms into two class
# Uneven lighting, Automatic threshold value calculation
# compute histogram, iterate through all possible threshold values, find value that minimises weight in class variance for the two classes
# threshold images with that value


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def OtsuBinary():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE) 

    plt.figure()
    plt.subplot(131)
    plt.imshow(imgGray, cmap='gray')
    plt.title('gray')

    #original threshold technique
    plt.subplot(132)
    thres=134 # can adjust to dip point in histogram
    maxval =255
    _,imgThres=cv.threshold(imgGray,thres,maxval,cv.THRESH_BINARY)
    plt.imshow(imgThres, cmap='gray')
    plt.title('global, matching to peak in hiostogram= thres= 134ish')


    # Otsu Binary Thresholding
    plt.subplot(133)
    arbthres=0
    _,imgOTSU=cv.threshold(imgGray,arbthres,maxval,cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.imshow(imgOTSU, cmap='gray')
    plt.title('OTSU')

    hist=cv.calcHist([imgGray],[0],None,[256],[0,256])
    plt.figure()
    plt.plot(hist)
    plt.xlabel=('intensity')
    plt.ylabel=('# no pixels')
    


    plt.show()


if __name__=='__main__':
    OtsuBinary()
