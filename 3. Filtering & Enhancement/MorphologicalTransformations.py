# Kernel based operations ,typically on binary images, emphasizes foreground or background by changing -
# the size or shape of the image
# used for noise reduction, image enhancement, segmentation
#



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def MorphTrans():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/BMW.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE) 

    maxValue= 255
    blocksize =7
    offsetC =2
    imgGaus=cv.adaptiveThreshold(imgGray, maxValue,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,blocksize,offsetC)
    imgGaus=cv.GaussianBlur(imgGaus,(7,7),sigmaX=2)
    plt.figure()
    plt.subplot(241)
    plt.imshow(imgGaus, cmap='gray')
    plt.title('gauss thresh')

    #erossion emphasises the black (background) more
    kernel= np.ones((7,7),np.uint8)
    erosion = cv.erode(imgGaus,kernel,iterations=1)
    plt.subplot(242)
    plt.imshow(erosion, cmap='gray')
    plt.title('erosion')

    #dialtion increase whites
    dialate = cv.dilate(imgGaus,kernel,iterations=1)
    plt.subplot(243)
    plt.imshow(dialate, cmap='gray')
    plt.title('dialate')

    morphTypes = [cv.MORPH_OPEN,cv.MORPH_CLOSE,cv.MORPH_GRADIENT,cv.MORPH_TOPHAT,cv.MORPH_BLACKHAT]
    morphTitles = ['open','close','gradient','tophat','blackhat']

    for i in range(len(morphTypes)):
        plt.subplot(2,4,4+i)
        plt.imshow(cv.morphologyEx(imgGaus, morphTypes[i],kernel),cmap= 'gray')
        plt.title(morphTitles[i])

    plt.show()

    # Some custom kernels
    ellipseKernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    crossKernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
    print(ellipseKernel)
    print(crossKernel)


if __name__=='__main__':
    MorphTrans()
