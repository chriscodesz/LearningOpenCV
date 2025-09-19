# Uses a histogram of a region to identify parts of an image with similar histogram distributions
# Onject Tracking, image segmenattion
# Compute the histogram of a region of interest (ROI) in an image, then use that histogram to find similar regions in the same or another image.


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def HistBackProj():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/BMW.jpg')
    img = cv.imread(imgPath) 
    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('Original Image')

    imgRegion=img[807:865,1877:2000,:] #region of interest
    plt.subplot(232)
    plt.imshow(imgRegion)
    plt.title('Region of Interest')

    #calcs back projection
    imgRegionHSV=cv.cvtColor(imgRegion, cv.COLOR_RGB2HSV)
    imgRegionHist=cv.calcHist([imgRegionHSV], [0,1], None, [180,256], [0,180,0,256])
    cv.normalize(imgRegionHist, imgRegionHist, 0, 255, cv.NORM_MINMAX)
    imgHSV=cv.cvtColor(img, cv.COLOR_RGB2HSV)
    out = cv.calcBackProject([imgHSV],[0,1], imgRegionHist, [0,180,0,256],1)
    plt.subplot(233)
    plt.imshow(out)
    plt.title('Back Projection')

    #expands the area of the car thats picked up as orange
    ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15)) 
    cv.filter2D(out,-1,ellipseKernel,out)
    plt.subplot(234)
    plt.imshow(out)
    plt.title('Back projection with filter, ellipse kernel')

    #thresholds the image to create a mask, converts to binary image, anything above 70 is white, below is black
    _,mask=cv.threshold(out,70,255,0)
    plt.subplot(235)
    plt.imshow(mask)
    plt.title('Threshold, mask')

    #uses the mask to segment the original image
    maskAllChannels=cv.merge((mask,mask,mask))
    imgSegm=cv.bitwise_and(img,maskAllChannels) #removes everything outside the mask
    plt.subplot(236)
    plt.imshow(imgSegm)
    plt.title('Segmented Image')

    plt.show()
    


if __name__=='__main__':
    HistBackProj()