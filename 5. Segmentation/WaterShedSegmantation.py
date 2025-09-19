# A method of segmenting an image
# Good for segmenting regions that are touching or overlapping
# Used in object detection, and tracking



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def WaterShedSeg():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    img = cv.imread(imgPath) 
    img= img[0:389, 1469:1806]
    imgRGB=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')

    #Thresholding, Dilation, Distance Transform, Watershed
    plt.subplot(232)
    _,imgThreshold=cv.threshold(img,65,255,cv.THRESH_BINARY)
    plt.imshow(imgThreshold, cmap='gray')
    plt.title('Thresholded')

    #dialte to get sure background area
    plt.subplot(233)
    kernel=np.ones((3,3), np.uint8)
    imgDilate=cv.morphologyEx(imgThreshold,cv.MORPH_DILATE,kernel)
    plt.imshow(imgDilate, cmap='gray')
    plt.title('Dilated')

    # distance transform
    plt.subplot(234)
    distanceTran=cv.distanceTransform(imgDilate,cv.DIST_L2,5) #calculates the distance to the nearest zero pixel for each pixel,L2 is euclidean distance
    plt.imshow(distanceTran)
    plt.title('Distance Transform')

    #threshold the distance transform to get the peaks
    plt.subplot(235)
    _,distThres=cv.threshold(distanceTran,8,255,cv.THRESH_BINARY) #thresholds the distance transform to get the peaks
    plt.imshow(distThres)
    plt.title('Distance Transform threshold')

    plt.subplot(236)
    distThres=np.uint8(distThres)
    _,labels=cv.connectedComponents(distThres) #labels the peaks
    plt.imshow(labels)
    plt.title('Labels')

    # watershed
    plt.figure()
    plt.subplot(121)
    labels=np.int32(labels)
    labels=cv.watershed(imgRGB, labels)
    plt.imshow(labels)
    plt.title('Watershed Segmentation')

    plt.subplot(122)
    imgRGB[labels==-1]=[255,0,0] #marks the boundaries in red
    plt.imshow(imgRGB)
    plt.title('Final Result')


    plt.show()


# NOTEEE: as seen its not perfect, which is why people resort to machine learning and AI methods for segmentation
# but this is a good method to understand and use when you have overlapping objects


if __name__=='__main__':
    WaterShedSeg()
