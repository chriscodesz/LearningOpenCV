# An algorithm typically more robust than standard gradient methods
# Good for OBJECT detection, IMAGE Segmentation, Feature Extraction
# Works via gaussian filter smoothing, gradient image woth sobel, apply a non maxium suppression to find maxoumum of reigion (thins out edges)
# Then double thresholds to keep edges that go above maxium




import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def callback(input):
    pass


def cannyEdge():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/lake.jpg')
    img = cv.imread(imgPath) 
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB) 

    height,width,_=img.shape 
    scale=1/5
    width=int(width*scale)
    height=int(height*scale)
    imgRGB=cv.resize(imgRGB,(width,height), interpolation=cv.INTER_LINEAR) #rescaling so it doesnt go out the screen

    winName='canny' # for trackbar for controlling sigma
    cv.namedWindow(winName)
    cv.createTrackbar('minThresh',winName,0,255,callback) # creates a track bar from 1-20
    cv.createTrackbar('maxThresh',winName,0,255,callback) # creates a track bar from 1-20

    while True:
        if cv.waitKey(1)==ord('q'):
            break

        minThres=cv.getTrackbarPos('minThresh',winName)
        maxThresh=cv.getTrackbarPos('maxThresh',winName)
        cannyEdge=cv.Canny(img, minThres,maxThresh)
        cv.imshow(winName, cannyEdge)

    cv.destroyAllWindows

if __name__=='__main__':
    cannyEdge()