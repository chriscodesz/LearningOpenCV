# Both Object Tracking Algorithms, mean shift tracks with a static box whilst cam shift (continuously adaptive meanshift) tracks with a dynamic box

#tracks images with boxes, used for object tracking robust

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def objectTracking(option):

    #Read Image
    root = os.getcwd()
    videoPath= os.path.join(root, 'LearnCVImages/cake2.MOV')
    videoCapObj=cv.VideoCapture(videoPath)
    _, frame=videoCapObj.read()
    

    #Plot first frame of video
    plt.figure()
    plt.subplot(121)
    plt.imshow(frame)

    #Extract region for tracking
    xTopLeft=249
    yTopleft=906
    w = 525-xTopLeft
    h = 1220-yTopleft
    windowTracker = (xTopLeft, yTopleft, w, h) #x,y,w,h
    imgRegion=frame[yTopleft:yTopleft+h, xTopLeft:xTopLeft+w]
    plt.subplot(122)
    plt.imshow(imgRegion)

    # set range to track based on hsv colors
    hsvImgRegion=cv.cvtColor(imgRegion, cv.COLOR_BGR2HSV)
    hsvLowerLimit=np.array([246/380*2,200,0])                               #input color of the dress min max values
    hsvUpperLimit=np.array([280/360*2,255,255])
    mask=cv.inRange(hsvImgRegion, hsvLowerLimit, hsvUpperLimit)         # creates a mask
    histImgRegion=cv.calcHist([hsvImgRegion],[0],mask,[180],[0,180])    #calcs histogram
    cv.normalize(histImgRegion, histImgRegion,0,255,cv.NORM_MINMAX)     # normalizes histogram to min max
    termCrit = (cv.TermCriteria_EPS|cv.TERM_CRITERIA_COUNT,10,1)        #creates terminate criteria, when to stop iterations, stops if min error (0) is reached or max iterations (10) is reached
    color = (144,238,144)                                               #colour for the box

    #Loop through each frame and track object
    while True:
        ret, videoFrame = videoCapObj.read()
        if ret == True:
            hsv= cv.cvtColor(videoFrame, cv.COLOR_BGR2HSV)
            imgBackProj = cv.calcBackProject([hsv],[0],histImgRegion,[0,180],1)
        
        if option == 'meanshift':
            _,windowTracker=cv.meanShift(imgBackProj, windowTracker, termCrit)
            xTopLeft, yTopleft, w, h = windowTracker
            videoFrame= cv.rectangle(videoFrame, (xTopLeft,yTopleft),(xTopLeft+w,yTopleft+h),color,2)

        if option == 'camshift':
            ret ,windowTracker=cv.CamShift(imgBackProj, windowTracker, termCrit) # ret is rotanted rectangle, centre size and angle
            boxPts= cv.boxPoints(ret)
            videoFrame= cv.polylines(videoFrame, [np.int32(boxPts)],True,color,2)
    
        cv.imshow('video', videoFrame)
        cv.waitKey(15)
        cv.destroyAllWindows()










    plt.show()


if __name__ == "__main__":
    #objectTracking(option ='meanshift')
    objectTracking(option = 'camshift')




