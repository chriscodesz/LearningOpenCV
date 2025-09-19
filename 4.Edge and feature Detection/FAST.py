# Feature from ACcelerated Segment Test (FAST)
# Fast method to find corners in an image
# Feature detection, robustness, faster for real time applications (i.e SLAM)




import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def FAST():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) 

    fast = cv.FastFeatureDetector_create() #create FAST object
    minInensityDiff=100
    fast.setThreshold(minInensityDiff)
    keypoints=fast.detect(imgGray) #detect keypoints
    imgGray=cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #draw keypoints

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

if __name__=='__main__':
    FAST()