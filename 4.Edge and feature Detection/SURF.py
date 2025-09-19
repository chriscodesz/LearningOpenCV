# Speeded-up robustness features (SURF), like sift
# Method for finding features in images, robust, FASTER for real time applications (compared to SIFT)

#Difficult cos me need to: pip install opencv-contrib-python==3.4.2.16
#DONT USE, PATENTED, MORE DIFFICULT, SO NOT USED MUCH

#USE FAST

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def surf():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) 
    

    hessianThreshold=3000
    surf=cv.xfeatures2d.SURF_create(hessianThreshold)
    keypoints=surf.detect(imgGray, None)
    imgGray=cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

if __name__=='__main__':
    surf()
