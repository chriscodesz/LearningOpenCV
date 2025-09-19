# Oriented fast and rotated brief (ORB)
# detctor and decriptor
# Alternative to SIFT or SURF, free to use even for commercial applications, 
# Feature detection and description, very fast, works well in real time applications (i.e SLAM)
# Feature Detection, efficient, low memory

#Very fast replacemnt for surf and sift for real time applications


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def ORB():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) 

    orb = cv.ORB_create()
    keypoints=orb.detect(imgGray,None) #detect keypoints
    keypoints,_=orb.compute(imgGray,keypoints) #compute the descriptors with ORB
    imgGray=cv.drawKeypoints(imgGray, keypoints, imgGray, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #draw keypoints

    plt.figure()
    plt.imshow(imgGray)
    plt.show()


    


if __name__=='__main__':
    ORB()