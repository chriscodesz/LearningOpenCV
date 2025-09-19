# SCALE INVARIANT FEATURE TRANSFORM- scale if point of intrest doesnt affect the method 
# A method to find features in images
# used by detecting blobs-"pathces of an image with a local apperance"
# finds using lapacian of of a gaussian and derivative of a gaussian (LoG and DoG)
# DoG are from diffreence, subtracted, of gasusians of different sigmas, (bell width)
# blobs are found as maxiums in the detector
# Then key point localisation is used
# Orientation--magnitude and angle are then calculated 



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def SIFT():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    imgGray = cv.imread(imgPath,cv.IMREAD_GRAYSCALE) 
    

    sift=cv.SIFT_create()
    keypoints=sift.detect(imgGray,None)
    imgGray=cv.drawKeypoints(imgGray,keypoints,imgGray,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(imgGray)
    plt.show()

if __name__=='__main__':
    SIFT()