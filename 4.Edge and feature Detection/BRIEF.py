# Binary robust independent elementary features (BRIEF)
# Method for finding features in images, FAST to find keypoints, BRIEF to describe them
# first must find points using, SIFT, FAST, SURF AGAST,ETC
# IT THEN SAMPLES 2 PAIRS 

#like SURF video need python 3.6 venv, and im not doing that rn
# pip install opencv-contrib-python
#ACTUALLY got it working

#creates a binary fingerprint of local patch, very niche, dont use much anymore
#BOOOOOO

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def BRIEF():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) 

    fast = cv.FastFeatureDetector_create() #create FAST object
    minInensityDiff=100
    brief=cv.xfeatures2d.BriefDescriptorExtractor_create() #create BRIEF object
    keypoints=fast.detect(imgGray,None) #detect keypoints
    keypoints, descriptors=brief.compute(imgGray,keypoints) #compute the descriptors with BRIEF

    print(brief.descriptorSize()) #size of each descriptor
    print(descriptors[0])
    print(' '.join([format(val,'08b') for val in descriptors[0]]))


if __name__=='__main__':
    BRIEF()