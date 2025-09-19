# Depth mapping for Stereo Vision (dispartity map)
# Describes the distance between each pixel
# Used for 3D reconstruction, AR, Depth estimation

#Works by block matching, find best window match between the two images,
    # SUM of absolute differences (SAD)
    # SUM of sqaured differences (SSD)
    # Normalized cross correleation (NCC)

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import glob


class DepthMap():
    def __init__(self,showImages):
        #LoadImages
        root = os.getcwd() 
        imgLeftPath = os.path.join(root, 'LearnCVImages/irishStreet1.jpg')
        imgRightPath = os.path.join(root, 'LearnCVImages/irishStreet2.jpg')
        self.imgLeft = cv.imread(imgLeftPath, cv.IMREAD_GRAYSCALE)
        self.imgRight = cv.imread(imgRightPath, cv.IMREAD_GRAYSCALE)

        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.imgLeft)
            plt.subplot(122)
            plt.imshow(self.imgRight)
            plt.show()

    def computeDepthMapBM(self):
        nDispFactor=4 # adjust this
        stereo = cv.StereoBM.create(numDisparities=16*nDispFactor, blockSize = 21) #CREATES A STERO BLOCK MATCHING OBJECT SO WE CAN CREATE A DISPARITY MAP
        disparity = stereo.compute(self.imgLeft, self.imgRight) #
        plt.imshow(disparity, 'gray')
        plt.show()

    def computeDepthMapSGBM(self):
        window_size =7
        min_disp = 16
        nDispFactor=15 #adjust this (14 is good)
        num_disp = 16*nDispFactor-min_disp

        stereo = cv.StereoSGBM_create(minDisparity=min_disp,
                                      numDisparities = num_disp,
                                      blockSize = window_size,
                                      P1=8*3*window_size**2,    #common penalty factors, pretty hard to tune
                                      P2 = 32*3*window_size**2,
                                      disp12MaxDiff=1,
                                      uniquenessRatio=15,
                                      speckleWindowSize=0,
                                      speckleRange=2,
                                      preFilterCap=63,
                                      mode= cv.STEREO_SGBM_MODE_SGBM_3WAY)
        
        #compute disparity map
        disparity = stereo.compute(self.imgLeft, self.imgRight).astype(np.float32)/16.0

        # Display the disparity map
        plt.imshow(disparity, 'gray')
        plt.colorbar()
        plt.show()




def demoViewPics():
    #see picture
    dp=DepthMap(showImages=True)

def demostereoBM():             #SHOWS BLOCK MATCHING METHOD
    dp = DepthMap(showImages=False)
    dp.computeDepthMapBM()

def demoStereoSGBM():
    dp= DepthMap( showImages=False)
    dp.computeDepthMapSGBM()



if __name__=='__main__':
    #demoViewPics()
    #demostereoBM() #Further away, less disparity, closer more
    demoStereoSGBM()
