# Also known as Homography
# Project 3D object to 2D plane, warps something to a 2D view
# image rectification-camera calibration, planar surface extraction, image stitching combing multiple images togetehr
# uses 4 pairs of points and a mtrix, that translates the area.

import cv2 as cv 
import os
import matplotlib.pyplot as plt
import numpy as np

def ImgPerspTransform():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    img = cv.imread(imgPath)

    p1=np.array([[1285,681],[1445,662],[1287,726],[1455,708]], dtype=np.float32) #no.plate coords

    p2=np.array([[0,0],[100,0],[0,60],[100,60]], dtype=np.float32) # 100 by 60 box

    T=cv.getPerspectiveTransform(p1,p2) #gets transform matrix to achieve this
    imgTrans=cv.warpPerspective(img,T,(100,60)) #transorms to new box


    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.plot(p1[:,0],p1[:,1],'r')
    plt.subplot(122)
    plt.imshow(imgTrans)
    plt.plot(p2[:,0],p2[:,1],'r')
    plt.show()

if __name__=='__main__':
    ImgPerspTransform()
