import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np

def Rotation():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/lake.jpg')
    img = cv.imread(imgPath)
    width, height,_ = img.shape
    T=cv.getRotationMatrix2D((width/2,height/2),180,0.5) #roatte 180 degrees
    imgTrans = cv.warpAffine(img,T,(width,height))



    cv.imshow('imgtrans', imgTrans)
    cv.waitKey(0)


if __name__=='__main__':
    Rotation()
