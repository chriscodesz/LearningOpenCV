# Kernel that filters based on colour intensity and spatial information
# Preserves edges, but removes noise
# kernel is a combinationof a gaussain and a step


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def BilateralFiltering():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    img = cv.imread(imgPath) 
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    height, width,_ =imgRGB.shape
    scale = 1/4
    width = int(width*scale)
    height = int(height*scale)
    imgRGB=cv.resize(imgRGB,(width,height))

    imgFilter= cv.bilateralFilter(imgRGB,25,100,100) #d=neighbours, real time, smaller, sigma =100

    plt.figure()
    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.show()


if __name__=='__main__':
    BilateralFiltering()