# A method to find corners in an image
# Used for feature detection, calibrate patterns
# equation works by finding the maxium change in directions

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def harrisCorner():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    img = cv.imread(imgPath) 
    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgGray=np.float32(imgGray) #data type conversion

    plt.figure()
    plt.subplot(131)
    plt.imshow(imgGray,cmap='gray')

    plt.subplot(132)
    blocksize=5
    sobelsize=3 #cresating harris required parameters, can vary dependent on imgae
    k=0.04
    harris=cv.cornerHarris(imgGray,blocksize, sobelsize,k) # creates an image using the corner harris function
    plt.imshow(harris) # plots small dos highlighted by the harris function

    plt.subplot(133)
    imgRGB[harris>0.05*harris.max()]=[255,0,0] # marks the image where it sees corners in red 
    plt.imshow(imgRGB)


    plt.show()
if __name__=='__main__':
    harrisCorner()