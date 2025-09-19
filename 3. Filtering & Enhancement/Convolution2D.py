# Smooths or blurs images, feature extraction, image enhancement, and edge detection
# Sliding kernel matrix that does element wise multiplication on the input image
#uses a 3x3 matrix to calc mean values of pixels, adds all up divideds by 9 (3x3).
#then slides to the next row of pixels.

    #Gaussian Filtering → Best for general noise reduction, smooth blur, edge-aware smoothing, natural-looking blur.
    #Average Filtering → Only good for quick & simple smoothing, but often inferior to Gaussian.
    #Median Filtering → Best for removing salt-and-pepper noise while keeping edges sharp.
    #General Convolution → Flexible framework (Gaussian is one special kernel).

import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np

def convolution2D():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath) 
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    


    n=100
    kernel=np.ones([n,n],np.float32)/(n*n) # ones matrix nXn big, allows float 32 data
    imgFilter =cv.filter2D(imgRGB,-1, kernel ) #creates filtered image, calls a convolution filter function, depth -1



    plt.figure()
    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.show()



if __name__=='__main__':
    convolution2D()