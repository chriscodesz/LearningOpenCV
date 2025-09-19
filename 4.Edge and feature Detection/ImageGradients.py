# Find the first or second derivative of an image
# Good for EDGE detection, FEATURE extraction, image ENHANCEMENT
# Convolve image with derivative kernel= SobelX, SobelY, Laplacian
#lapacian is considered 'UNI-directional', sensitive to both x and y direction







import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def imageGradient():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) 
    
    plt.figure()
    plt.subplot(221)
    plt.imshow(img, cmap='gray')

    laplacian=cv.Laplacian(img,cv.CV_64F,ksize=21) # type 64 for high res, and kernel size 21
    plt.subplot(222)
    plt.imshow(laplacian,cmap='grey')

    kx,ky = cv.getDerivKernels(1,0,3) #deriveitve in the x not y hence 1, 0, 3 by 3 matrix
    print(ky@kx.T) #this is for sobel x
    sobelx=cv.Sobel(img,cv.CV_64F,1,0,ksize=21)
    plt.subplot(223)
    plt.imshow(sobelx,cmap='grey') # capture alot of the verticle line (left and right)

    kx,ky = cv.getDerivKernels(0,1,3) #deriveitve in the y not x hence 0, 1
    print(ky@kx.T) #this is for sobel y This and above line not really needed
    sobely=cv.Sobel(img,cv.CV_64F,0,1,ksize=21)
    plt.subplot(224)
    plt.imshow(sobely,cmap='grey') # capture alot of the horizontal line (kernel goes up and down)




    plt.show()

if __name__=='__main__':
    imageGradient()