#filtering that finds the median value within a given region
# improved noise reduction and edge preservation
# still uses the kernel but uses the median value not mean, median = middle number

    #Gaussian Filtering → Best for general noise reduction, smooth blur, edge-aware smoothing, natural-looking blur.
    #Average Filtering → Only good for quick & simple smoothing, but often inferior to Gaussian.
    #Median Filtering → Best for removing salt-and-pepper noise while keeping edges sharp.
    #General Convolution → Flexible framework (Gaussian is one special kernel).

import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np


def medianFiltering():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath) 
    

    height,width,_=img.shape 
    scale=2.4
    width=int(width*scale)
    height=int(height*scale)
    img=cv.resize(img,(width,height)) #increasing image size to reduce residula white spots

    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    noisyImg = imgRGB.copy() #adding noise to image
    noiseProb = 0.1
    noise=np.random.rand(noisyImg.shape[0],noisyImg.shape[1]) #creates noise matrix in the shape of our image
    noisyImg[noise<noiseProb/2]=0 #simple easy way to add salt&pepper noise, white and black, adds grain
    noisyImg[noise>1- noise/2]=255

    imgFilter=cv.medianBlur(noisyImg,5) #uses image median blur function with kernel size of 5, to remove blur


    #reduces noise whilts maintaining lining and details


    plt.figure()
    plt.subplot(121)
    plt.imshow(noisyImg)
    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.show()



if __name__=='__main__':
    medianFiltering()