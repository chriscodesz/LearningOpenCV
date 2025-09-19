# Makes images bigger or smaller, used to reduce data, good for preprocessing
# Can improve resolution




import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np

def imageResize():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath) 
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB) #gets image and converst to a RGB format

    img = img[903:1121,1826:2080] #resizes image
    height,width,_=img.shape      #gets image dimensions
    scale=4                     #determines scale of shrinkage, <1 zoom out, >1 scale in

    interpMethods= [ #list of all the interperlation methods, look at word doc.
        cv.INTER_AREA,
        cv.INTER_LINEAR,
        cv.INTER_NEAREST,
        cv.INTER_CUBIC,
        cv.INTER_LANCZOS4
    ]
    interpTitle= ['area','linear','nearest','cubic','lanczos']


    plt.figure()
    plt.subplot(2,3,1)
    plt.imshow(img)

    for i in range(len(interpMethods)): #creating a for loop to go through and print every interpolation method, more efficient
        plt.subplot(2,3,i+2) #cylces through each subplot
        imgResize=cv.resize(img,(int(width*scale),int(height*scale)), interpolation=interpMethods[i]) #resizes images based on interpolation method selected on cycle
        plt.imshow(imgResize) #adds image to figure
        plt.title(interpTitle[i]) #adds title of method
    plt.show()

if __name__=='__main__':
    imageResize()