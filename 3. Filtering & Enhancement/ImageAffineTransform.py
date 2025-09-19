# Linear mapping that preserves straight lines and ratios of distances
# But allows for Position Orientation and scale of image to be changed
# can be used for image registration-to align images, image rectifciaction, for camera calibration, or image warping/ morphing to deform
# Uses a tranfomation matrix, must ahve 3 pairs of corrssponding points.

import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np

def affineTransform():
    root= os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/catpic1.jpg')
    img = cv.imread(imgPath)
    imgRGB=cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width,_ = img.shape
    #/Users/chrisdixon/Robotics/Coding/OpenCV/LearnOpenCV/LearnCVImages/catpic1.jpg
    p1= np.array([[100,100],[900,100],[100,900]],dtype = np.float32) #chosen befor and after points, need 3 points
    p2 =np.array([[50,150],[900,100],[150,800]], dtype = np.float32)

    T = cv.getAffineTransform(p1,p2)
    imgTrans=cv.warpAffine(imgRGB,T,(width,height))

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgTrans)
    plt.show()



if __name__=='__main__':
    affineTransform()