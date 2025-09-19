# Method to extract circles in an image using Hough Circle Transform
# Used in object detection, and tracking
# circle extraction, find partial circles, find circles of differnet sizes



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def HoughCircleTransform():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    img = cv.imread(imgPath) 
    imgRGB=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGray=cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgGray= cv. medianBlur(imgGray,5) #blurs the image to reduce noise and improve circle detection
    circles=cv.HoughCircles(imgGray, cv.HOUGH_GRADIENT, dp=1, minDist=400, param1=200, param2=5, minRadius=50, maxRadius=150) #detects circles in the image
    #param1 = higher threshold for the Canny edge detector (lower is half this)
    #param2 = accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values
    #minDist = minimum distance between the centers of the detected circles
    #dp = inverse ratio of the accumulator resolution to the image resolution. 1 means same resolution, 2 means half resolution
    circles=np.uint16(np.around(circles)) #rounds the values to integers

    for circle in circles[0,:]:
        cv.circle(imgRGB, (circle[0], circle[1]), circle[2], (255,255,255), 10) #draws the circle in green, circle[2] is the radius, and 0,1 is centre
        
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()








if __name__=='__main__':
    HoughCircleTransform()