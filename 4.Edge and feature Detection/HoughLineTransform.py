# A method to extract lines from and image



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np



def houghline():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    img = cv.imread(imgPath,cv.IMREAD_GRAYSCALE) 
    imgBlur= cv. GaussianBlur(img,(21,21),3)
    cannyedge=cv.Canny(imgBlur,50,80)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    plt.subplot(222)
    plt.imshow(imgBlur)
    plt.subplot(223)
    plt.imshow(cannyedge)

    distResol = 1 #distance resolution
    angleResol= np.pi/180 # angle resolution
    threshold= 170
    lines =cv.HoughLines(cannyedge,distResol,angleResol,threshold)
    k=3000

    for curLine in lines: #calculates line
        rho,theta=curLine[0]
        dhat=np.array([[np.cos(theta)],[np.sin(theta)]])
        d=rho*dhat
        lhat=dhat=np.array([[-np.sin(theta)],[np.cos(theta)]])
        p1=d+ k*lhat
        p2=d- k*lhat
        p1=p1.astype(int)
        p2=p2.astype(int)
        cv.line(img,(p1[0][0],p1[1][0]),(p2[0][0],p2[1][0]), (255,255,255),10) 
    
    plt.subplot(224)
    plt.imshow(img)


    plt.show()




if __name__=='__main__':
    houghline()