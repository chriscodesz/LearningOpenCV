# Controus are a peice wise collection of lines that describe the outline or location of objects
# Used in Object detction, segemntation, ROI selection



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def Contours():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/cross.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) 
    

    plt.figure()
    plt.subplot(231)
    plt.imshow(img,cmap='gray')
   
    height, width = img.shape
    scale=4
    heightScale=int(height*scale)
    widthScale=int(width*scale)
    img = cv.resize(img,(widthScale,heightScale))

    _,thresh=cv.threshold(img, 120,255,cv.THRESH_BINARY)
    kernel= np.ones((7,7),np.uint8)
    thresh=cv.dilate(thresh,kernel)
    contours,_=cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) #find controus of threshold image, uses a retieval tree method (gets all contors)
    contours= [contours[0]]
    cv.drawContours(img,contours,-1,(0,0,255),3) #draws all contours on the image, -1 means all contours, 3 is the thickness of the line
    plt.subplot(232)
    plt.imshow(thresh,cmap='gray')
    plt.subplot(233)
    plt.imshow(img,cmap='gray')

    #can calculate the centre of mass
    M = cv.moments(contours[0])
    cx = int(M['m10']/M['m00']) #centroid x
    cy = int(M['m01']/M['m00']) #centroid y

    plt.subplot(234)
    plt.imshow(img,cmap='gray')
    plt.plot(cx,cy,'r*') #plots centre of mass with red star
    plt.title('Centre of Mass')

    #can also do contour approximation
    area=cv.contourArea(contours[0])
    perimeter=cv.arcLength(contours[0],True) #true means its a closed contour
    epsilon= .01*perimeter
    approx=cv.approxPolyDP(contours[0],epsilon,True) #approxim
    approx=np.array(approx)
    approx=np.concatenate((approx, approx[:1]), axis =0)
    plt.plot(approx[:,0,0],approx[:,0,1])


    #creates hull
    hull = cv.convexHull(contours[0]) #convex hull is the smallest convex shape that can enclose the contour
    hull=hull[:,0,:]
    hull=np.concatenate((hull, hull[:1]), axis = 0)
    plt.subplot(235)
    plt.imshow(img,cmap='gray')
    plt.plot(hull[:,0],hull[:,1],'r-')
    plt.title('hull')

    #bounding box
    x,y,w,h=cv.boundingRect(contours[0])
    plt.subplot(236)
    cv.rectangle(img,(x,y),(x+w,y+h),(255,255,255),2)
    plt.imshow(img,cmap='gray')
    plt.title('Bounding Box')

    aspectRatio = w/h
    extent = area/(w*h) # how much of the bounding box is filled by the contour
    solidity = area/cv.contourArea(hull) #how much of the convex hull is filled by the contour
    equalDiameter = np.sqrt(4*area/np.pi) #diameter of a circle with the same area as the contour
    _,_,_angle=cv.fitEllipse(contours[0]) #fits an ellipse to the contour, returns the angle of the ellipse
    print('Aspect Ratio: ', aspectRatio)
    print('Extent: ', extent)
    print('Solidity: ', solidity)
    print('Equal Diameter: ', equalDiameter)
    print('Angle: ', _angle)


    plt.show()






if __name__=='__main__':
    Contours()



