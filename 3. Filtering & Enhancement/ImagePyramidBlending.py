# Uses a gaussina pyramid and a lapacian pyramid
# Used fro smoother blending
# good for combining images and making nice and smooth




import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def ImgPyramidBlnd():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/BMW.jpg')
    imgBGR = cv.imread(imgPath) 
    imgRGB =cv.cvtColor(imgBGR, cv.COLOR_BGRA2RGB)

    plt.figure()
    downSamp= imgBGR.copy() #down sample the image,
    BGR_gausPyramidList=[downSamp]
    plt.subplot(231)
    plt.imshow(imgRGB)
    for i in range(5):
        plt.subplot(2,3,i+2)
        downSamp=cv.pyrDown(downSamp) #down samples the image
        plt.imshow(downSamp)
        BGR_gausPyramidList.append(downSamp)
    
    plt.figure()
    BGR_lapPyramidList=[BGR_gausPyramidList[4]]
    for i in range(4,0,-1):
        plt.subplot(2,3,4-i+1)
        upSamp=cv.pyrUp(BGR_gausPyramidList[i])
        diff=cv.subtract(BGR_gausPyramidList[i-1],upSamp)#laplacian is the difference between the gaus pyramid level and the expanded next level
        BGR_lapPyramidList.append(diff)
        plt.imshow(diff)

    plt.figure()
    downSamp= imgRGB.copy() #down sample the image,
    RGB_gausPyramidList=[downSamp]
    plt.subplot(2,3,1)
    plt.imshow(downSamp)
    for i in range(5):
        plt.subplot(2,3,i+2)
        downSamp=cv.pyrDown(downSamp) #down samples the image
        plt.imshow(downSamp)
        RGB_gausPyramidList.append(downSamp)

    plt.figure()
    RGB_lapPyramidList=[RGB_gausPyramidList[4]]
    for i in range(4,0,-1):
        print(4-i+1)
        plt.subplot(2,3, 4-i+1)
        upSamp=cv.pyrUp(RGB_gausPyramidList[i])
        diff=cv.subtract(RGB_gausPyramidList[i-1],upSamp)#laplacian is the difference between the gaus pyramid level and the expanded next level
        RGB_lapPyramidList.append(diff)
        plt.imshow(diff)

    

    combinedList=[]
    plt.figure()
    offset=7
    for i in range(len(RGB_lapPyramidList)):
        left=RGB_lapPyramidList[i]
        right= BGR_lapPyramidList[i]
        _,cols,_=left.shape
        combined=np.hstack((left[:,0:cols//2+offset],right[:,cols//2+offset:]))
        combinedList.append(combined)
        plt.subplot(2,3,i+1)
        plt.imshow(combined)

    blend=combinedList[0]
    for i in range(1,4):
        blend=cv.pyrUp(blend)
        blend=cv.add(blend, combinedList[i])

    plt.figure()
    plt.imshow(blend)




    plt.show()


if __name__=='__main__':
    ImgPyramidBlnd()
