# Method of an instance of a template in an image, finding a feature in an image
# Used in object detection, and tracking



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def TempMatch():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    img = cv.imread(imgPath) 
    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)
    teslaLogo=img[496:577,915:1010]
    height, width,_ = teslaLogo.shape

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(teslaLogo)
    

    methods=[cv.TM_CCOEFF, cv.TM_CCOEFF_NORMED, cv.TM_CCORR, cv.TM_CCORR_NORMED, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]
    Titles=['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

    for i in range(len(methods)):
        curImg=img.copy()
        templateMap=cv.matchTemplate(img,teslaLogo,methods[i])
        _,_,minLoc,maxLoc=cv.minMaxLoc(templateMap) #finds the min and max locations in the template map

        if methods[i] == cv.TM_SQDIFF or methods[i]== cv.TM_SQDIFF_NORMED:
            topLeft=minLoc
        else:
            topLeft=maxLoc
        bottomRight=(topLeft[0]+width, topLeft[1]+height)
        cv.rectangle(curImg,topLeft,bottomRight,(255,255,255),10)
        plt.figure()
        plt.subplot(121)
        plt.imshow(templateMap, cmap='gray')
        plt.title(Titles[i])
        plt.subplot(122)
        plt.imshow(curImg)

    #TM SQDIFF = squared, lower values are better matches, black spot
    #TM_CCORR NORMED = normalised cross correlation, higher values are better matches, white spot
    



    
    plt.show()





if __name__=='__main__':
    TempMatch()