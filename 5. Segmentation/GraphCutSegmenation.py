# A method of egmenting object in image and removing background
# can interactively update segmenation results




import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def GraphCut():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/Tesla.jpg')
    img = cv.imread(imgPath) 
    img=cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(231)
    plt.imshow(img)

    plt.subplot(232)
    rows,cols,_ =img.shape
    mask=np.zeros((rows,cols), np.uint8) #initial mask, 0
    bgdModel=np.zeros((1,65), np.float64) #background model
    fgdModel=np.zeros((1,65), np.float64) #foreground model
    x0=366
    y0=153
    x1=1556
    y2=903
    rect=(x0,y0,x1-x0,y2-y0) #rectangle around the object to segment
    iter=1 
    cv.grabCut(img, mask, rect, bgdModel,fgdModel,iter, cv.GC_INIT_WITH_RECT)
    plt.imshow(mask)

    plt.subplot(233)
    maskGC=np.where((mask==2)|(mask==0),0,1).astype('uint8') #if mask is 0 or 2, set to 0, else set to 1
    imgSeg=img*maskGC[:,:,np.newaxis] #apply the mask to the image
    plt.imshow(imgSeg)

    #you can physically draw in areas you do / dont want to keep to improve the segmentation

    plt.show()

if __name__=='__main__':
    GraphCut()

    
