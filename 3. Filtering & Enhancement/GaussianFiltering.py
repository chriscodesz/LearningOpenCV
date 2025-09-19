# uses a gaussian kernel for conviolution, bigger and strong in the middel of the kernel
#smoothing noise and general preprocessing
#centre of the kernel takes heighest filter weighting
#look at word, sigma term determines how fat or skinny bell kernel is, small =skinny, large = fat

    #Gaussian Filtering → Best for general noise reduction, smooth blur, edge-aware smoothing, natural-looking blur.
    #Average Filtering → Only good for quick & simple smoothing, but often inferior to Gaussian.
    #Median Filtering → Best for removing salt-and-pepper noise while keeping edges sharp.
    #General Convolution → Flexible framework (Gaussian is one special kernel).


import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np

def callback(input):
    pass #call back filter track bar function will do nothing rn


def gaussianKernel(size, sigma):#creates a gausian kernel
    kernel=cv.getGaussianKernel(size, sigma) #only creates 1D kernel
    kernel=np.outer(kernel,kernel) #creates 2D kernel
    return kernel


def GaussianFiltering():
    root = os.getcwd() 
    imgPath= os.path.join(root, 'LearnCVImages/castle.jpg')
    img = cv.imread(imgPath) 

    n=51 #kernel size 51x51
    fig = plt.figure()
    plt.subplot(121)
    kernel = gaussianKernel(n,8) #size n sigma=8
    plt.imshow(kernel) #shows plane 2d 51x51 kernel
   

    ax = fig.add_subplot(122,projection='3d') #adds axis to figure 122, for a 3D view
    x = np.arange(0,n,1) #creates 0-n array spacing of one
    y = np.arange(0,n,1)
    X,Y=np.meshgrid(x,y) #creates mesh grid
    ax.plot_surface(X,Y,kernel,cmap='viridis') #displays meshgrid against the kernel
    plt.show()

    winName='gaus filter' # for trackbar for controlling sigma
    cv.namedWindow(winName)
    cv.createTrackbar('sigma',winName,1,20,callback) # creates a track bar from 1-20

    height,width,_=img.shape 
    scale=1/4
    width=int(width*scale)
    height=int(height*scale)
    img=cv.resize(img,(width,height)) #increasing image size to reduce residula white spots

    while True:
        if cv.waitKey(1)==ord('q'):
            break
            
        sigma = cv.getTrackbarPos('sigma',winName)
        imgFilter = cv.GaussianBlur(img,(n,n),sigma) #blurs the image using gaussina blur, using nxn kernel and sigma from trackbar
        cv.imshow(winName,imgFilter)

    cv.destroyAllWindows()






if __name__=='__main__':
    GaussianFiltering()