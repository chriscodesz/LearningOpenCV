# Converts a 2D image into frequency domain
# Used for filtering an ddata compression
# can also be used as image compression as tiny low frequency area represenmts the final Low pass image, see below



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def fourierTransform():
    root = os.getcwd()
    imgPath= os.path.join(root, 'LearnCVImages/pole.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE) 

    plt.figure()
    plt.subplot(231)
    plt.imshow(img,cmap='gray')

    imgDFT=cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT) #performs the fourier transform, outputs a complex number
    imgDFT_DB = 20*np.log(cv.magnitude(imgDFT[:,:,0], imgDFT[:,:,1])) #gets the magnitude of the complex number, converts to decibels
    plt.subplot(232)
    plt.imshow(imgDFT_DB,cmap='gray')
    plt.title('DFT, magnitude spectrum')
    plt.xlabel('frequencies')
    plt.ylabel('magnitude')

    imgDFTshift=np.fft.fftshift(imgDFT) #shifts the zero frequency component to the center of the spectrum
    imgDFTshiftDB=20*np.log(cv.magnitude(imgDFTshift[:,:,0], imgDFTshift[:,:,1]))
    plt.subplot(233)
    plt.imshow(imgDFTshiftDB,cmap='gray')

    r,c=img.shape #creating a mask so we can apply a low pass filter to low frequencies in the centre of the DFT
    mask=np.zeros((r,c,2),np.uint8)
    offset=50
    mask[int(r/2)-offset:int(r/2)+offset, int(c/2)-offset:int(c/2)+offset]=1 #creates a square mask in the centre of the image
    plt.subplot(234)
    plt.imshow(mask[:,:,0],cmap='gray')

    imgDFTshiftLP= imgDFTshift* mask #applies the mask to the DFT
    imgDFTshiftLPDB=20*np.log(cv.magnitude(imgDFTshiftLP[:,:,0], imgDFTshiftLP[:,:,1]))
    plt.subplot(235)
    plt.imshow(imgDFTshiftLPDB,cmap='gray')

    imgInvDFT_LP=np.fft.ifftshift(imgDFTshiftLP) #inverse shift
    imgDFTLP=cv.idft(imgInvDFT_LP)
    img_LP=cv.magnitude(imgDFTLP[:,:,0], imgDFTLP[:,:,1]) #gets the magnitude of the complex number
    plt.subplot(236)
    plt.imshow(img_LP,cmap='gray')
    plt.title('Low Pass Filtered Image')


    #filters
    coef = cv.getGaussianKernel(7,5) #creates a 1D gaussian kernel
    gaussianKernel=coef@coef.T #creates a 2D gaussian kernel by multiplying the 1D kernel by its transpose
    laplacianKernel=np.array([[0,1,0],[1,-4,1],[0,1,0]]) #creates a laplacian kernel

    plt.figure()
    plt.subplot(121)
    gaussFFT=np.fft.fft2(gaussianKernel) #performs fourier transform on the kernel
    gaussshift=np.fft.fftshift(gaussFFT) #shifts the zero frequency component to the center of the spectrum
    gausMag = np.log(np.abs(gaussshift)+1) #gets the magnitude of the complex number
    plt.imshow(gausMag,cmap='gray') 
    plt.title('Gaussian Filter Frequency Response, Low Pass')

    plt.subplot(122)
    lapFFT=np.fft.fft2(laplacianKernel) 
    lapshift=np.fft.fftshift(lapFFT) 
    lapMag = np.log(np.abs(lapshift)+1) 
    plt.imshow(lapMag,cmap='gray')
    plt.title('Laplacian Filter Frequency Response, High Pass')

    #Note: Gaussian is light middle, and dark on edges, so it is a low pass filter
    #Laplacian is dark in middle and light on edges, so it is a high pass filter



    plt.show()


if __name__=='__main__':
    fourierTransform()

