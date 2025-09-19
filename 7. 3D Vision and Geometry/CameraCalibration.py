# used to obtain camera parameters
# Intrinsics
    #focus length (f)
    #prinicpal point (cx,cy) or also called px,py
    #distortion (k1,k2,p1,p2,k3)
# Extrinsics
    #Rotation (R-3x3 matrix)
    #translation (T - 3x1 matrix)

#Good for AR (AUgmented reality), SLAM, 3D reconstruction, remove image distortion
#works by finding inner corners of an chessboard image

#NOTEE: NOT PERFECT

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

def calibrate(showPics=True):
    #Read Image
    root = os.getcwd() 
    calibrationDir = os.path.join(root, 'CalibrationImages')
    imgPathList= glob.glob(os.path.join(calibrationDir,'*.jpg')) #gets all the images inside the directory path, ie multiple images

    #initialise
    nRows = 8                   #look at a 9*7 chessboard only interested in inmternal corners
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)  #termination criteria
    worldPtsCur = np.zeros((nRows*nCols,3),np.float32)                          # creates place holders for number of worldpoints
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsList=[]                     #create two empty list to store points
    imgPtsList=[]

    #Find Corners
    for curImgPath in imgPathList:
        imgBGR= cv.imread(curImgPath)
        imgGray= cv.cvtColor(imgBGR,cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray,(nRows,nCols), None) # no flags=None, returns nx1x2 array, x,y coords of the corners, here 2x1x2 as garscale

        # cv.imshow('gray', imgGray)
        # cv.waitKey(0)
        
        if cornersFound ==True:
            
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1), termCriteria) #this then finds a more accurate location of the corners (11,11) window size, (-1,-1) = no zeros zone
            imgPtsList.append(cornersRefined)               #add refined corners to the list
            if showPics:
                cv.drawChessboardCorners(imgBGR, (nRows, nCols), cornersRefined,cornersFound)     #plots the corners on the img
                cv.imshow('ChessBoard',imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows

    #calibrate1
    repError,camMatrix, distCoeff,rvecs,tvecs=cv.calibrateCamera(worldPtsList,imgPtsList, imgGray.shape[::-1],None,None)
    #Reprojection error:float, maMtrix:3x3 array, distCoeff (1x5 array), rvecs=rotation vector nPics tuple of 3x1 array, roation, tvecs translation vector of nPics tuple 3x1
    print('Camera Matrix: \n', camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))

    #Save calibration Parameters, to be used in anothe rproject
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder,'calibration.npz')
    np.savez(paramPath, repError=repError,camMatrix=camMatrix, distCoeff=distCoeff,rvecs=rvecs,tvecs=tvecs)
    return camMatrix,distCoeff





def removeDistortion(camMatrix, distCoeff):
    
    root = os.getcwd()
    imgPath = os.path.join(root, 'LearnCVImages/Box.jpg')
    img = cv.imread(imgPath)
    height,width=img.shape[:2]
    camMatrixNew, roi=cv.getOptimalNewCameraMatrix(camMatrix,distCoeff, (width,height),1,(width,height))
    imgUndist=cv.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

    #draw line to see Distortion change
    cv.line(img,(906,447),(954,1250),(0,0,255),2)

    #maps lines to the distorted image plane
    pts = np.array([[[906,447]], [[954,1250]]], dtype=np.float32)
    undistorted_pts = cv.undistortPoints(
    pts,
    cameraMatrix=camMatrix,
    distCoeffs=distCoeff,
    P=camMatrixNew )

    # Extract back to tuple form
    p1 = tuple(undistorted_pts[0,0].astype(int))
    p2 = tuple(undistorted_pts[1,0].astype(int))

    # Draw on undistorted image
    cv.line(imgUndist, p1, p2, (0,0,255), 2)
    #cv.line(imgUndist,(906,447),(954,1250),(0,0,255),2)


    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()










def runcalibration():
    calibrate(showPics=True)

def runremoveDistortion():
    camMatrix,distCoeff=calibrate(showPics=False)
    removeDistortion(camMatrix, distCoeff)




if __name__=='__main__':
    #runcalibration()
    runremoveDistortion()



