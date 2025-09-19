#finding both position and orientation in image or video
# for augmented reality, activity recongnition, filma and animation



import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from enum import Enum

class DrawOption(Enum): #allows us to toggle between axes and cube
    AXES = 1
    CUBE = 2

def drawAxes(img, corners, imgpts):
    def tupleOfInts(arr):
        return tuple(int(x) for x in arr) # allows for input imgpoints to be stored in a tuple for adding the lines
    
    corner = tupleOfInts(corners[0].ravel())
    img = cv.line(img,corner,tupleOfInts(imgpts[0].ravel()),(255,0,0),5) # adds line from corner to image point and changes color with width 5
    img = cv.line(img,corner,tupleOfInts(imgpts[1].ravel()),(0,255,0),5)
    img = cv.line(img,corner,tupleOfInts(imgpts[2].ravel()),(0,0,255),5)
    return img




def drawCube(img,imgpts):
    imgpts=np.int32(imgpts).reshape(-1,2) #read in image points

    #Add green plane
    img= cv.drawContours(img,[imgpts[:4]],-1,(0,255,0),-3) #creates green plane on image

    #Add box borders
    for i in range(4):
        j=i+4
        img = cv.line(img,tuple(imgpts[i]),tuple(imgpts[j]),(255),3)
        img = cv.drawContours(img,[imgpts[4:]],-1,(0,0,255),3)
    return img




def poseEstimation (option: DrawOption):
    # Retrieve calibration parameters (from CameraCalibration)
    root=os.getcwd()
    paramPath= os.path.join(root,'Calibration.npz')
    data= np.load(paramPath)
    camMatrix = data['camMatrix']
    distCoeff= data['distCoeff']

    # Read Image
    calibrationDir = os.path.join(root, 'CalibrationImages')
    imgPathList= glob.glob(os.path.join(calibrationDir,'*.jpg'))

    # Initialize
    #initialise
    nRows = 8                   #look at a 9*7 chessboard only interested in inmternal corners
    nCols = 6
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)  #termination criteria
    worldPtsCur = np.zeros((nRows*nCols,3),np.float32)                          # creates place holders for number of worldpoints
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)

    # World Points of object to be drawn
    axis = np.float32([[3,0,0],[0,3,0],[0,0,-3]]) #creates axes 3 chess board length from each other
    cubeCorners = np.float32(([0,0,0],[0,3,0],[3,3,0],[3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3])) #creates cornes 3 chess board rows along from each other

    # Find Corners
    for curImgPath in imgPathList:
        imgBGR = cv. imread(curImgPath)
        imgGray= cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg=cv.findChessboardCorners(imgGray, (nRows,nCols),None) # finds chesboard corners on the image

        if cornersFound ==True:
            cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1), termCriteria)
            _, rvecs,tvecs = cv.solvePnP(worldPtsCur,cornersRefined,camMatrix,distCoeff) # Finds roation and translation vectors

            if option == DrawOption.AXES:
                imgpts,_= cv.projectPoints(axis, rvecs, tvecs, camMatrix, distCoeff) # projects points on image, from found axis, roation tranlastional vectors and camera options
                imgBGR= drawAxes(imgBGR, cornersRefined, imgpts)                    # draw the line sonto the chess board from home made subplot

            if option == DrawOption.CUBE:
                imgpts,_= cv.projectPoints(cubeCorners, rvecs, tvecs, camMatrix, distCoeff)
                imgBGR= drawCube(imgBGR, imgpts)

            cv.imshow('Chessboard', imgBGR)
            cv.waitKey(1000)

if __name__=='__main__':
    poseEstimation(DrawOption.CUBE)
    poseEstimation(DrawOption.AXES)





