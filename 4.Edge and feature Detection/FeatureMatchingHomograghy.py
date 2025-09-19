# Find corresponding images between two images y findig which points map between each other (isolates points we want)
# Used in image sticthing, SLAM, otion tracking and removing outliers
# makes feature detection cleaner, removes bad matches that dont fit the model
# also uses RANSAC to remove outliers


import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def featureMatchingHomography():
    root = os.getcwd()
    img1Path= os.path.join(root, 'LearnCVImages/irishStreet1.jpg')
    img2Path= os.path.join(root, 'LearnCVImages/irishStreet2.jpg')
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE) 
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE) 

    sift=cv.SIFT_create()
    keypoints1, descriptor1 =sift.detectAndCompute(img1,None) #detect keypoints
    keypoints2, descriptor2 =sift.detectAndCompute(img2,None) #detect keypoints

    FlANN_INDEX_KDTREE=1
    nKDtrees=5
    nLeafChecks=50
    nNeighbours=2
    indexParams= dict(algorithm=FlANN_INDEX_KDTREE, trees=nKDtrees)
    searchParams= dict(checks=nLeafChecks)
    flann=cv.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(descriptor1,descriptor2, k=nNeighbours)
    
    goodMatches=[]
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodMatches.append(m)

    minGoodMatches=20

    if len(goodMatches)>minGoodMatches:
        srcPts = np.float32([ keypoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2) #queryIdx is index of descriptor in first image, source pints
        dstPts = np.float32([ keypoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2) #trainIdx is index of descriptor in second image, destination points
        errorThreshold =5
        M, mask = cv.findHomography(srcPts, dstPts, cv.RANSAC, errorThreshold) #find homography matrix using RANSAC to remove outliers
        matchesMask = mask.ravel().tolist() #list of 1s and 0s, 1 if match is good, 0 if not
        h,w =img1.shape
        imgBorder= np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2) #corners of image 1
        warpedImgBorder=cv.perspectiveTransform(imgBorder,M) #warp corners to image 2
        img2=cv.polylines(img2,[np.int32(warpedImgBorder)],True,255,3,cv.LINE_AA) #draw border on image 2, line AAnti aliasing
    else:
        print(f'Not enough good matches are found {len(goodMatches)}/{minGoodMatches}')
        matchesMask = None  
    
    green=(0,255,0)
    drawParams = dict(matchColor = green, #draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, #draw only inliers
                   flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) #draw only inlier points
    
    imgMatch=cv.drawMatches(img1,keypoints1,img2,keypoints2,goodMatches,None,**drawParams) #draw matches

    plt.figure()
    plt.imshow(imgMatch, 'gray')
    plt.show()

    #creates a white box to indicate the previous view point.



if __name__=='__main__':
    featureMatchingHomography()