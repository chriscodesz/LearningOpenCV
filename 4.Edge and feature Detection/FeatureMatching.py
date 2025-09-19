# Finds corresponding features between two images
# Useful for image stitching, slam and motion tracking applications
#  Bruteforce not super fast, more for offline processing

# can reduce bad feature matching with homography

import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

def bruteforce():
    root = os.getcwd()
    img1Path= os.path.join(root, 'LearnCVImages/irishStreet1.jpg')
    img2Path= os.path.join(root, 'LearnCVImages/irishStreet2.jpg')
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE) 
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE) 

    orb = cv.ORB_create()
    keypoints1, descriptor1 =orb.detectAndCompute(img1,None) #detect keypoints
    keypoints2, descriptor2 =orb.detectAndCompute(img2,None) #detect keypoints
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True) #create brute force matcher object
    matches = bf.match(descriptor1,descriptor2) #find matches
    matches = sorted(matches, key = lambda x:x.distance) #sort matches by distance
    nMatches=20
    imgMatch=cv.drawMatches(img1,keypoints1,img2,keypoints2,matches[:nMatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) #draw matches

    plt.figure()
    plt.imshow(imgMatch)
    plt.show()

def knnBruteforce(): #uses k nearest neighbors to find matches, more robust than regular brute force
    root = os.getcwd()
    img1Path= os.path.join(root, 'LearnCVImages/irishStreet1.jpg')
    img2Path= os.path.join(root, 'LearnCVImages/irishStreet2.jpg')
    img1 = cv.imread(img1Path, cv.IMREAD_GRAYSCALE) 
    img2 = cv.imread(img2Path, cv.IMREAD_GRAYSCALE) 

    sift=cv.SIFT_create()
    keypoints1, descriptor1 =sift.detectAndCompute(img1,None) #detect keypoints
    keypoints2, descriptor2 =sift.detectAndCompute(img2,None) #detect keypoints
    
    bf = cv.BFMatcher() #create brute force matcher object
    nNeighbors=2
    matches = bf.knnMatch(descriptor1,descriptor2, k=nNeighbors) 
    goodMatches=[]
    testRatio=0.75 #lower is more strict
    for m, n in matches:
        if m.distance < testRatio*n.distance:
            goodMatches.append(m)

    imgMatch=cv.drawMatches(img1,keypoints1,img2,keypoints2,goodMatches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) #draw matches

    plt.figure()
    plt.imshow(imgMatch)
    plt.title('KNN Matches')
    plt.show()





def FLANN():
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
    matchesMask= [[0,0] for i in range(len(matches))]
    testRatio=0.75
    for i ,(m,n) in enumerate(matches):
        if m.distance < testRatio*n.distance:
            matchesMask[i]=[1,0]
    drawParams= dict(matchColor=(0,255,0), singlePointColor=(255,0,0), matchesMask=matchesMask, flags=cv.DrawMatchesFlags_DEFAULT)
    imgMatches=cv.drawMatchesKnn(img1,keypoints1,img2,keypoints2,matches,None,**drawParams)

    plt.figure()
    plt.imshow(imgMatches)
    plt.show()

    #should eb alot faster than brute force, good for large datasets









if __name__=='__main__':
    #bruteforce()
    #knnBruteforce()
    FLANN()