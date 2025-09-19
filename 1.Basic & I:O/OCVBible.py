import cv2 as cv #pip install opencv-python
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pa





'''Reading and writing images___________________________________________________________________________________'''

def readImages(): #imports from folder and displays an image
    root=os.getcwd()    #gets file directory (current work directory)
    imgPath = os.path.join(root, 'LearnCVImages/catpic1.jpg') #allows for images and the code files to be stored seperately, nice and cleaner
    img = cv.imread(imgPath)    #reads the image into the code, matrix of MxNx3
    debug =1                    #use to look at image data in data viewer(excel)
    cv.imshow('img',img)        #shows image from matrix read
    cv.waitKey(0)               #waits for inifte time (0)



def writeImage(): #collects from folder and image and saves it as output
    root=os.getcwd()    #gets file directory (current work directory)
    imgPath = os.path.join(root, 'LearnCVImages/catpic1.jpg') #allows for images and the code files to be stored seperately, nice and cleaner
    img = cv.imread(imgPath)    #reads the image into the code, matrix of MxNx3
    outpath=os.path.join(root, 'LearnCVImages/ouput.jpg')
    cv.imwrite(outpath,img) #writes and saves the image "img" to the output path location




'''reading and writing videos--------------------------------------------------------------------------------------'''

def VideoFromWebcam(): #reads a video from the laptop webcam
    cap = cv.VideoCapture(1) #video capture object, 0- is the ownly camera i have 0 indexed, laptopcamera is 1

    if not cap.isOpened(): #checks if web cam is already in use, if so stops trying
        exit()

    while True:
        ret, frame = cap.read() # returns a return value and each image (ret and frame)
        if ret:
            cv.imshow('Webcam', frame) #if retruning images, then show the webcam frames

        if cv.waitKey(1) == ord('q'):   #breaks out of infite loop once q is pressed
            break

    cap.release()                       #stops using the camera
    cv.destroyAllWindows                #closes all windows



def videoFromFile(): #reads a video from file
    root=os.getcwd()
    vidpath=os.path.join(root, 'LearnCVImages/paris.M4V')
    cap=cv.VideoCapture(vidpath) #captures video from file

    while cap.isOpened():
        ret, frame= cap.read() #reads video frames from files
        cv.imshow('video', frame) #shows images, and names video "video"
        delay = int(1000/60) #assuming 1000milseconds
        if cv.waitKey(delay) == ord('q'): #waits till q is pressed to quit
            break



def writeVideoToFile():
    cap = cv.VideoCapture(1)                #video capture object, 0- is the ownly camera i have 0 indexed, laptopcamera is 1

    fourcc = cv.VideoWriter_fourcc(*'mp4v') #MPEG compression encoding
    root = os.getcwd()
    outPath=os.path.join(root,'LearnCVImages/webcam.mp4') #creates path fro saved file as avi file
    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) #gets camera height and width, as it can fail due to mismatching
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out=cv.VideoWriter(outPath, fourcc, 30.0, (width,height)) #outputs video, to path, with fourcc format, 20 frames per second, and 640x480 window view.



    if not cap.isOpened():                  #checks if web cam is already in use, if so stops trying
        exit()


    while cap.isOpened():
        ret, frame = cap.read()             # returns a return value and each image (ret and frame)
        if ret:
            out.write(frame)                #writes the frame
            cv.imshow('Webcam', frame)      #if retruning images, then show the webcam frames
        if cv.waitKey(1) == ord('q'):       #breaks out of infite loop once q is pressed
            break

    cap.release()                           #stops using the camera
    out.release()                           #stops using the output write
    cv.destroyAllWindows                    #closes all windows




'''reading and writing pixels----------------------------------------------------------------------------'''































if __name__=='__main__':
    print(cv.__version__)
    #readImages()
    #writeImage()
    #VideoFromWebcam()
    #videoFromFile()
    writeVideoToFile()
















'''Theory---------------------------------------------------------------------------------------------------------'''




'''Pictures'''
#pictures are represented as a large matrix with numbers(representing each pixel)


'''Pixel Range'''
    # 8 bit represenattion, 2**8= 256
    # Min:0 (black)
    # Max: 255 (white)

    #BGR , three numbers representing how much Blue Green Red, ranges between 0-255

'''image Dimension'''
    # Dim:(M,N,3)
    # M-height, y of image
    # N-width , x of image
    # 3, corresonds to the level of BGR, B=0, G=1, R=2, if grey scale will be 1D here

'''Reading and writing Images'''
    # Read to interpret the images, for some image processing
    # Write images for saving

'''reading writing viudeo'''
    #each video fram is though as a picture
    #each fam has their own MxNx3 matrix

'''read and writing pixels'''
    #why, needed to interpret part of the image
    #save for analysis