import cv2
import numpy as np
import os
from threading import Thread

#cap.set(10, 0)
        
def preprocIMG(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    
    return imgThres
    
def getContures(img):
    biggest = np.array([])
    maxArea = 0
    conturs, hiers = cv2.findContours(img, 
                                      cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_NONE)
    for cnt in conturs:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgCont, biggest, -1, (255,0,0), 20)     
    return biggest

def reorder_points(points):
    
    cont_points = points.reshape((4,2))
    mpn = np.zeros((4,1,2), np.int32)
    a = cont_points.sum(1)

    mpn[0] = cont_points[np.argmin(a)]
    mpn[3] = cont_points[np.argmax(a)]
    diff = np.diff(cont_points, axis=1)
    mpn[1] = cont_points[np.argmin(diff)]
    mpn[2] = cont_points[np.argmax(diff)]
    return mpn
    
def getWarp(img, biggest):
    if len(biggest) > 0:
        biggest = reorder_points(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0], [A4_1, 0],
                           [0, A4_2], [A4_1, A4_2]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOut = cv2.warpPerspective(img, matrix, (A4_1, A4_2))
        
        imgCrop = imgOut[8:imgOut.shape[0]-8, 8:imgOut.shape[1]-8]
        imgCrop = cv2.resize(imgCrop, (A4_1, A4_2))
        imgOut = imgCrop
    else:
        imgOut = img
    
    return imgOut

def stackImg(imgarray):
    
    for i in range(len(imgarray)):
        cv2.resize(imgarray[i], (240, 80), None, 0.5, 0.5)
    
    imgvert = np.hstack((imgarray[0],
                         imgarray[1],
                         imgarray[2]))
    
    return imgvert
    
queue = 0
fq = 0



try:
    os.makedirs('images') #if !path create path, else none
    #print('we created the ' + path)
except:
    #print('we cant create the path')
    pass

WIDTH = 1280
HEIGHT = 720
FPS = 15

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(WIDTH,HEIGHT),framerate=FPS):
        # Initialize the PiCamera and the camera image stream
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FPS, FPS)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH , WIDTH)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT , HEIGHT)
            
        # Read first frame from the stream
        (_, self.frame) = self.vid.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.vid.release()
                return

            # Otherwise, grab the next frame from the stream
            (_, self.frame) = self.vid.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True


frameWidth = 1280
frameHeigth = 720

A4_2 = 630
A4_1 = 891
cons = 5

width = 1280
heigth = 720
cap = VideoStream().start()

while True:
    img = cap.read()
    #cv2.resize(img, (width, heigth))
    
    imgThres = preprocIMG(img)
    
    imgCont = img.copy()
    biggest = getContures(imgThres)
    warpedIMG = getWarp(img, biggest)
    
    cv2.imshow('Real Camera img and contours', imgCont)
    cv2.imshow('Wrapped IMG', warpedIMG)
    
    if cv2.waitKey(1) & 0xFF == 27:
        cap.stop()
        break
        
    if cv2.waitKey(1) & 0xFF == 13:
        
        step = queue % cons
        if step == 0:
            fq += 1
            try:
                os.makedirs('images/' + str(fq)) #if !path create path, else none
                #print('we created the ' + path)
            except:
                #print('we cant create the path')
                pass
        
        cv2.imwrite('images/' + str(fq) +'/'+ str(queue) + '.jpg', warpedIMG)
        queue += 1
        print(str(queue) + '.jpg' + ' is saved')
        
cap.release()
cv2.destroyAllWindows()
        