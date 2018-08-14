# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import imutils

body_classifier = cv2.CascadeClassifier('haarcascade_pedestrians.xml')

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))
 
# allow the camera to warmup
time.sleep(0.1)

for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = f.array
    frame = imutils.resize(frame, width=500)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.imshow('Pedestrians', frame)

    #if cv2.waitKey(1) == 13: #13 is the Enter Key
     #   break
    cv2.imshow('img',frame)

    rawCapture.truncate(0)
    
    k = cv2.waitKey(1) & 0xFF 
    if k == 13:
        break
    
cv2.destroyAllWindows()
