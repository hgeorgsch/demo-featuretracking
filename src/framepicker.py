#! /usr/bin/env python

"""
This script slowly displays a video frame by frame.
When the user hits space, the frame is saved.
The filename shows the frame number.
"""

import numpy as np
import cv2 as cv

cap = cv.VideoCapture("video.mp4")
if not cap.isOpened():
    print("Cannot open camera")
    exit()

framecount = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    key = cv.waitKey(400) 
    framecount += 1
    if key == ord(' '):
        cv.imwrite("frame%2d.jpeg" % (framecount,), gray )
        print( "Saving no. ", framecount )
    else:
        print( "Frame no. ", framecount )
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
