#! /usr/bin/env python
"""Demo of a feature detection.

It loads a test image and draws red circles 
around the feature points.  The result is
saved to a file in the current directory.
"""

import cv2 as cv
from tracking import *

img = cv.imread( "../data/frame28.jpeg")
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

print( "Loaded image.  Size ", grey.shape )

cornerlist = getHarris(grey,debug=1)

for (pos,s) in cornerlist:
    # print ( pos, grey.shape )
    (y,x) = pos
    cv.circle(img,(x,y),10,(0,0,255))
cv.imwrite( "features.jpeg", img)
