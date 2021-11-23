#! /usr/bin/env python
"""Demo of a feature tracker.

NB: Untested, unfinished!"""

import cv2 as cv
from tracking import *

img1 = cv.imread( "../data/frame27.jpeg")
img2 = cv.imread( "../data/frame28.jpeg")
grey1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
grey2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

ml = getMotion(grey1, grey2, numpts=5, debug=3)
img = drawMotionList( img2, ml )

cornerlist = getHarris(grey1,debug=1)

for (pos,s) in cornerlist:
    (y,x) = pos
    cv.circle(img,(x,y),1,(0,255,255),-1)
cv.imwrite( "motions.jpeg", img)
