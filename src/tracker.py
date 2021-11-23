#! /usr/bin/env python
"""Demo of a feature tracker.

The second frame is saved with annotations
showing feature points in red, feature points
from the previous frame in yellow, and estimated
motion vectors in blue.

Note that the motion calculation is not good.
Multi-scale tracking is probably required to
give meaningful results.

The test images show a lot of weak feature points
which do not match between images, with many random
motion vectors.
"""

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
