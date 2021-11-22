#! /usr/bin/env python
"""Demo of a feature tracker.

NB: Untested, unfinished!"""

import cv2 as cv
from tracking import *

img1 = cv.imread( "../data/frame27.jpeg")
img2 = cv.imread( "../data/frame28.jpeg")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

optical_flow(img1, img2, numpts=5, debug=10)
