#! /usr/bin/env python
"""Demo of a feature tracker.

NB: Untested, unfinished!"""

import cv2 as cv
from tracking import *

img1 = cv.imread( "../data/frame27.jpeg")
img2 = cv.imread( "../data/frame28.jpeg")


optical_flow(img1, img2, numpts=5, debug=10)
