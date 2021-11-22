#! /usr/bin/env python
"""Demo of a feature detection.

NB: Untested, unfinished!"""

import cv2 as cv
from tracking import *

img = cv.imread( "../data/frame28.jpeg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

getHarris(img)
