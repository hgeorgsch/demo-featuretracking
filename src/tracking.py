"""Basic routines for corner detection and tracking.

NB: Untested, unfinished!"""

import cv2 as cv
import numpy as np
from scipy import signal

__all__ = [ "optical_flow" ]

def getGradient(img):
    Ix = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=0, ksize=5)
    Iy = cv.Sobel(img, ddepth=cv.CV_32F, dx=0, dy=1, ksize=5)
    return (Ix,Iy)


def getMotion(Ix, Iy, It, x):
    """Not used"""
    pass

def optical_flow(img1, img2, numpts=5, debug=0):
    """
    Calculate motion/optical flow between two frames

        Parameters:
            img1 (numpy array): previous frame
            img2 (numpy array): current frame
            numpts (int=5): number of feature points to track
            debug (int=0): print debug information if >0
    """


    # Step 1.  Spacial and temporal derivatives
    (Ix,Iy) = getGradient(img2)
    It = img2 - img1

    # Step 2.  Calculations for the G matrix
    # Auxiliaries
    ixix = Ix * Ix
    ixiy = Ix * Iy
    iyiy = Iy * Iy
    ixit = Ix * It
    iyit = Iy * It

    # Window for summation
    sumf = np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])

    # Summation over the window
    ixs = signal.convolve2d(ixix, sumf, mode='full', boundary='symm')  
    ixys = signal.convolve2d(ixiy, sumf, mode='full', boundary='symm')  
    iys = signal.convolve2d(iyiy, sumf, mode='full', boundary='symm')  
    ixits = signal.convolve2d(ixit, sumf, mode='full', boundary='symm')  
    iyits = signal.convolve2d(iyit, sumf, mode='full', boundary='symm')  

    # Step 3.  Feature points (Harris detector)

    # Find corners
    cx_answer = cv.cornerHarris(img2, 5, 5, 0.06)
    if debug > 0:
        print( "Return value from the Harris Detector:" )
        print( cx_answer )


    ## What happens here?
    # We should sort the corners by strength and pick top five or whatever.
    min_cx = np.min(cx_answer)
    max_cx = np.max(cx_answer)
    T = min_cx + (max_cx - min_cx) * 50 / 100
    T = T * 3

    # -G(x) - lb(x, t)
    
    # Step 4.  Motion for each corner
    # Iterate over corners
    # TODO: This is only one pixel per corner, filter around corners?
    for i in range(cx_answer.shape[0]):
        for j in range(cx_answer.shape[1]):
            if cx_answer[i, j] > T:
                a = ixs[i][j]
                c = iys[i][j]
                b = ixys[i][j]
                Gmatrix = np.array([[a, b], [b, c]])
                bvector = np.array([[ixits[i][j]],
                               [iyits[i][j]]])
                u = - np.linalg.inv(Gmatrix) @ bvector