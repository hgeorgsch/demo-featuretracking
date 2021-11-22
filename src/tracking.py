"""Basic routines for corner detection and tracking.

NB: Untested, unfinished!"""

import cv2 as cv
import numpy as np
from scipy import signal

__all__ = [ "optical_flow", "getHarris" ]

def getGradient(img):
    Ix = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=0, ksize=5)
    Iy = cv.Sobel(img, ddepth=cv.CV_32F, dx=0, dy=1, ksize=5)
    return (Ix,Iy)


def getMotion(Ix, Iy, It, x):
    """Not used"""
    pass

def tileSelect(tile,count=5,debug=0):
    """
    Return sorted feature points with score.

    Given an array of corner detection scores,
    sort and return the required number of corners.

        Parameters:
            tile (numpy array): array of corner detection scores
            count (int=5): number of feature points to return
            debug (int=0): print debug information if >0
    """
    if debug > 1:
       print( "Tile input ", tile )
    ((i,j),tile) = tile
    cornerlist = [ ((x+i,y+j),s) 
            for ((x,y),s) in np.ndenumerate(np.abs(tile)) ]
    cornerlist.sort(key=lambda x : x[1] )
    cornerlist = cornerlist[:count]
    if debug > 1:
        print( "tileSelect returns ", cornerlist )
    return cornerlist

def getHarris(img,count=5,tiling=(10,10),debug=0):
    """
    Tiled corner detection.

    Run the Harris corner detector at identify the required
    number of corners for each tile of the image.

        Parameters:
            img (numpy array): image
            count (int=5): number of feature points per tile
            tiling (int,int=(10,10)): number of tiles to use
            debug (int=0): print debug information if >0
    """
    # Find corners
    cx_answer = cv.cornerHarris(img, 5, 5, 0.06)
    if debug > 0:
        print( "Return value from the Harris Detector:" )
        print( cx_answer )

    # Tiling
    ## Tile parameters
    (Nx,Ny) = cx_answer.shape
    (Tx,Ty) = tiling
    (Sx,Sy) = (int(np.ceil(Nx/Tx)),int(np.ceil(Ny/Ty)))
    if debug > 0:
        print( "Image size ", (Nx,Ny), ". Tile size ", (Sx,Sy) )
    ## Making the tiles.
    ## This functional style is more elegant than what we should expect at this level.
    tiles = [ ((Sx*i,Sy*j),cx_answer[Sx*i:Sx*(i+1),Sy*j:Sy*(j+1)]) 
              for (i,j) in np.ndindex(tiling) ]
    cornerlist = []
    for tile in tiles:
        cornerlist.extend( tileSelect(tile,count=count,debug=debug) )
    ## Sort and return.
    ## Sorting may not be important at this stage, but why not.
    cornerlist.sort(key=lambda x : x[1] )
    return cornerlist

    
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
    cornerlist = getHarris(img2,debug=debug)

    # Step 4.  Motion for each corner
    # Iterate over corners
    # TODO: This is only one pixel per corner, filter around corners?
    motionlist = []
    for ((i,j),s) in cornerlist:
        if debug > 2: print( (i,j), s )
        a = ixs[i][j]
        c = iys[i][j]
        b = ixys[i][j]
        Gmatrix = np.array([[a, b], [b, c]])
        bvector = np.array([[ixits[i][j]], [iyits[i][j]]])
        # -G(x) - lb(x, t)
        u = - np.linalg.inv(Gmatrix) @ bvector
        motionlist.extend( ((i,j),s,u) )
        if debug > 0: print ( "Motion ", u )
