"""Basic routines for corner detection and tracking.

The getHarris() function and its auxiliaries seem to work.
The tracker remains to be tested.
"""

import cv2 as cv
import numpy as np
from scipy import signal

__all__ = [ "getMotion", "getHarris", "drawMotionList" ]

def getGradient(img):
    Ix = cv.Sobel(img, ddepth=cv.CV_32F, dx=1, dy=0, ksize=5)
    Iy = cv.Sobel(img, ddepth=cv.CV_32F, dx=0, dy=1, ksize=5)
    return (Ix,Iy)


def getMotion(Ix, Iy, It, x):
    """Not used"""
    pass

def filterPoints(cl,count=5,separation=9,debug=0):
    """
    Filter a list of scored Harris corners to enforce a given
    separation and limit the number of points.
    """
    rl = []
    n = 0
    for ((x,y),s) in cl:
        sd = [ (x-x1)**2+(y-y1)**2 for ((x1,y1),s) in rl ]
        if (len(sd) == 0) or (min(sd) > separation**2):
            rl.extend([((x,y),s)])
            n += 1
        if n >= count: break
    return rl

def tileSelect(tile,count=5,separation=9,debug=0):
    """
    Return sorted feature points with score.

    Given an array of corner detection scores,
    sort and return the required number of corners.

        Parameters:
            tile (numpy array): array of corner detection scores
            count (int=5): number of feature points to return
            debug (int=0): print debug information if >0
    """
    ((i,j),tile) = tile

    if debug > 2:
       print( "Tile input ", tile )
    if debug > 0:
       print( "Tile Harris Score Range ", max(tile.flatten()), min(tile.flatten()) )

    cornerlist = [ ((x+i,y+j),s) 
            for ((x,y),s) in np.ndenumerate(np.abs(tile)) ]
    cornerlist.sort(key=lambda x : x[1], reverse=True )
    cornerlist = filterPoints(cornerlist,count=count,separation=separation)
    if debug > 2:
        print( "tileSelect returns ", cornerlist )
    return cornerlist

def getHarris(img,count=5,separation=20,tiling=(10,10),debug=0):
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
    tiles = [ ((Sx*i,Sy*j),cx_answer[Sx*i:Sx*(i+1),Sy*j:Sy*(j+1)]) 
              for (i,j) in np.ndindex(tiling) ]
    cornerlist = []
    for tile in tiles:
        cornerlist.extend( tileSelect(tile,separation=separation,count=count,debug=debug) )
    ## We need to sort and filter to avoid clusters of
    ## feature points along tile boundaries.
    cornerlist.sort(key=lambda x : x[1], reverse=True )
    cornerlist = filterPoints(cornerlist,count=tiling[0]*tiling[1]*count,separation=separation)
    return cornerlist

    
def getMotion(img1, img2, numpts=5, debug=0):
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
    convolutionmode = "full"
    ixs = signal.convolve2d(ixix, sumf, mode=convolutionmode, boundary='symm')  
    ixys = signal.convolve2d(ixiy, sumf, mode=convolutionmode, boundary='symm')  
    iys = signal.convolve2d(iyiy, sumf, mode=convolutionmode, boundary='symm')  
    ixits = signal.convolve2d(ixit, sumf, mode=convolutionmode, boundary='symm')  
    iyits = signal.convolve2d(iyit, sumf, mode=convolutionmode, boundary='symm')  

    # Step 3.  Feature points (Harris detector)
    cornerlist = getHarris(img2,debug=debug)

    # Step 4.  Motion for each corner
    motionlist = []
    (Nx,Ny) = img2.shape

    # Iterate over corners
    for ((i,j),s) in cornerlist:
        # We do not track corners close to the edges
        if i < 2 or i > Nx-3: continue
        if j < 2 or j > Ny-3: continue
        if debug > 2:
            print( (i,j), s, ". ixs shape ", ixs.shape, ". Image shape ", img2.shape )
        a = ixs[i][j]
        c = iys[i][j]
        b = ixys[i][j]
        Gmatrix = np.array([[a, b], [b, c]])
        bvector = np.array([[ixits[i][j]], [iyits[i][j]]])
        # -G(x) - lb(x, t)
        u = - np.linalg.inv(Gmatrix) @ bvector
        motionlist.extend( [ ((i,j),s,u) ] )
        if debug > 2: print ( "Motion ", u.flatten() )
    return motionlist

def drawMotion(img,pt,u,scale=2):
    """Draw a motion vector u from a given point pt."""
    (x,y) = pt
    u = u.flatten()
    (x1,y1) = (x+int(np.round(scale*u[0])),y+int(np.round(scale*u[1])))
    return cv.arrowedLine(img, pt, (x1,y1), (255,0,0), 1)

def drawMotionList(img,motionlist,scale=5):
    """Draw motion vectors as returned from getMotion()."""
    image = img
    for (pt,s,u) in motionlist: 
        (y,x) = pt
        pos = (x,y)
        cv.circle(img,pos,1,(0,0,255),-1)
        image = drawMotion(image,pos,u,scale=scale)
    return image
