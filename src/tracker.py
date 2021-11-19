"""Demo of a feature tracker.

NB: Untested, unfinished!"""

import cv2 as cv

def getGradient(img):
    Ix = cv.Sobel(img1, ddepth=cv.CV_32F, dx=1, dy=0, ksize=5)
    Iy = cv.Sobel(img1, ddepth=cv.CV_32F, dx=0, dy=1, ksize=5)
    return (Ix,Iy)


def getMotion(Ix, Iy, It, x):
    pass

def optical_flow(img1, img2, pa):
    """
    Calculate motion/optical flow between two frames

        Parameters:
            img1 (numpy array): previous frame
            img2 (numpy array): current frame
            pa (__): ??
    """

    # Step 1.  Feature points (Harris detector)
    cx_answer = cv.cornerHarris(img2, 5, 5, 0.06)

    # Step 2.  Spacial and temporal derivatives
    (Ix,Iy) = getGradient(img2)
    It = img2 - img1

    # Step 3.  The G matrix
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

    # Step 4.  Corners
    # -G(x) - lb(x, t)
    # Find corners
    min_cx = np.min(cx_answer)
    max_cx = np.max(cx_answer)
    T = min_cx + (max_cx - min_cx) * 50 / 100
    T = T * 3
    
    # Iterate over corners
    # TODO: This is only one pixel per corner, filter around corners?
    for i in range(cx_answer.shape[0]):
        for j in range(cx_answer.shape[1]):
            if cx_answer[i, j] > T:
                a = ixs[i][j]
                c = iys[i][j]
                b = ixys[i][j]
                Gx = np.array([[a, b], [b, c]])
                bx = np.array([[ixits[i][j]],
                               [iyits[i][j]]])
                u = - np.linalg.inv(Gx) @ bx
