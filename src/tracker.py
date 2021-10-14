"""Optical flow test-code.
NB: Untested, unfinished!"""

def optical_flow(I1g, I2g, img1, img2, pa):
    cx_answer = cv.cornerHarris(I1g, 5, 5, 0.06)
    cx_answer_2 = cv.cornerHarris(I2g, 5, 5, 0.06)
    gX = cv.Sobel(I1g, ddepth=cv.CV_32F, dx=1, dy=0, ksize=5)
    gY = cv.Sobel(I1g, ddepth=cv.CV_32F, dx=0, dy=1, ksize=5)
    it = I2g - I1g

    ixix = gX * gX
    ixiy = gX * gY
    iyiy = gY * gY
    ixit = gX * it
    iyit = gY * it

    sumf = np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])

    ixs = signal.convolve2d(ixix, sumf, mode='full', boundary='symm')  # / 25
    ixys = signal.convolve2d(ixiy, sumf, mode='full', boundary='symm')  # / 25
    iys = signal.convolve2d(iyiy, sumf, mode='full', boundary='symm')  # / 25
    ixits = signal.convolve2d(ixit, sumf, mode='full', boundary='symm')  # / 25
    iyits = signal.convolve2d(iyit, sumf, mode='full', boundary='symm')  # / 25

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
                u = - np.linalg.inv(Gx) * bx
