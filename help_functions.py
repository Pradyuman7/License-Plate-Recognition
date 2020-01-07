import cv2
import numpy as np


def find_contour(contours):
    contour = []

    for i, c in enumerate(contours):
        peri = cv2.arcLength(c, True)

        if 300 < peri < 600:
            contour.append(i)

    return contour


def isodata_threshold(img):
    try:
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])
        hist = hist[:120]
        t = [60]
        i = 0
        epsilon = 0.6

        ginf = np.arange(0, t[0])
        gsup = np.arange(t[0], 120)

        m1 = np.average(ginf, weights=hist[:t[i]])
        m2 = np.average(gsup, weights=hist[t[i]:])

        t.append(int(np.average([m1, m2])))
        i = 1

        while np.abs(t[i - 1] - t[i]) > epsilon:
            ginf = np.arange(0, t[i])
            gsup = np.arange(t[i], 120)
            m1 = np.average(ginf, weights=hist[:t[i]])
            m2 = np.average(gsup, weights=hist[t[i]:])
            t.append(int(np.average([m1, m2])))
            i += 1

        # print("threshold is : ", t[i])
        return t[i]
    except ZeroDivisionError:
        return 0
