import numpy as np
import cv2


def isodata_threshold(img):
    # reference youtube video and wikipedia
    hist, bins = np.histogram(img.ravel(), 256, [0, 256])
    h = 1 / 8 * np.ones(8)
    hist = np.convolve(h, hist)[:256]

    N = len(hist)
    T = 100

    s = 0
    while ~((hist[s] <= T) & (hist[s + 1] > T)) & (s < N - 2):
        s += 1
    tmin = s

    s = N - 1
    while ~((hist[s - 1] > T) & (hist[s] <= T)) & (s > 1):
        s -= 1
    tmax = s

    t0 = int(np.average((tmin, tmax)))
    t = [t0]

    return iso_helper(tmin, t, tmax, hist)


def iso_helper(tmin, t, tmax, hist):
    constant = 0.5

    ginf = np.arange(tmin, t[0])
    gsup = np.arange(t[0], tmax)

    if np.sum(hist[tmin:t[0]]) and np.sum(hist[t[0]:tmax]):
        m1 = np.average(ginf, weights=hist[tmin:t[0]])
        m2 = np.average(gsup, weights=hist[t[0]:tmax])

    t.append(int(np.average([m1, m2])))
    i = 1

    while np.abs(t[i - 1] - t[i]) > constant:
        ginf = np.arange(tmin, t[i])
        gsup = np.arange(t[i], tmax)
        m1 = np.average(ginf, weights=hist[tmin:t[i]])
        m2 = np.average(gsup, weights=hist[t[i]:tmax])
        t.append(int(np.average([m1, m2])))
        i += 1

    return t[i]


def remove(plate_image, constant):
    height = plate_image.shape[0]
    width = plate_image.shape[1]

    plate_image[0:constant[0], :] = 0
    plate_image[height - constant[0]:height, :] = 0
    plate_image[:, 0:constant[1]] = 0
    plate_image[:, width - constant[1]:width] = 0

    return plate_image


def dashes(string, search):
    l1 = []
    length = len(string)
    index = 0

    while index < length:
        i = string.find(search, index)

        if i == -1:
            return l1

        l1.append(i)
        index = i + 1

    return l1


def check(box):
    rect = rectagnle(box)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth and maxHeight:
        aspect_ratio = maxHeight / maxWidth
    else:
        aspect_ratio = 1
    area = cv2.contourArea(box)

    return (maxWidth > 100) and (aspect_ratio < 0.3) and (area > 2600)


def rectagnle(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
