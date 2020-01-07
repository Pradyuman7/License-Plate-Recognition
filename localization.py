import cv2
import numpy as np


def check_number_platee(box):
    width = box[3][0] - box[0][0]
    # height = box[0][0] - box[0][1]
    # aspect_ratio = width / height
    return width > 100


def rectangle(coords):
    rect = np.zeros((4, 2), dtype="float32")

    s = coords.sum(axis=1)
    rect[0] = coords[np.argmin(s)]
    rect[2] = coords[np.argmax(s)]

    diff = np.diff(coords, axis=1)
    rect[1] = coords[np.argmin(diff)]
    rect[3] = coords[np.argmax(diff)]

    return rect


def four_point_transform(image, coords):
    rect = rectangle(coords)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (maxWidth, maxHeight))
    # cv2.imshow('Localized', image)


def find_plate_in_frame(image, contours):
    contour = []
    i = 0
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)

        if area > 2500:  # area < 2800
            contour.append(box)

    # Finds the minimal rectangle that bounds the contour
    if len(contour) == 0:
        return
    return four_point_transform(image, contour[0])
