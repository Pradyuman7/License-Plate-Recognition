import cv2
import numpy as np


def four_point_transform(image, coords):
    rect = make_rect(coords)
    (a, b, c, d) = rect

    wA = np.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2))
    hA = np.sqrt(((b[0] - c[0]) ** 2) + ((b[1] - c[1]) ** 2))

    wB = np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))
    hB = np.sqrt(((a[0] - d[0]) ** 2) + ((a[1] - d[1]) ** 2))

    maxHeight = max(int(hA), int(hB))
    maxWidth = max(int(wA), int(wB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (maxWidth, maxHeight))


def make_rect(coords):
    rect = np.zeros((4, 2), dtype="float32")

    s = coords.sum(axis=1)
    diff = np.diff(coords, axis=1)

    rect[0] = coords[np.argmin(s)]
    rect[1] = coords[np.argmin(diff)]
    rect[2] = coords[np.argmax(s)]
    rect[3] = coords[np.argmax(diff)]

    return rect


def find_plate_in_frame(image, contours):
    contour = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = cv2.contourArea(box)

        if area > 2500:
            contour.append(box)

    # Finds the minimal rectangle that bounds the contour
    if len(contour) == 0:
        return image
    else:
        return four_point_transform(image, contour[0])
