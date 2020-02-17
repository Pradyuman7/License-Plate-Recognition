import cv2
import numpy as np


def warp_perspective_image(image, coords):
    rect = make_rect(coords)
    (a, b, c, d) = rect

    # width of image will be max distance between bottom_left and bottom_right
    wA = np.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2))
    hA = np.sqrt(((b[0] - c[0]) ** 2) + ((b[1] - c[1]) ** 2))

    # height of image will be max distance between top_right and bottom_right
    wB = np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))
    hB = np.sqrt(((a[0] - d[0]) ** 2) + ((a[1] - d[1]) ** 2))

    maxHeight = max(int(hA), int(hB))
    maxWidth = max(int(wA), int(wB))

    # get top down view of the final points in same order as rect
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # warpPerspective : Applies a perspective transformation to an image.
    # params : src – input image, dst – output image that has the size dsize and the same type as src,
    # M – 3 x 3 transformation matrix.
    # dst(x,y) = ((m11.x + m12.y + m13)/(m31.x + m32.y + m33), (m21.x + m22.y + m23)/(m31.x + m32.y + m33))

    # getPerspective : Calculates a perspective transform from four pairs of the corresponding points.
    # src, dst
    # | ti.xi' |                    | xi |
    # | ti.yi' |   =   map_matrix . | yi |
    # | ti     |                    |  1 |
    # dst[i] = (xi', yi'), src[i] = (xi,yi) i = 0,1,2,3

    # return the warped image
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (maxWidth, maxHeight))


def make_rect(coords):
    # reference wikipedia, opencv tutorial, youtube
    # makes the rectangle from top_left to bottom_left order of points
    rect = np.zeros((4, 2), dtype="float32")

    s = coords.sum(axis=1)

    # top_left will have smallest sum, bottom_right will have largest
    rect[0] = coords[np.argmin(s)]
    rect[2] = coords[np.argmax(s)]

    # in difference top_right smallest diff, bottom_left largest diff
    diff = np.diff(coords, axis=1)

    rect[1] = coords[np.argmin(diff)]
    rect[3] = coords[np.argmax(diff)]

    return rect


def localise_plates(image, contours):
    # localise straight plates
    contour = []

    if len(contours) != 0:
        for c in contours:
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area = cv2.contourArea(box)

            if area > 2500:
                contour.append(box)

    # minAreaRect : Finds a rotated rectangle of the minimum area enclosing the input 2D point set.
    # input points parameter
    # The function calculates and returns the minimum-area bounding rectangle (possibly rotated)
    # for a specified point set

    # boxPoints : boxes the points

    # contourArea : Calculates a contour area of the box given

    # if len(contour) == 0:
    #     return image
    # else:
    #     return warp_perspective_image(image, contour[0])

    all_img = []
    for c in contour:
        all_img.append(warp_perspective_image(image, c))

    return all_img
