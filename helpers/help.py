import numpy as np
import cv2

from helpers import functions


values = {
        "0": "B",
        "23": "6",
        "19": "2",
        "17": "0",
        "3": "G",
        "25": "8",
        "1": "D",
        "6": "K",
        "8": "M",
        "5": "J",
        "11": "R",
        "26": "9",
        "9": "N",
        "24": "7",
        "16": "Z",
        "27": "-",
        "18": "1",
        "10": "P",
        "4": "H",
        "12": "S",
        "21": "4",
        "13": "T",
        "14": "V",
        "15": "X",
        "20": "3",
        "22": "5",
        "7": "L",
        "2": "F",
}


def boundary_2(hp, T):
    # reference youtube video
    N = len(hp)
    i = 0

    while ~((hp[i] <= T) & (hp[i + 1] > T)) & (i < int(N / 2)):
        i += 1
    inf_bound = 0 if i == int(N / 2) else i

    i = N - 1
    while ~((hp[i - 1] > T) & (hp[i] <= T)) & (i > int(N / 2)):
        i -= 1
    sup_bound = i

    return [inf_bound, sup_bound]


def boundary_1(vertical_projection):
    N = len(vertical_projection)
    bool_bounds = (vertical_projection >= 2000)
    start_ind = 0
    end_ind = 1
    bounds = []

    for b in range(N - 1):
        if bool_bounds[end_ind] & ~bool_bounds[start_ind]:  # upwards transition
            bounds.append(end_ind)
            last_bound = bounds[len(bounds) - 1]

            if end_ind - last_bound >= 99:
                bounds.append(last_bound + 98)

        start_ind += 1
        end_ind += 1

    return bound_1_help(bounds, end_ind)


def bound_1_help(bounds, end_ind):
    if bounds:
        last_bound = bounds[len(bounds) - 1]
        if end_ind - last_bound < 99:
            bounds.append(end_ind)
        else:
            bounds.append(last_bound + 98)
    return bounds


def warp_perspective(image, coords):
    rect = functions.rectagnle(coords)
    (a, b, c, d) = rect

    w1 = np.sqrt(((c[0] - d[0]) ** 2) + ((c[1] - d[1]) ** 2))
    w2 = np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))

    h1 = np.sqrt(((b[0] - c[0]) ** 2) + ((b[1] - c[1]) ** 2))
    h2 = np.sqrt(((a[0] - d[0]) ** 2) + ((a[1] - d[1]) ** 2))

    return warp_help(max(int(w1), int(w2)), max(int(h1), int(h2)), rect, image)


def warp_help(maxW, maxH, rect, image):
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxW, maxH))

    return warped