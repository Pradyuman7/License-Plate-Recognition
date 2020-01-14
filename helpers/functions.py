import numpy as np


valuees = {
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


def clear_border(plate, ep):
    # removes borders from gray scale image
    # ep is the max width of part that should be removed around the image
    h = plate.shape[0]
    w = plate.shape[1]

    plate[0:ep[0], :] = 0
    plate[h - ep[0]:h, :] = 0
    plate[:, 0:ep[1]] = 0
    plate[:, w - ep[1]:w] = 0

    return plate


def isodata_threshold(img):
    # isodata threshold the image (gray scale)
    # isodata_threshold :
    try:
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])
        hist = hist[:120]
        t = [60]
        i = 0
        # ep = 1

        gi = np.arange(0, t[0])
        gs = np.arange(t[0], 120)

        m1 = np.average(gi, weights=hist[:t[i]])
        m2 = np.average(gs, weights=hist[t[i]:])

        t.append(int(np.average([m1, m2])))
        i = 1

        while np.abs(t[i - 1] - t[i]) > 1:
            gi = np.arange(0, t[i])
            gs = np.arange(t[i], 120)
            m1 = np.average(gi, weights=hist[:t[i]])
            m2 = np.average(gs, weights=hist[t[i]:])
            t.append(int(np.average([m1, m2])))
            i += 1

        return t[i]
    except ZeroDivisionError:
        return 0


def search_boundary_1(hp, T):
    # t is the threshold for finding difference betweeen blank and characters
    # vertical boundary
    N = len(hp)
    i = 0

    while ~((hp[i] <= T) & (hp[i + 1] > T)) & (i < int(N / 2)):
        i += 1

    lower = 0 if i == int(N / 2) else i
    i = int(N / 2)

    while ~((hp[i - 1] > T) & (hp[i] <= T)) & (i < N - 1):
        i += 1

    higher = i
    return [lower, higher]


def rectangle(coords):
    # makes the rectangle from top_left to bottom_left order of points
    rect = np.zeros((4, 2), dtype="float32")

    s = coords.sum(axis=1)

    # top_left will have smallest sum, bottom_right will have largest

    # in difference top_right smallest diff, bottom_left largest diff
    diff = np.diff(coords, axis=1)

    rect[0] = coords[np.argmin(s)]
    rect[1] = coords[np.argmin(diff)]
    rect[2] = coords[np.argmax(s)]
    rect[3] = coords[np.argmax(diff)]

    return rect


def search_boundary_2(proj):
    N = len(proj)
    start = 0
    end = 1
    bounds = []
    bool_bounds = (proj >= 255)

    for b in range(N - 1):
        if bool_bounds[end] & ~bool_bounds[start]:
            bounds.append(end)

        start += 1
        end += 1

    bounds.append(end - 20)
    return bounds
