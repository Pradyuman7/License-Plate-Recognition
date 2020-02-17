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
    # reference youtube video and wikipedia
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
    # reference youtube video
    try:
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])
        hist = np.convolve(np.ones(8) / 8, hist)[:256]

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

        gi = np.arange(tmin, t[0])
        gs = np.arange(t[0], tmax)

        # if np.sum(hist[tmin:t[0]]) and np.sum(hist[t[0]:tmax]):
        m1 = np.average(gi, weights=hist[tmin:t[0]])
        m2 = np.average(gs, weights=hist[t[0]:tmax])

        t.append(int(np.average([m1, m2])))
        i = 1

        while np.abs(t[i - 1] - t[i]) > 0.6:
            gi = np.arange(tmin, t[i])
            gs = np.arange(t[i], tmax)
            m1 = np.average(gi, weights=hist[tmin:t[i]])
            m2 = np.average(gs, weights=hist[t[i]:tmax])
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
    i = N - 1

    while ~((hp[i - 1] > T) & (hp[i] <= T)) & (i < N - 1):
        i += 1

    higher = i
    return [lower, higher]


def search_boundary_2(proj):
    N = len(proj)
    start = 0
    end = 1
    bounds = []
    bool_bounds = (proj >= 2000)

    for b in range(N - 1):
        if bool_bounds[end] & ~bool_bounds[start]:
            bounds.append(end)
            last = bounds[len(bounds) - 1]
            if end - last >= 99:
                bounds.append(last + 98)

        start += 1
        end += 1

    if bounds:
        last = bounds[len(bounds) - 1]
        if end - last < 99:
            bounds.append(end)
        else:
            bounds.append(last + 98)
    return bounds
