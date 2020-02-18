import numpy as np
import cv2


def isodata_threshold(img):
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

    epsilon = 0.5

    ginf = np.arange(tmin, t[0])
    gsup = np.arange(t[0], tmax)

    if np.sum(hist[tmin:t[0]]) and np.sum(hist[t[0]:tmax]):
        m1 = np.average(ginf, weights=hist[tmin:t[0]])
        m2 = np.average(gsup, weights=hist[t[0]:tmax])

    t.append(int(np.average([m1, m2])))
    i = 1

    while np.abs(t[i - 1] - t[i]) > epsilon:
        ginf = np.arange(tmin, t[i])
        gsup = np.arange(t[i], tmax)
        m1 = np.average(ginf, weights=hist[tmin:t[i]])
        m2 = np.average(gsup, weights=hist[t[i]:tmax])
        t.append(int(np.average([m1, m2])))
        i += 1

    return t[i]


def create_bar_character(img_shape, bar_thickness, bar_width):
    ch_height = img_shape[0] + 50
    ch_width = img_shape[1]
    bar = np.zeros((ch_height, ch_width), np.uint8)
    bart_init = int(ch_height / 2) - int(bar_thickness / 2)
    bart_end = bart_init + bar_thickness
    barw_init = 0
    barw_end = barw_init + bar_width
    bar[bart_init:bart_end, barw_init:barw_end] = 255 * np.ones([bar_thickness, min((bar_width, ch_width))])
    return bar


def clean_borders(plate_image, epsilon):
    height = plate_image.shape[0]
    width = plate_image.shape[1]

    plate_image[0:epsilon[0], :] = 0
    plate_image[height - epsilon[0]:height, :] = 0
    plate_image[:, 0:epsilon[1]] = 0
    plate_image[:, width - epsilon[1]:width] = 0

    return plate_image


def verify_plate(box):
    rect = order_points(box)
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


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


lookup_table = {
    "0": "B", "1": "D", "2": "F", "3": "G", "4": "H",
    "5": "J", "6": "K", "7": "L", "8": "M", "9": "N",
    "10": "P", "11": "R", "12": "S", "13": "T", "14": "V",
    "15": "X", "16": "Z", "17": "0", "18": "1", "19": "2",
    "20": "3", "21": "4", "22": "5", "23": "6", "24": "7",
    "25": "8", "26": "9", "27": "-"
}


def find_vertical_bounds(hp, T):
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


def find_horizontal_bounds(vertical_projection):
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

    if bounds:
        last_bound = bounds[len(bounds) - 1]
        if end_ind - last_bound < 99:
            bounds.append(end_ind)
        else:
            bounds.append(last_bound + 98)
    return bounds


def find_all_indexes(input_str, search_str):
    l1 = []
    length = len(input_str)
    index = 0
    while index < length:
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def four_point_transform(image, pts):
    rect = order_points(pts)
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

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped