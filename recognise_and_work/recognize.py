import cv2
import numpy as np

from helpers import functions
from helpers import help


def seperate(image, bounds):
    N = len(bounds)
    plate_number = ""

    for i in range(N - 1):
        character_image = image[:, bounds[i]:bounds[i + 1]]
        plate_number = plate_number + recognise_characters(character_image)

    return make_plate_number(plate_number)


def make_plate_number(plate_number):
    indexes = functions.dashes(plate_number, "-")
    N = len(indexes)
    M = len(plate_number)

    if N:
        if (N != 2) or (indexes[0] == 0) or (indexes[N - 1] == M - 1):
            return None

    return plate_number


def recognise_characters(image_part):
    width = image_part.shape[1]
    score = np.zeros(28)
    curr = []

    if width <= 95:
        for i in range(17):
            check = cv2.imread("data/SameSizeLetters/" + str(i + 1) + ".bmp", cv2.IMREAD_GRAYSCALE)

            # bitwise_not : The function calculates per-element bit-wise inversion of the input array:
            # dst[i] = ~src[i]

            # bitwise_xor : Calculates the per-element bit-wise “exclusive or” operation on two arrays or an array and a scalar.
            # Two arrays when src1 and src2 have the same size:
            # dst[i] = src1[i] xor src2[i] if mask[i] != 0

            for i in range(min([check.shape[1] - width - 1, 2])):
                find = check[:, i:i + width]
                curr.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(find, image_part))) / (check.shape[0] * width * 255))

            score[i] = max(curr)
            curr = []

        for i in range(10):
            check = cv2.imread("data/SameSizeNumbers/" + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)

            # bitwise_not : The function calculates per-element bit-wise inversion of the input array:
            # dst[i] = ~src[i]

            # bitwise_xor : Calculates the per-element bit-wise “exclusive or” operation on two arrays or an array and a scalar.
            # Two arrays when src1 and src2 have the same size:
            # dst[i] = src1[i] xor src2[i] if mask[i] != 0

            for i in range(min([check.shape[1] - width - 1, 2])):
                find = check[:, i:i + width]
                curr.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(find, image_part))) / (check.shape[0] * width * 255))

            score[17 + i] = max(curr)
            curr = []

        h = image_part.shape[0] + 50
        w = image_part.shape[1]
        dash = np.zeros((h, w), np.uint8)
        dash_one = int((h / 2) - 5)
        dash_end = dash_one + 10
        k = 0
        dash_two_end = k + 15
        dash[dash_one:dash_end, k:dash_two_end] = 255 * np.ones([10, min((15, w))])

        for i in range(dash.shape[0] - image_part.shape[0] - 1):
            find = dash[i:i + image_part.shape[0], :]
            # bitwise_not : The function calculates per-element bit-wise inversion of the input array:
            # dst[i] = ~src[i]

            # bitwise_xor : Calculates the per-element bit-wise “exclusive or” operation on two arrays or an array and a scalar.
            # Two arrays when src1 and src2 have the same size:
            # dst[i] = src1[i] xor src2[i] if mask[i] != 0

            curr.append(
                np.sum(cv2.bitwise_not(cv2.bitwise_xor(find, image_part))) / (image_part.shape[0] * width * 255))

        score[27] = max(curr)

        return help.values[str(np.argmax(score))]

    else:
        return ""


def recognition_segment(plate):
    plate = functions.remove(plate, (4, 7))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    plate = cv2.dilate(plate, kernel, iterations=1)

    hor = np.sum(plate, axis=1)
    ver = help.boundary_2(hor, 16800)

    new_plate = plate[ver[0] + 1:ver[1]][:]

    new_plate = cv2.resize(new_plate, (int(new_plate.shape[1] * (85 / new_plate.shape[0])), 85),
                           interpolation=cv2.INTER_LINEAR)

    sum = np.sum(new_plate, axis=0)
    hor_b = help.boundary_1(sum)

    if len(hor_b) < 6:
        return

    new_plate = make_boxes(new_plate, ver, hor_b, plate)
    plate_number = seperate(new_plate, hor_b)

    return plate_number


def make_boxes(new_plate, ver, hor_b, plate):
    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

    for b in ver:
        plate = cv2.line(plate, (0, new_plate.shape[0] - b), (new_plate.shape[1], new_plate.shape[0] - b), (0, 0, 255),2)
    for b in hor_b:
        new_plate = cv2.line(new_plate, (b, 0), (b, new_plate.shape[0]), (0, 0, 255), 2)

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_BGR2GRAY)

    return new_plate
