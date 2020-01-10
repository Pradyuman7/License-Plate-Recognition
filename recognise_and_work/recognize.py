import cv2
import numpy as np
from helpers import functions


#  Otsu‘s method which calculates an “optimal” threshold by maximizing the variance between two classes of pixels
# after denosising

def form_plate_number(image, bounds):
    N = len(bounds)
    plate_number = ""

    for i in range(N - 1):
        plate_number += recognize_character(image[:, bounds[i]:bounds[i + 1]])

    if plate_number.startswith('H') and plate_number.endswith('H'):
        plate_number = plate_number[1:-1]
        return plate_number.replace('H', '-')
    elif plate_number.startswith('H'):
        plate_number = plate_number[1:]
        return plate_number.replace('H', '-')
    elif plate_number.endswith('H'):
        plate_number = plate_number[:-1]
        return plate_number.replace('H', '-')
    else:
        return plate_number.replace('H', '-')


def recognize_character(image):
    width = image.shape[1]
    score = np.zeros(28)
    list_of_score = []

    # bitwise_not : The function calculates per-element bit-wise inversion of the input array:
    # dst[i] = ~src[i]

    # bitwise_xor : Calculates the per-element bit-wise “exclusive or” operation on two arrays or an array and a scalar.
    # Two arrays when src1 and src2 have the same size:
    # dst[i] = src1[i] xor src2[i] if mask[i] != 0

    for i in range(1, 18):
        character = cv2.imread("data/SameSizeLetters/" + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
        coef = character.shape[0] * width * 255

        for j in range(character.shape[1] - width - 1):
            temp = np.sum(cv2.bitwise_not(cv2.bitwise_xor(character[:, j:j + width], image)))

            if temp is not None:
                list_of_score.append(temp / coef)

        if len(list_of_score) != 0:
            score[i - 1] = max(list_of_score)
            list_of_score = []

    for i in range(10):
        character = cv2.imread("data/SameSizeNumbers/" + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
        coef = character.shape[0] * width * 255

        for j in range(character.shape[1] - width - 1):
            temp = np.sum(cv2.bitwise_not(cv2.bitwise_xor(character[:, j:j + width], image)))

            if temp is not None:
                list_of_score.append(temp / coef)

        if len(list_of_score) != 0:
            score[17 + i] = max(list_of_score)
            list_of_score = []

    return functions.valuees[str(np.argmax(score))]


def recognition(plate):
    plate = functions.clear_border(plate, (13, 13))

    hori = np.sum(plate, axis=1)
    verti_end = functions.search_boundary_1(hori, 1000)

    new_plate = plate[verti_end[0] + 1:verti_end[1]][:]
    new_plate = cv2.resize(new_plate, (int(new_plate.shape[1] * (85 / new_plate.shape[0])), 85), interpolation=cv2.INTER_LINEAR)

    ver_sum = np.sum(new_plate, axis=0)
    hori_end = functions.search_boundary_2(ver_sum)

    w = new_plate.shape[1]
    h = new_plate.shape[0]

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

    # draw columns
    for i in hori_end:
        new_plate = cv2.line(new_plate, (i, 0), (i, h), (0, 255, 0), 2)

    for i in verti_end:
        plate = cv2.line(plate, (0, h - i), (w, h - i), (0, 255, 0), 2)

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_BGR2GRAY)

    cv2.imshow("plate", new_plate)
    return form_plate_number(new_plate, hori_end)
