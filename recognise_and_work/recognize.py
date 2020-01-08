import cv2
import numpy as np
from helpers import functions


def form_plate_number(image, bounds):
    N = len(bounds)
    plate_number = ""

    for i in range(N - 1):
        plate_number += recognize_character(image[:, bounds[i]:bounds[i + 1]])

    return plate_number


def recognize_character(image):
    width = image.shape[1]
    score = np.zeros(28)
    list_of_score = []

    for i in range(17):
        character = cv2.imread("SameSizeLetters/" + str(i + 1) + ".bmp", cv2.IMREAD_GRAYSCALE)
        coef = character.shape[0] * width * 255

        for start in range(character.shape[1] - width - 1):
            temp = np.sum(cv2.bitwise_not(cv2.bitwise_xor(character[:, start:start + width], image)))

            if temp is not None:
                list_of_score.append(temp / coef)

        if len(list_of_score) == 0:
            continue

        score[i] = max(list_of_score)
        list_of_score = []

    for i in range(10):
        character = cv2.imread("SameSizeNumbers/" + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
        coef = character.shape[0] * character.shape[1] * 255

        for start in range(character.shape[1] - width - 1):
            temp = np.sum(cv2.bitwise_not(cv2.bitwise_xor(character[:, start:start + width], image)))

            if temp is not None:
                list_of_score.append(temp / coef)

        if len(list_of_score) == 0:
            continue

        score[17 + i] = max(list_of_score)
        list_of_score = []

    character = blank_characters(image.shape, 10, 20)
    coef = character.shape[0] * character.shape[1] * 255

    for start in range(character.shape[1] - width - 1):
        temp = np.sum(cv2.bitwise_not(cv2.bitwise_xor(character[:, start:start + width], image)))

        if temp is not None:
            list_of_score.append(temp / coef)

    if len(list_of_score) != 0:
        score[27] = max(list_of_score)

    # print(characters[str(np.argmax(score))])
    return functions.valuees[str(np.argmax(score))]


def blank_characters(img_shape, bar_thickness, bar_width):
    ch_height = img_shape[0]
    ch_width = img_shape[1] + 20

    bar = np.zeros((ch_height, ch_width), np.uint8)
    bart_init = int(ch_height / 2) - int(bar_thickness / 2)

    bart_end = bart_init + bar_thickness
    barw_init = int(ch_width / 2) - int(bar_width / 2)
    barw_end = barw_init + bar_width

    bar[bart_init:bart_end, barw_init:barw_end] = 255 * np.ones([bar_thickness, bar_width])
    return bar


def recognition(plate):
    plate = functions.remove_borders(plate, (13, 13))

    hori = np.sum(plate, axis=1)
    verti_end = functions.search_boundary_1(hori, 1000)

    new_plate = plate[verti_end[0] + 1:verti_end[1]][:]
    new_plate = cv2.resize(new_plate, (int(new_plate.shape[1] * (85 / new_plate.shape[0])), 85), interpolation=cv2.INTER_LINEAR)

    ver_sum = np.sum(new_plate, axis=0)
    hori_end = functions.search_boundary_2(ver_sum)

    w = new_plate.shape[1]
    h = new_plate.shape[0]

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

    for i in hori_end:
        new_plate = cv2.line(new_plate, (i, 0), (i, h), (0, 255, 0), 1)

    for i in verti_end:
        plate = cv2.line(plate, (0, h - i), (w, h - i), (160, 0, 0), 1)

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_BGR2GRAY)

    cv2.imshow("plate", new_plate)
    return form_plate_number(new_plate, hori_end)
