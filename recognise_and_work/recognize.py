import cv2
import numpy as np
from helpers import functions


def form_plate_number(image, bounds):
    N = len(bounds)
    plate_number = ""

    for i in range(N - 1):
        plate_number += recognize_character(image[:, bounds[i]:bounds[i + 1]])

    if plate_number.startswith('H') and plate_number.endswith('H'):
        plate_number = plate_number[1:-1]
    #     return plate_number.replace('H', '-')
    elif plate_number.startswith('H'):
        plate_number = plate_number[1:]
    #     return plate_number.replace('H', '-')
    elif plate_number.endswith('H'):
        plate_number = plate_number[:-1]
    #     return plate_number.replace('H', '-')
    # else:
    #     return plate_number.replace('H', '-')

    return plate_number


# def recognise_sift(image):
#
#     for i in range(1, 18):
#         character = cv2.imread("data/SameSizeLetters/" + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
#
#         orb = cv2.ORB()
#         kp1, des1 = orb.detectAndCompute(character, None)
#         kp2, des2 = orb.detectAndCompute(character, None)


def recognize_character_pixel(image):
    score = np.zeros(28)
    list_of_score = []

    for number in range(1, 18):
        character = cv2.imread("data/SameSizeLetters/" + str(number) + ".bmp", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (character.shape[0], character.shape[1]))
        count = 0

        for i in range(0, character.shape[0]):
            for j in range(0, character.shape[1]):
                for k in range(0, image.shape[0]):
                    for l in range(0, image.shape[1]):
                        if character[i][j] == image[k][l]:
                            count += 1

        list_of_score.append(count)
        score[number - 1] = max(list_of_score)

    for number in range(10):
        character = cv2.imread("data/SameSizeNumbers/" + str(number) + ".bmp", cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, character.shape[0], character.shape[1])
        count = 0

        for i in range(0, character.shape[0]):
            for j in range(0, character.shape[1]):
                for k in range(0, image.shape[0]):
                    for l in range(0, image.shape[1]):
                        if character[i][j] == image[k][l]:
                            count += 1

        list_of_score.append(count)
        score[number + 17] = max(list_of_score)

    return functions.valuees[str(np.argmax(score))]


def recognize_template_matching(image):
    scores = np.zeros(28)
    list_of_score = []

    # cv2.imshow("character we found", image)
    # cv2.waitKey()

    for i in range(1, 18):
        character = cv2.imread("data/SameSizeLetters/" + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(image, character, cv2.TM_CCOEFF)

        if result is not None:
            (_, score, _, _) = cv2.minMaxLoc(result)
            list_of_score.append(score)

        if len(list_of_score) != 0:
            scores[i - 1] = max(list_of_score)
            list_of_score = []

    for i in range(10):
        character = cv2.imread("data/SameSizeNumbers/" + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(character, image, cv2.TM_CCOEFF)

        if result is not None:
            (_, score, _, _) = cv2.minMaxLoc(result)
            list_of_score.append(score)

        if len(list_of_score) != 0:
            scores[i - 1] = max(list_of_score)
            list_of_score = []

    # print("scores found", scores)
    # print(str(np.argmax(scores)))
    return functions.valuees[str(np.argmax(scores))]


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

    he = image.shape[0] + 40
    wi = image.shape[1]

    char = np.zeros((he, wi), np.uint8)
    start = int(he / 2) - 5
    end = start + 12
    from_ = 0
    end_ = from_ + 14

    char[start:end, from_:end_] = 255 * np.ones([12, min(14, wi)])

    test_character_height = char.shape[0]
    coef = image.shape[0] * width * 255

    for start in range(test_character_height - image.shape[0] - 1):
        temp = np.sum(cv2.bitwise_not(cv2.bitwise_xor(char[start:start + image.shape[0], :], image)))

        if temp is not None:
            list_of_score.append(temp / coef)

    if len(list_of_score) != 0:
        score[27] = max(list_of_score)
        list_of_score.clear()

    return functions.valuees[str(np.argmax(score))]


def recognition(plate):
    plate = functions.clear_border(plate, (13, 13))

    hori = np.sum(plate, axis=1)
    verti_end = functions.search_boundary_1(hori, 1000)

    new_plate = plate[verti_end[0] + 1:verti_end[1]][:]
    new_plate = cv2.resize(new_plate, (int(new_plate.shape[1] * (85 / new_plate.shape[0])), 85),
                           interpolation=cv2.INTER_LINEAR)

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
