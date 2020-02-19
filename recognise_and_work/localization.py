import cv2
import numpy as np

from helpers import functions


# localise straight plates

def localise_plates(image, contours):
    all_con = []
    i = 0

    for c in contours:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        if functions.check(box):
            all_con.append(box)

        i += 1

    return do_warp(image, all_con)


def do_warp(image, all_con):
    plate = []

    # if len(contour) == 0:
    #     return image
    # else:
    #     return warp_perspective_image(image, contour[0])

    for c in all_con:
        plate.append(functions.warp_perspective(image, c))

    return plate


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

