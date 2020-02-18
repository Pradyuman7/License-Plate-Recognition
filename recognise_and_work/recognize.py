import cv2
import numpy as np

from helpers import functions


def divide_characters(image, bounds):
    N = len(bounds)
    plate_number = ""

    for i in range(N - 1):
        character_image = image[:, bounds[i]:bounds[i + 1]]
        plate_number = plate_number + match_characters(character_image)

    indexes = functions.find_all_indexes(plate_number, "-")
    N = len(indexes)
    M = len(plate_number)

    if N:
        if (N != 2) or (indexes[0] == 0) or (indexes[N - 1] == M - 1):
            return None

    return plate_number


def match_characters(character_image):
    character_image_width = character_image.shape[1]
    score = np.zeros(28)
    intermediate_score = []

    if character_image_width <= 98:
        for i in range(17):
            file_path = "data/SameSizeLetters/" + str(i + 1) + ".bmp"
            test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            test_character_width = test_char.shape[1]
            normalize_coef = test_char.shape[0] * character_image_width * 255

            # bitwise_not : The function calculates per-element bit-wise inversion of the input array:
            # dst[i] = ~src[i]

            # bitwise_xor : Calculates the per-element bit-wise “exclusive or” operation on two arrays or an array and a scalar.
            # Two arrays when src1 and src2 have the same size:
            # dst[i] = src1[i] xor src2[i] if mask[i] != 0
            for start in range(min([test_character_width - character_image_width - 1, 2])):
                crop_tc = test_char[:, start:start + character_image_width]

                intermediate_score.append(
                    np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image))) / normalize_coef)
            score[i] = max(intermediate_score)

            intermediate_score.clear()

        for i in range(10):
            file_path = "data/SameSizeNumbers/" + str(i) + ".bmp"
            test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            test_character_width = test_char.shape[1]
            normalize_coef = test_char.shape[0] * character_image_width * 255

            for start in range(min([test_character_width - character_image_width - 1, 2])):
                crop_tc = test_char[:, start:start + character_image_width]
                # bitwise_not : The function calculates per-element bit-wise inversion of the input array:
                # dst[i] = ~src[i]

                # bitwise_xor : Calculates the per-element bit-wise “exclusive or” operation on two arrays or an array and a scalar.
                # Two arrays when src1 and src2 have the same size:
                # dst[i] = src1[i] xor src2[i] if mask[i] != 0

                intermediate_score.append(
                    np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image))) / normalize_coef)
            score[17 + i] = max(intermediate_score)
            intermediate_score.clear()

        test_char = functions.create_bar_character(character_image.shape, 10, 15)  # bar character
        test_character_height = test_char.shape[0]
        normalize_coef = character_image.shape[0] * character_image_width * 255

        for start in range(test_character_height - character_image.shape[0] - 1):
            crop_tc = test_char[start:start + character_image.shape[0], :]
            # bitwise_not : The function calculates per-element bit-wise inversion of the input array:
            # dst[i] = ~src[i]

            # bitwise_xor : Calculates the per-element bit-wise “exclusive or” operation on two arrays or an array and a scalar.
            # Two arrays when src1 and src2 have the same size:
            # dst[i] = src1[i] xor src2[i] if mask[i] != 0
            intermediate_score.append(
                np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image))) / normalize_coef)
        score[27] = max(intermediate_score)
        intermediate_score.clear()

        return functions.lookup_table[str(np.argmax(score))]

    else:
        return ""


def segment_and_recognize(plate_img):
    plate_img = functions.clean_borders(plate_img, (4, 7))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    plate_img = cv2.dilate(plate_img, kernel, iterations=1)

    horizontal_project = np.sum(plate_img, axis=1)
    vertical_bounds = functions.find_vertical_bounds(horizontal_project, 16800)

    new_plate = plate_img[vertical_bounds[0] + 1:vertical_bounds[1]][:]

    resize_factor = 85 / new_plate.shape[0]
    dim = (int(new_plate.shape[1] * resize_factor), 85)
    new_plate = cv2.resize(new_plate, dim, interpolation=cv2.INTER_LINEAR)

    vertical_project = np.sum(new_plate, axis=0)
    horizontal_bounds = functions.find_horizontal_bounds(vertical_project)

    if len(horizontal_bounds) < 6:
        return None

    img_width = new_plate.shape[1]
    img_height = new_plate.shape[0]

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

    for bnd in vertical_bounds:
        plate_img = cv2.line(plate_img, (0, img_height - bnd), (img_width, img_height - bnd), (160, 0, 0), 1)
    for bnd in horizontal_bounds:
        new_plate = cv2.line(new_plate, (bnd, 0), (bnd, img_height), (0, 255, 0), 1)

    cv2.imshow('Plate image', new_plate)
    cv2.waitKey(25)

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_BGR2GRAY)
    plate_number = divide_characters(new_plate, horizontal_bounds)

    return plate_number
