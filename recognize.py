import cv2
import numpy as np
import matplotlib.pyplot as plt


def search_boundary_1(hp, T):
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


def search_boundary_2(proj):
    N = len(proj)
    bool_bounds = (proj >= 255)
    start = 0
    end = 1
    bounds = []

    for b in range(N - 1):
        if bool_bounds[end] & ~bool_bounds[start]:
            bounds.append(end)

        start += 1
        end += 1

    bounds.append(end - 20)
    return bounds


def segmentation(image, bounds):
    N = len(bounds)
    plate_number = ""

    for i in range(N - 1):
        # filename = "Characters/character_" + str(i) + ".jpg"
        character_image = image[:, bounds[i]:bounds[i + 1]]
        # cv2.imwrite(filename, character_image)
        plate_number += recognize_character(character_image)

    return plate_number


def recognize_character(image):
    characters = {
        "0": "B", "1": "D", "2": "F", "3": "G", "4": "H",
        "5": "J", "6": "K", "7": "L", "8": "M", "9": "N",
        "10": "P", "11": "R", "12": "S", "13": "T", "14": "V",
        "15": "X", "16": "Z", "17": "0", "18": "1", "19": "2",
        "20": "3", "21": "4", "22": "5", "23": "6", "24": "7",
        "25": "8", "26": "9", "27": "-"
    }

    image_width = image.shape[1]
    score = np.zeros(28)
    intermediate_score = []

    for i in range(17):
        file_path = "SameSizeLetters/" + str(i + 1) + ".bmp"
        test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        test_character_width = test_char.shape[1]
        normalize_coef = test_char.shape[0] * image_width * 255

        for start in range(test_character_width - image_width - 1):
            crop_tc = test_char[:, start:start + image_width]
            intermediate_score.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, image))) / normalize_coef)

        if len(intermediate_score) == 0:
            continue

        score[i] = max(intermediate_score)
        intermediate_score.clear()

    for i in range(10):
        file_path = "SameSizeNumbers/" + str(i) + ".bmp"
        test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        test_character_width = test_char.shape[1]
        normalize_coef = test_char.shape[0] * image_width * 255

        for start in range(test_character_width - image_width - 1):
            crop_tc = test_char[:, start:start + image_width]
            intermediate_score.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, image))) / normalize_coef)

        if len(intermediate_score) == 0:
            continue

        score[17 + i] = max(intermediate_score)
        intermediate_score.clear()

    test_char = blank_characters(image.shape, 10, 20)
    test_character_width = test_char.shape[1]
    normalize_coef = test_char.shape[0] * image_width * 255

    for start in range(test_character_width - image_width - 1):
        crop_tc = test_char[:, start:start + image_width]
        intermediate_score.append(np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, image))) / normalize_coef)

        if len(intermediate_score) == 0:
            continue

    score[27] = max(intermediate_score)
    intermediate_score.clear()

    print("Character: ", characters[str(np.argmax(score))])
    return characters[str(np.argmax(score))]


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


def dft(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def remove_borders(plate_image, epsilon):
    height = plate_image.shape[0]
    width = plate_image.shape[1]

    plate_image[0:epsilon[0], :] = 0
    plate_image[height - epsilon[0]:height, :] = 0
    plate_image[:, 0:epsilon[1]] = 0
    plate_image[:, width - epsilon[1]:width] = 0

    return plate_image


def recognition(plate_img):
    plate_img = remove_borders(plate_img, (13, 13))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    horizontal_project = np.sum(plate_img, axis=1)
    vertical_bounds = search_boundary_1(horizontal_project, 1000)

    new_plate = plate_img[vertical_bounds[0] + 1:vertical_bounds[1]][:]
    resize_factor = 85 / new_plate.shape[0]
    dim = (int(new_plate.shape[1] * resize_factor), 85)
    new_plate = cv2.resize(new_plate, dim, interpolation=cv2.INTER_LINEAR)

    vertical_project = np.sum(new_plate, axis=0)
    horizontal_bounds = search_boundary_2(vertical_project)

    img_width = new_plate.shape[1]
    img_height = new_plate.shape[0]

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

    for bnd in vertical_bounds:
        plate_img = cv2.line(plate_img, (0, img_height - bnd), (img_width, img_height - bnd), (160, 0, 0), 1)
    for bnd in horizontal_bounds:
        new_plate = cv2.line(new_plate, (bnd, 0), (bnd, img_height), (0, 255, 0), 1)

    # cv2.imshow('Plate image', new_plate)
    # cv2.waitKey(0)

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_BGR2GRAY)
    plate_number = segmentation(new_plate, horizontal_bounds)

    return plate_number
