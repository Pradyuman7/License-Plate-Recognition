import cv2
import numpy as np


lookup_table = {
    "0": "B", "1": "D", "2": "F", "3": "G", "4": "H",
    "5": "J", "6": "K", "7": "L", "8": "M", "9": "N",
    "10": "P", "11": "R", "12": "S", "13": "T", "14": "V",
    "15": "X", "16": "Z", "17": "0", "18": "1", "19": "2",
    "20": "3", "21": "4", "22": "5", "23": "6", "24": "7",
    "25": "8", "26": "9", "27": "-"
}


# T is the threshold to identify the transition between blank and characters region
def find_vertical_bounds(hp, T):
    N = len(hp)
    # Find inferior bound
    i = 0
    while ~((hp[i] <= T) & (hp[i + 1] > T)) & (i < int(N / 2)):
        i += 1
    inf_bound = 0 if i == int(N / 2) else i

    # Find superior bound
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


def divide_characters(image, bounds):
    N = len(bounds)
    plate_number = ""
    # compute_dft(test_char)  # dft of the character image
    for i in range(N - 1):
        # filename = "Characters/character_" + str(i) + ".jpg"
        character_image = image[:, bounds[i]:bounds[i + 1]]
        # cv2.imwrite(filename, character_image)
        plate_number = plate_number + match_characters(character_image)
    # compute_dft(character_image)

    # Check for problems with "-" character
    indexes = find_all_indexes(plate_number, "-")
    N = len(indexes)
    M = len(plate_number)
    if N:
        if (N != 2) or (indexes[0] == 0) or (indexes[N - 1] == M - 1):
            return None

    # Check for plates with length != than 8 characters
    if M != 8:
        return None

    # Check for plates with consecutive "-" characters
    for i in range(N - 1):
        if indexes[i] == indexes[i + 1] - 1:
            return None

    return plate_number


def match_characters(character_image):
    # show image
    # cv2.imshow('match', character_image)
    # cv2.waitKey(0)

    character_image_width = character_image.shape[1]
    score = np.zeros(28)
    intermediate_score = []

    if character_image_width <= 98:
        # Compute scores for Letters
        for i in range(17):
            file_path = "data/SameSizeLetters/" + str(i + 1) + ".bmp"
            test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            test_character_width = test_char.shape[1]
            normalize_coef = test_char.shape[0] * character_image_width * 255
            # crop_tc = test_char[:, :character_image_width]
            for start in range(min([test_character_width - character_image_width - 1, 2])):
                crop_tc = test_char[:, start:start + character_image_width]
                # cv2.imshow('match', cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))
                # cv2.waitKey(25)
                intermediate_score.append(
                    np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image))) / normalize_coef)
            score[i] = max(intermediate_score)
            # score[i] = np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))/normalize_coef
            intermediate_score.clear()

        # Compute scores for Numbers
        for i in range(10):
            file_path = "data/SameSizeNumbers/" + str(i) + ".bmp"
            test_char = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            test_character_width = test_char.shape[1]
            normalize_coef = test_char.shape[0] * character_image_width * 255
            # crop_tc = test_char[:, :character_image_width]
            for start in range(min([test_character_width - character_image_width - 1, 2])):
                crop_tc = test_char[:, start:start + character_image_width]
                # cv2.imshow('match', cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))
                # cv2.waitKey(25)
                intermediate_score.append(
                    np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image))) / normalize_coef)
            score[17 + i] = max(intermediate_score)
            # score[17 + i] = np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))/normalize_coef
            intermediate_score.clear()

        # Test for bar character
        test_char = create_bar_character(character_image.shape, 10, 15)  # bar character
        test_character_height = test_char.shape[0]
        normalize_coef = character_image.shape[0] * character_image_width * 255
        for start in range(test_character_height - character_image.shape[0] - 1):
            crop_tc = test_char[start:start + character_image.shape[0], :]
            # cv2.imshow('match', cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image)))
            # cv2.waitKey(0)
            intermediate_score.append(
                np.sum(cv2.bitwise_not(cv2.bitwise_xor(crop_tc, character_image))) / normalize_coef)
        score[27] = max(intermediate_score)
        intermediate_score.clear()

        sum_pix = np.sum(character_image)
        if sum_pix < 40000:
            return ""

        # print maximal score for each characters to compare and its index
        # print("Character matched is : ", lookup_table[str(np.argmax(score))])
        return lookup_table[str(np.argmax(score))]

    else:
        return ""


def create_bar_character(img_shape, bar_thickness, bar_width):
    # Check for special characters i.e. blank or bar
    ch_height = img_shape[0] + 50
    ch_width = img_shape[1]
    bar = np.zeros((ch_height, ch_width), np.uint8)
    bart_init = int(ch_height / 2) - int(bar_thickness / 2)
    bart_end = bart_init + bar_thickness
    # barw_init = int(ch_width / 2) - int(bar_width / 2)
    barw_init = 0
    barw_end = barw_init + bar_width
    bar[bart_init:bart_end, barw_init:barw_end] = 255 * np.ones([bar_thickness, min((bar_width, ch_width))])
    return bar


# This function aims to clean the pixels close to the borders of the plate image (should be gray_scaled)
# epsilon is the width of the cleaning zone around the borders
# epsilon = (epsilon_h, epsilon_w)
def clean_borders(plate_image, epsilon):
    height = plate_image.shape[0]
    width = plate_image.shape[1]

    plate_image[0:epsilon[0], :] = 0
    plate_image[height - epsilon[0]:height, :] = 0
    plate_image[:, 0:epsilon[1]] = 0
    plate_image[:, width - epsilon[1]:width] = 0

    return plate_image


def segment_and_recognize(plate_img):
    # Maybe implement erosion or dilatation techniques to improve recognition
    # Explore connected components analysis

    # Clean image borders
    plate_img = clean_borders(plate_img, (4, 7))

    # dilatation (to close gaps in the letters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    plate_img = cv2.dilate(plate_img, kernel, iterations=1)
    # plate_img = cv2.morphologyEx(plate_img, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('Clean borders', plate_img)
    # cv2.waitKey(0)

    # Perform projections
    horizontal_project = np.sum(plate_img, axis=1)
    vertical_bounds = find_vertical_bounds(horizontal_project, 16800)  # 2nd argument is the threshold to detect bounds

    # crop the upper and lower boundaries
    new_plate = plate_img[vertical_bounds[0] + 1:vertical_bounds[1]][:]

    # reshape the plate image
    resize_factor = 85 / new_plate.shape[0]
    dim = (int(new_plate.shape[1] * resize_factor), 85)
    new_plate = cv2.resize(new_plate, dim, interpolation=cv2.INTER_LINEAR)

    # perform projections
    vertical_project = np.sum(new_plate, axis=0)
    horizontal_bounds = find_horizontal_bounds(vertical_project)
    if len(horizontal_bounds) < 6:  # 6 bounds = 5 characters, no plate has usually less than 5 characters
        return None

    # find vertical bounds
    img_width = new_plate.shape[1]
    img_height = new_plate.shape[0]

    # turn gray_scale to BGR
    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_GRAY2BGR)

    # draw bounding lines
    for bnd in vertical_bounds:
        plate_img = cv2.line(plate_img, (0, img_height - bnd), (img_width, img_height - bnd), (160, 0, 0), 1)
    for bnd in horizontal_bounds:
        new_plate = cv2.line(new_plate, (bnd, 0), (bnd, img_height), (0, 255, 0), 1)

    cv2.imshow('Plate image', new_plate)
    cv2.waitKey(25)

    # afficher les projections horizontales et verticales
    """xx = np.arange(len(horizontal_project))
    plt.plot(xx, horizontal_project)
    plt.show()"""

    new_plate = cv2.cvtColor(new_plate, cv2.COLOR_BGR2GRAY)
    plate_number = divide_characters(new_plate, horizontal_bounds)

    return plate_number
