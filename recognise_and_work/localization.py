import cv2
import numpy as np

from helpers import functions


def plate_detection(image, contours):

    final_contours = []
    plate_img = []
    i = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if functions.verify_plate(box):
            final_contours.append(box)
        i += 1


    for f_cnt in final_contours:
        plate_img.append(functions.four_point_transform(image, f_cnt))
        cv2.imshow('Localized', plate_img[len(plate_img) - 1])
        cv2.waitKey(25)

    return plate_img
