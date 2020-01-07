import cv2
import numpy as np
import os
from imutils import contours


# python3 start.py --file_path '/Users/pradyuman.dixit/Desktop/Students_upload/trainingsvideo.avi'

# def work(image):
#     mask = np.zeros(image.shape, dtype=np.uint8)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
#     cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
#     ROI_number = 0
#
#     for c in cnts:
#         area = cv2.contourArea(c)
#
#         if 800 > area > 200:
#             x, y, w, h = cv2.boundingRect(c)
#             ROI = 255 - thresh[y:y + h, x:x + w]
#             cv2.drawContours(mask, [c], -1, (255, 0, 0), 1)
#             # cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
#             ROI_number += 1
#
#     cv2.imshow('mask', mask)
#     cv2.imshow('thresh', thresh)
#     cv2.waitKey()

def work(image):
    path = "/Users/pradyuman.dixit/Desktop/Students_upload"

    for image_path in os.listdir(path + "/characters-2"):
        template = cv2.imread(os.path.join(path, "characters-2", image_path))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        template = template.astype(np.uint8)
        image = image.astype(np.uint8)

        res = cv2.matchTemplate(template, image, cv2.TM_SQDIFF_NORMED)
        mn, _, mnLoc, _ = cv2.minMaxLoc(res)

        if res is not None:
            return image_path.replace(".bmp", "")


def show(image):
    plate = ""
    # mask = np.zeros(image.shape, dtype=np.uint8)
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(image.shape)
    # print(image)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")

    for ctr in cnts:
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = thresh[y:y + h, x:x + w]

        character = work(roi)

        if character is not None:
            plate += character

        # cv2.imshow('character: %d' % d, roi)
        # area = cv2.contourArea(con)
        #
        # if 800 > area > 200:
        #     x, y, w, h = cv2.boundingRect(con)
        #     # cv2.drawContours(mask, [c], 1, (255, 0, 0), 2)
        #     temp = thresh[y:y+h, x:x+w]
        #
        #     character = work(temp)
        #
        #     if character is not None:
        #         plate += character

    return plate
