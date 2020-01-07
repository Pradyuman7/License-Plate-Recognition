import cv2
import imutils
from PIL import Image
import os
import find_simple

import Recognize_help
import numpy as np
import math
import work
import find_tesseract

"""
In this file, you will define your own CaptureFrame_Process funtion. In this function,
you need three arguments: file_path(str type, the video file), sample_frequency(second), save_path(final results saving path).
To do:
	1. Capture the frames for the whole video by your sample_frequency, record the frame number and timestamp(seconds).
	2. Localize and recognize the plates in the frame.(Hints: need to use 'Localization.plate_detection' and 'Recognize.segmetn_and_recognize' functions)
	3. If recognizing any plates, save them into a .csv file.(Hints: may need to use 'pandas' package)
Inputs:(three)
	1. file_path: video path
	2. sample_frequency: second
	3. save_path: final .csv file path
Output: None
"""


def matchingChars(possibleC, possibleChars):
    listOfMatchingChars = []

    for possibleMatchingChar in possibleChars:
        if possibleMatchingChar == possibleC:
            continue

        distanceBetweenChars = Recognize_help.distanceBetweenCharacters(possibleC, possibleMatchingChar)

        angleBetweenChars = Recognize_help.angleBetweenCharacters(possibleC, possibleMatchingChar)

        changeInArea = float(abs(possibleMatchingChar.boundingRectArea - possibleC.boundingRectArea)) / float(
            possibleC.boundingRectArea)

        changeInWidth = float(abs(possibleMatchingChar.boundingRectWidth - possibleC.boundingRectWidth)) / float(
            possibleC.boundingRectWidth)

        changeInHeight = float(abs(possibleMatchingChar.boundingRectHeight - possibleC.boundingRectHeight)) / float(
            possibleC.boundingRectHeight)

        if distanceBetweenChars < (possibleC.diagonalSize * 5) and \
                angleBetweenChars < 12.0 and \
                changeInArea < 0.5 and \
                changeInWidth < 0.8 and \
                changeInHeight < 0.2:
            listOfMatchingChars.append(possibleMatchingChar)

    return listOfMatchingChars


def CaptureFrame_Process(file_path, sample_frequency, save_path):
    cap = cv2.VideoCapture(file_path)
    number = 0
    count = 0

    while True:
        flag, frame = cap.read()
        plateNumber = ""

        if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('frame_gray', gray)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        # cv2.imshow('frame_hsv', hsv)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        topHat = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, kernel)
        blackHat = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, kernel)

        add = cv2.add(value, topHat)
        subtract = cv2.subtract(add, blackHat)

        blur = cv2.GaussianBlur(subtract, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

        # opencv 4
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        height, width = thresh.shape
        imageContours = np.zeros((height, width, 3), dtype=np.uint8)

        characters = []
        contours_first = []
        plates_list = []
        list_of_list = []
        number_of_characters = 0

        for i in range(0, len(contours)):
            cv2.drawContours(imageContours, contours, i, (255, 255, 255))

            character = Recognize_help.Character(contours[i])

            if Recognize_help.isThisACharacter(character) is True:
                number_of_characters += 1
                characters.append(character)

        imageContours = np.zeros((height, width, 3), np.uint8)

        for char in characters:
            contours_first.append(char.contour)

        cv2.drawContours(imageContours, contours_first, -1, (255, 255, 255))

        for possibleC in characters:
            list_of_match = matchingChars(possibleC, characters)

            list_of_match.append(possibleC)
            if len(list_of_match) < 3:
                continue

            list_of_list.append(list_of_match)
            # listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

            recursiveListOfListsOfMatchingChars = []

            for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
                list_of_list.append(recursiveListOfMatchingChars)

            break

        imageContours = np.zeros((height, width, 3), np.uint8)

        for list_of_match in list_of_list:
            contoursColor = (255, 0, 255)

            contours = []

            for matchingChar in list_of_match:
                contours.append(matchingChar.contour)

            cv2.drawContours(imageContours, contours, -1, contoursColor)

        for list_of_match in list_of_list:
            plate = Recognize_help.LicensePlate()

            list_of_match.sort(key=lambda matchingChar: matchingChar.centerX)

            plateCenterX = (list_of_match[0].centerX + list_of_match[
                len(list_of_match) - 1].centerX) / 2.0
            plateCenterY = (list_of_match[0].centerY + list_of_match[
                len(list_of_match) - 1].centerY) / 2.0

            plateCenter = plateCenterX, plateCenterY

            plateWidth = int((list_of_match[len(list_of_match) - 1].boundingRectX + list_of_match[
                len(list_of_match) - 1].boundingRectWidth - list_of_match[0].boundingRectX) * 1.3)

            total_character_height = 0

            for matchingChar in list_of_match:
                total_character_height = total_character_height + matchingChar.boundingRectHeight

            averageCharHeight = total_character_height / len(list_of_match)

            plateHeight = int(averageCharHeight * 1.5)

            opposite = list_of_match[len(list_of_match) - 1].centerY - list_of_match[0].centerY

            hypotenuse = Recognize_help.distanceBetweenCharacters(list_of_match[0],
                                                                  list_of_match[len(list_of_match) - 1])
            correctionAngleInRad = math.asin(opposite / hypotenuse)
            correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

            plate.locationOfPlate = (tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

            rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

            height, width, numChannels = frame.shape

            imgRotated = cv2.warpAffine(frame, rotationMatrix, (width, height))
            imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))
            plate.Plate = imgCropped

            plate.chars = list_of_match

            if plate.Plate is not None:
                plates_list.append(plate)

            for i in range(0, len(plates_list)):
                p2fRectPoints = cv2.boxPoints(plates_list[i].locationOfPlate)

                rectColour = (255, 0, 0)

                cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

                cv2.line(frame, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
                cv2.line(frame, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
                cv2.line(frame, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
                cv2.line(frame, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

                # cv2.imshow("plate", plate.Plate)
                # temp.find(possiblePlate.Plate)
                # findNumber(plate)
                find_simple.find(plate.Plate)

            number += 1

    cap.release()
    cv2.destroyAllWindows()


def find_matches(small_image, large_image):
    method = cv2.TM_SQDIFF_NORMED

    result = cv2.matchTemplate(small_image, large_image, method)

    # We want the minimum squared difference
    mn, _, mnLoc, _ = cv2.minMaxLoc(result)

    # Draw the rectangle:
    # Extract the coordinates of our best match
    MPx, MPy = mnLoc

    # Step 2: Get the size of the template. This is the same size as the match.
    trows, tcols = small_image.shape[:2]

    # Step 3: Draw the rectangle on large_image
    cv2.rectangle(large_image, (MPx, MPy), (MPx + tcols, MPy + trows), (0, 0, 255), 2)

    # Display the original image with the rectangle around the match.
    cv2.imshow('output', large_image)

    # The image is only displayed if we call this
    cv2.waitKey(0)


def findNumber(plate):
    path = "/Users/pradyuman.dixit/Desktop/Students_upload/characters-2"
    images = os.listdir(path)

    for im in images:
        tempImage = cv2.imread(path + "/" + im)
        find_matches(tempImage, plate.Plate)