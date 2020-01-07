import cv2
import imutils

import Recognize_help
import numpy as np
import math
import work
import temp

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

        distanceBetweenChars = Recognize_help.distanceBetweenChars(possibleC, possibleMatchingChar)

        angleBetweenChars = Recognize_help.angleBetweenChars(possibleC, possibleMatchingChar)

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

        possibleChars = []
        ctrs = []
        plates_list = []
        listOfListsOfMatchingChars = []
        countOfPossibleChars = 0

        for i in range(0, len(contours)):
            cv2.drawContours(imageContours, contours, i, (255, 255, 255))

            possibleChar = Recognize_help.ifChar(contours[i])

            if Recognize_help.checkIfChar(possibleChar) is True:
                countOfPossibleChars += 1
                possibleChars.append(possibleChar)

        imageContours = np.zeros((height, width, 3), np.uint8)

        for char in possibleChars:
            ctrs.append(char.contour)

        cv2.drawContours(imageContours, ctrs, -1, (255, 255, 255))

        for possibleC in possibleChars:
            listOfMatchingChars = matchingChars(possibleC, possibleChars)

            listOfMatchingChars.append(possibleC)
            if len(listOfMatchingChars) < 3:
                continue

            listOfListsOfMatchingChars.append(listOfMatchingChars)
            # listOfPossibleCharsWithCurrentMatchesRemoved = list(set(possibleChars) - set(listOfMatchingChars))

            recursiveListOfListsOfMatchingChars = []

            for recursiveListOfMatchingChars in recursiveListOfListsOfMatchingChars:
                listOfListsOfMatchingChars.append(recursiveListOfMatchingChars)

            break

        imageContours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingChars:
            contoursColor = (255, 0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

            cv2.drawContours(imageContours, contours, -1, contoursColor)

        for listOfMatchingChars in listOfListsOfMatchingChars:
            possiblePlate = Recognize_help.PossiblePlate()

            listOfMatchingChars.sort(key=lambda matchingChar: matchingChar.centerX)

            plateCenterX = (listOfMatchingChars[0].centerX + listOfMatchingChars[
                len(listOfMatchingChars) - 1].centerX) / 2.0
            plateCenterY = (listOfMatchingChars[0].centerY + listOfMatchingChars[
                len(listOfMatchingChars) - 1].centerY) / 2.0

            plateCenter = plateCenterX, plateCenterY

            plateWidth = int((listOfMatchingChars[len(listOfMatchingChars) - 1].boundingRectX + listOfMatchingChars[
                len(listOfMatchingChars) - 1].boundingRectWidth - listOfMatchingChars[0].boundingRectX) * 1.3)

            totalOfCharHeights = 0

            for matchingChar in listOfMatchingChars:
                totalOfCharHeights = totalOfCharHeights + matchingChar.boundingRectHeight

            averageCharHeight = totalOfCharHeights / len(listOfMatchingChars)

            plateHeight = int(averageCharHeight * 1.5)

            opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].centerY - listOfMatchingChars[0].centerY

            hypotenuse = Recognize_help.distanceBetweenChars(listOfMatchingChars[0],
                                                             listOfMatchingChars[len(listOfMatchingChars) - 1])
            correctionAngleInRad = math.asin(opposite / hypotenuse)
            correctionAngleInDeg = correctionAngleInRad * (180.0 / math.pi)

            possiblePlate.rrLocationOfPlateInScene = (
                tuple(plateCenter), (plateWidth, plateHeight), correctionAngleInDeg)

            rotationMatrix = cv2.getRotationMatrix2D(tuple(plateCenter), correctionAngleInDeg, 1.0)

            height, width, numChannels = frame.shape

            imgRotated = cv2.warpAffine(frame, rotationMatrix, (width, height))
            imgCropped = cv2.getRectSubPix(imgRotated, (plateWidth, plateHeight), tuple(plateCenter))
            possiblePlate.Plate = imgCropped

            if possiblePlate.Plate is not None:
                plates_list.append(possiblePlate)

            for i in range(0, len(plates_list)):
                p2fRectPoints = cv2.boxPoints(plates_list[i].rrLocationOfPlateInScene)

                rectColour = (255, 0, 0)

                cv2.line(imageContours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
                cv2.line(imageContours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

                cv2.line(frame, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), rectColour, 2)
                cv2.line(frame, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), rectColour, 2)
                cv2.line(frame, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), rectColour, 2)
                cv2.line(frame, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), rectColour, 2)

                # cv2.imshow("detected", imageContours)
                # cv2.imwrite(temp_folder + '11 - detected.png', imageContours)

                # plateNumber = work.show(possiblePlate.Plate)
                # print(plateNumber, "found")
                # cv2.imshow("frame_detected", frame)

                # temp.startcode(possiblePlate.Plate)

            number += 1
            cv2.imshow("plate", possiblePlate.Plate)
            temp.find(possiblePlate.Plate)
            # print(findNumber(possiblePlate.Plate))

    cap.release()
    cv2.destroyAllWindows()


def findNumber(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU)
    cv2.imshow('image', thresh)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    digitCnts = []

    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        print(x,y,w,h)

        # if the contour is sufficiently large, it must be a digit
        if w >= 15 and (h >= 30 and h <= 40):
            digitCnts.append(c)

    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)
        roi = thresh[y:y + h, x:x + w]

        # compute the width and height of each of the 7 segments
        # we are going to examine
        (roiH, roiW) = roi.shape
        (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
        dHC = int(roiH * 0.05)