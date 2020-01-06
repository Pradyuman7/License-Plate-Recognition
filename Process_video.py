import cv2
import Recognize_help
import numpy as np
import math
import work

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

                work.work(possiblePlate.Plate)
                # print(plateNumber, "found")

                cv2.imshow("frame_detected", frame)

            number += 1
            # plates = glob.glob("plates\\*.png")
            # processed = glob.glob("processed\\*.png")
            # resized = glob.glob("resized\\*.png")
            # bordered = glob.glob("bordered\\*.png")
            # work.work(plates, processed, resized, bordered)

    cap.release()
    cv2.destroyAllWindows()

# class Plates:
#     def __init__(self, ori_img, img_name, temp_num, color, pm_thresh, option=None):
#         self.img_name = img_name
#         self.ori = ori_img.copy()
#         self.result = ori_img
#         self.img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
#         self.temp_num = temp_num
#         self.color = color
#         self.pm_thresh = pm_thresh
#         self.option = option
#
#         # Do the work
#         self.img_process()
#         self.contour()
#         self.pattern_matching()
#         self.show_plt()
#
#         cv2.waitKey(0)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#
#     def img_process(self):
#         '''
#         Using standard Threshold function and inRange function to filter
#         Get the high threshold and filter it again with inRange color extraction
#         Some of images are categorized as 'special' where they have their own upper and lower threshold color range
#         Did canny edge to detect edges after the images feature are highlighted
#         Thus, the edges are identified from the highlighted area only.
#         '''
#         height = self.img.shape[0]
#         width = self.img.shape[1]
#         kernel = np.ones((2, 2), np.uint8)
#
#         # Threshold
#         _, mask = cv2.threshold(self.img, thresh=200, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         if self.option is not None and self.option['no_bitwise']:
#             self.img_mask = self.img
#         else:
#             self.img_mask = cv2.bitwise_and(self.img, mask)
#
#         # inRange Threshold function
#         if self.color in ('white', 'yellow'):
#             hsv = cv2.cvtColor(self.ori, cv2.COLOR_BGR2HSV)
#             h, s, v1 = cv2.split(hsv)
#             if self.color == 'white':
#                 lower_white = np.array([0, 0, 160], dtype=np.uint8)
#                 upper_white = np.array([255, 40, 255], dtype=np.uint8)
#             elif self.color == 'yellow':
#                 lower_white = np.array([20, 100, 100], dtype=np.uint8)
#                 upper_white = np.array([30, 255, 255], dtype=np.uint8)
#             res_mask = cv2.inRange(hsv, lower_white, upper_white)
#             self.res_img = cv2.bitwise_and(v1, self.img, mask=res_mask)
#         else:
#             if self.color == 'special' and self.option is not None and self.option['type'] == 'color':
#                 print(":::::Special - color")
#                 hsv = cv2.cvtColor(self.ori, cv2.COLOR_BGR2HSV)
#                 h, s, v1 = cv2.split(hsv)
#                 upper_white = self.option['upper_white']
#                 lower_white = self.option['lower_white']
#                 res_mask = cv2.inRange(hsv, lower_white, upper_white)
#                 self.res_img = cv2.bitwise_and(v1, self.img, mask=res_mask)
#             else:
#                 self.res_img = self.img_mask
#
#         # Edge Detection
#         self.edges = cv2.Canny(self.res_img, height, width)
#         # self.edges = cv2.Laplacian(self.res_img, cv2.CV_64F)
#         # self.edges = cv2.Sobel(self.res_img, cv2.CV_64F, 0, 1, ksize=5)
#         return
#
#     def contour(self):
#         '''
#         Contour area of highlighted images
#         Get some of the most contour areas
#         Calculate the polygonal curve, pick with 4 curve (rect) OR
#             Draw convex hull on the biggest contour area
#         Crop of the picked contour/convex area
#         '''
#
#         # Contours
#         contours, hierarchy = cv2.findContours(
#             self.res_img,
#             cv2.RETR_TREE,
#             cv2.CHAIN_APPROX_SIMPLE
#         )
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
#         # cv2.drawContours(self.result, contours, -1, (150, 150, 255), 2)
#         NumberPlateCnt = None
#         found = False
#         lt, rb = [10000, 10000], [0, 0]
#
#         if self.color == 'white_bg':
#             # Calculate polygonal curve, see if it has 4 curve
#             for c in contours:
#                 peri = cv2.arcLength(c, True)
#                 approx = cv2.approxPolyDP(c, 0.06 * peri, True)
#                 if len(approx) == 4:
#                     found = True
#                     NumberPlateCnt = approx
#                     break
#             if found:
#                 cv2.drawContours(self.result, [NumberPlateCnt], -1, (255, 0, 255), 2)
#
#                 for point in NumberPlateCnt:
#                     cur_cx, cur_cy = point[0][0], point[0][1]
#                     if cur_cx < lt[0]: lt[0] = cur_cx
#                     if cur_cx > rb[0]: rb[0] = cur_cx
#                     if cur_cy < lt[1]: lt[1] = cur_cy
#                     if cur_cy > rb[1]: rb[1] = cur_cy
#
#                 cv2.circle(self.result, (lt[0], lt[1]), 2, (150, 200, 255), 2)
#                 cv2.circle(self.result, (rb[0], rb[1]), 2, (150, 200, 255), 2)
#
#                 self.crop = self.res_img[lt[1]:rb[1], lt[0]:rb[0]]
#                 self.crop_res = self.ori[lt[1]:rb[1], lt[0]:rb[0]]
#             else:
#                 self.crop = self.res_img.copy()
#                 self.crop_res = self.ori.copy()
#         elif len(contours) > 0:
#             # Convex Hull
#             hull = cv2.convexHull(contours[0])
#             # cv2.drawContours(ori_img, [hull], -1, (255, 0, 255),  2, 8)
#             approx2 = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
#             cv2.drawContours(self.result, [approx2], -1, (255, 0, 255), 2, lineType=8)
#
#             for point in approx2:
#                 cur_cx, cur_cy = point[0][0], point[0][1]
#                 if cur_cx < lt[0]: lt[0] = cur_cx
#                 if cur_cx > rb[0]: rb[0] = cur_cx
#                 if cur_cy < lt[1]: lt[1] = cur_cy
#                 if cur_cy > rb[1]: rb[1] = cur_cy
#
#             cv2.circle(self.result, (lt[0], lt[1]), 2, (150, 200, 255), 2)
#             cv2.circle(self.result, (rb[0], rb[1]), 2, (150, 200, 255), 2)
#
#             self.crop = self.res_img[lt[1]:rb[1], lt[0]:rb[0]]
#             self.crop_res = self.ori[lt[1]:rb[1], lt[0]:rb[0]]
#         else:
#             self.crop = self.res_img.copy()
#             self.crop_res = self.ori.copy()
#
#         return
#
#     def pattern_matching(self):
#         '''
#         Pattern Matching is used to identify numbers in the cropped image
#         CV_TM_CCOEFF is used in this case
#         * Result is still not accurate
#         '''
#         self.pm = {}
#
#         method = cv2.TM_CCOEFF_NORMED
#         threshold = self.pm_thresh
#         cw, ch = self.crop.shape[::-1]
#
#         # cv2.imshow("crop", self.crop)
#
#         for temp in self.temp_num:
#             highest = 0
#             highest_pt = []
#             for i in range(1, 4):
#                 temp_result = []
#                 t_img = cv2.imread("./temp-num/{}-0{}.png".format(temp, str(i)), 0)
#                 t_img = imutils.resize(t_img, height=ch - 2)
#                 w, h = t_img.shape[::-1]
#
#                 res = cv2.matchTemplate(self.crop, t_img, method)
#                 loc = np.where(res >= threshold)
#                 for pt in zip(*loc[::-1]):
#                     temp_result.append(pt)
#                     cv2.rectangle(self.crop_res, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
#                 if len(temp_result) > highest:
#                     highest = len(temp_result)
#                     highest_pt = temp_result
#
#             for pt in highest_pt:
#                 cv2.rectangle(self.crop_res, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 1)
#                 self.pm[pt[0]] = temp
#
#         self.pm = collections.OrderedDict(sorted(self.pm.items()))
#         self.pm_result = ''
#         for _, pm in self.pm.items():
#             self.pm_result += pm
#
#         print("::::::RESULT = {}\n".format(self.pm_result))
#         return
#
#     def show_plt(self):
#         '''
#         Showing 6 main step of the process for turning original frame to the result frame
#         '''
#         title = [
#             'Black and White',
#             'Threshold',
#             'Canny',
#             'Num Plate Detected',
#             'Num Plate Cropped',
#             'Predicted Num :\n' + self.pm_result
#         ]
#         result = [self.img, self.res_img, self.edges, self.result[:, :, ::-1], self.crop, self.crop_res]
#         num = [231, 232, 233, 234, 235, 236]
#
#         for i in range(len(result)):
#             plt.subplot(num[i]), plt.imshow(result[i], cmap='gray')
#             plt.title(title[i]), plt.xticks([]), plt.yticks([])
#
#         plt.suptitle(self.img_name)
#         plt.show()
#
#
# def CaptureFrame_Process_ha(file_path, sample_frequency, output_path):
#     cap = cv2.VideoCapture(file_path)
#     temp_num = [f for f in os.listdir('./temp-num') if os.path.isfile(os.path.join('./temp-num', f))]
#     # plt.rcParams["figure.figsize"] = (15, 10)
#
#     while True:
#         flag, image = cap.read()
#
#         if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
#             break
#
#         Plates(image, 'name', temp_files, 'type', 0.20, None)
#
#     cap.release()
#     cv2.destroyAllWindows()


# def CaptureFrame_Process_try(file_path, sample_frequency, save_path):
#     cap = cv2.VideoCapture(file_path)
#
#     characters = []
#     glob_characters = glob.glob("characters/*.png")
#     for char in glob_characters:
#         characters.append(char)
#
#     while True:
#         flag, image = cap.read()
#
#         if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
#             break
#
#         # Input image processing
#         img_original = image
#         mask = np.zeros(img_original.shape[:2], np.uint8)
#         img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
#         img_contrast = cv2.convertScaleAbs(img_grayscale, alpha=1.25, beta=0)
#         (_, img_threshold) = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_OTSU)
#         (contours, _) = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
#         # Üres lista, amibe később a talált karakterek kerülnek
#         found_characters = []
#
#         for contour in contours:
#             (x, y, width, height) = cv2.boundingRect(contour)
#
#             area = width * height
#             roi = img_contrast[y:y + height, x:x + width]
#
#             if (2000 < area < 50000) and (width >= height * 2) and (width <= height * 6):
#                 possible_plate = cv2.resize(roi, (230, 50))
#
#                 characters_found = 0
#
#                 for character in characters:
#                     char = cv2.imread(character, 0)
#                     (w, h) = char.shape[::-1]
#                     res = cv2.matchTemplate(possible_plate, char, cv2.TM_CCOEFF_NORMED)
#
#                     if (w <= 20) and (h >= 20):  # '1' and 'I' characters
#                         threshold = 0.9
#                     elif h <= 20:  # '-' character
#                         threshold = 0.92
#                     else:
#                         threshold = 0.8
#
#                     loc = np.where(res > threshold)
#
#                     for pt in zip(*loc[::-1]):
#                         name = str(path.basename(character)).split(".")[0]
#                         top_left = pt
#                         bottom_right = (pt[0] + w, pt[1] + h)
#                         col = top_left[0]
#                         row = top_left[1]
#                         if mask[row + h // 2, col + w // 2] != 255:
#                             mask[row:row + h, col:col + w] = 255
#                             cv2.rectangle(possible_plate, top_left, bottom_right, (0, 0, 255), 1)
#                             characters_found += 1
#                             found_characters.append((name, col))
#
#                 if characters_found > 1:
#                     img_license_plate = possible_plate
#                     cv2.rectangle(img_original, (x, y), (x + width, y + height), (0, 255, 0), 2)
#
#         # expected_plate = str(path.basename(image)).split(".")[0]
#         # print("\nVárt rendszám:\n{0}".format(expected_plate))
#
#         def sortBySecond(element):
#             return element[1]
#
#         found_characters.sort(key=sortBySecond)
#
#         license_plate = ""
#
#         for character in found_characters:
#             license_plate += character[0]
#
#         # print(license_plate)
#         # print(
#         #     "\nProvision number recognized successfully" if license_plate is not None else "\nProperty recognition "
#         #                                                                                    "failed!")
#
#         if len(found_characters) > 0:
#             print(license_plate)
#             cv2.imshow("frame", img_original)
#             # cv2.imshow("Found System", img_license_plate)
#         else:
#             print("\nNone found!")
#
#     cap.release()
#     cv2.destroyAllWindows()
