import cv2
import pandas as pd

from recognise_and_work import localization
from helpers import functions
from recognise_and_work import recognize

# # recognizes the frame
def help_recognize(frame):
    ## blur the image to make all colors of the frame uniform

    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    # #hue, saturation, value model, used to select various different colors needed for a particular picture

    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # #find only orange parts of the image
    light_orange = (15, 60, 50)
    dark_orange = (37, 255, 220)
    # #masking happens -> from light orage to dark orange


    ## inRange : Checks if array elements lie between the elements of two other arrays.
    # #params, src is first input array, lower-bound inclusive, upper bound inclusive
    ## return dst where dst[i[ = low[i] <= src[i] <= up[i]
    mask = cv2.inRange(hsv_img, light_orange, dark_orange)
    ## bitwise_and : calculates the per-element bit-wise logical conjunction
    # #Two arrays when src1 and src2 have the same size:
    ## dst[i] = scr1[i] ^ scr2[i] if mask[i] != 0
    masked = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow("Masked", masked)
    cv2.waitKey(10)
    # convert hsv to gray
    ## convert hsv to gray
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    ## binarisation of the image
    (thresh, binary) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    ## threshold : Applies a fixed-level threshold to each array element.
    ## params, src, threshold value, max value


    # edge detection using canny algorithm

    edged = cv2.Canny(binary, 50, 100)

   # # Canny : Finds edges in an image using the [Canny86] algorithm.
    ## params, image, thesh_1, thresh_2
    ## The smallest value between threshold1 and threshold2 is used for edge linking.
    ## The largest value is used to find initial segments of strong edges.
    ## canny algorithm works as following :
    ## Apply Gaussian filter to smooth the image in order to remove the noise
    ## Find the intensity gradients of the image (An image gradient is a
    ## directional change in the intensity or color in an image)
    ## Apply non-maximum suppression to get rid of spurious response to edge detection
    ## NMS : transform a smooth response map that triggers many imprecise object
    ## window hypotheses in, ideally, a single bounding-box for each detected object.
    ## Apply double threshold to determine potential edges
    ## Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges
    ## that are weak and not connected to strong edges.
    ## hysterisis: hysteresis compares two images to build an intermediate image. The function
    ## takes two binary images that have been thresholded at different levels. The higher threshold
    ## has a smaller population of white pixels. The values in the higher threshold are more likely to be real edges.

    ## find contours of the image

    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    ## findContours : Finds contours in a binary image.
    ## params, image, mode, method
    ## retr_tree retrieves all of the contours and reconstructs a full hierarchy of nested contours
    ## chain_approx_none stores absolutely all the contour points. That is, any 2 subsequent points
    ## (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors,
    ## that is, max(abs(x1-x2),abs(y2-y1))==1.
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ## the image and the contours are passed to the plate detection method
    plates = localization.plate_detection(gray_original, contours)

    if plates is not None:
        plate_number = []

        for plate_image in plates:
            # intermediate_plate_number =helper(plate_image)

            resize_factor = 85 / plate_image.shape[0]
            dim = (int(plate_image.shape[1] * resize_factor), 85)
            plate_image = cv2.resize(plate_image, dim, interpolation=cv2.INTER_LINEAR)
            epsilon = 10
            plate_image = plate_image[epsilon:plate_image.shape[0] - epsilon, epsilon:plate_image.shape[1] - epsilon]
            plate_image = cv2.GaussianBlur(plate_image, (5, 5), 0)
            cv2.imshow("plate_image", plate_image)
            cv2.waitKey(25)

            first_time = 0
            intermediate_plate_number = None
            while (first_time < 5) and (intermediate_plate_number is None):
                if first_time == 0:
                    T = functions.isodata_threshold(plate_image)
                    bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
                    first_time += 1
                else:
                    T = T - 10
                    bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
                    first_time += 1
                cv2.imshow("bin_plate", bin_plate)
                cv2.waitKey(10)
                intermediate_plate_number = recognize.segment_and_recognize(bin_plate)
            plate_number.append(intermediate_plate_number)

    else:
        plate_number = None

    ## resize : Resizes an image.
    ## params, image, dst_size, interpolation
    ## inter_linar is a bilinear interpolation (used by default)
    ## The function resize resizes the image src down to or up to the specified size. Note that the
    ## initial dst type or size are not taken into account. Instead, the size and type are derived
    ## from the src, dsize, fx, and fy
    return plate_number
#
# def helper(plate_image):
#     resize_factor = 85 / plate_image.shape[0]
#     dim = (int(plate_image.shape[1] * resize_factor), 85)
#     plate_image = cv2.resize(plate_image, dim, interpolation=cv2.INTER_LINEAR)
#     epsilon = 10
#     plate_image = plate_image[epsilon:plate_image.shape[0] - epsilon, epsilon:plate_image.shape[1] - epsilon]
#     plate_image = cv2.GaussianBlur(plate_image, (5, 5), 0)
#     cv2.imshow("plate_image", plate_image)
#     cv2.waitKey(25)
#     first_time = 0
#     intermediate_plate_number = None
#     while (first_time < 5) and (intermediate_plate_number is None):
#         if first_time == 0:
#             T = functions.isodata_threshold(plate_image)
#             bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
#             first_time += 1
#         else:
#             T = T - 10
#             bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
#             first_time += 1
#         cv2.imshow("bin_plate", bin_plate)
#         cv2.waitKey(10)
#         intermediate_plate_number = recognize.segment_and_recognize(bin_plate)
#         return intermediate_plate_number


def start_video(file_path, sample_frequency, output_path):
    cap = cv2.VideoCapture(file_path)
    ## define the fps and speed
    fps = 12
    speed = 0
    ## this is where recognized plates will be added
    recognized_plates = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, speed)

    while True:
        flag, frame = cap.read()

        if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        plates = help_recognize(frame)
        if plates != None:
            for ind in range(len(plates)):
                recognized_plates.append([plates[ind], speed, speed / fps])
        # plates_found.append(do_everything(frame))
        speed += 24
        cap.set(cv2.CAP_PROP_POS_FRAMES, speed)

        data_frame = pd.DataFrame(recognized_plates, columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])

        if output_path is None:
            data_frame.to_csv('record.csv', index=None)
        else:
            data_frame.to_csv(output_path + '/record.csv', index=None)
        # cv2.imshow("original_frame", frame)

    show_plates(recognized_plates)

    cap.release()
    cv2.destroyAllWindows()


def show_plates(plates):
    for plate in plates:
        print(plate)


def do_everything(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    plate = ''

    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            [x, y, w, h] = cv2.boundingRect(cnt)

            if h > 28:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                roi = thresh[y:y + h, x:x + w]
                roismall = cv2.resize(roi, (10, 10))
                # cv2.imshow('normal', frame)
                sample = roismall.reshape((1, 100))

                cv2.imshow("image", roi)
                char = recognize.recognize_template_matching(sample)
                # print(char)

                plate += char
    return plate


# file_path = "/Users/pd/Desktop/License-Plate-Recognition/trainingsvideo.avi"  # "TrainingSet/Categorie III/Video47_2.avi"  #
# capture = cv2.VideoCapture(file_path)
#
# # parameters
# act_frame = 0
# fps = 12
# sample_frequency = 0.5  # frequency for choosing the frames to analyze
#
# # initialization
# ret, frame = capture.read()
# recognized_plates = []
#
# # display image to analyze (each 24 frames)
# while ret:
#     # Show actual frame
#     cv2.imshow('Frame', frame)
#     cv2.waitKey(10)  # gives enough time for image to be displayed
#     mode = 0
#     plates = help_recognize(frame)
#     if plates != None:
#         for ind in range(len(plates)):
#             recognized_plates.append([plates[ind], act_frame, act_frame / fps])
#
#     # Write csv file (using pandas) to keep a record of plate number
#     df = pd.DataFrame(recognized_plates, columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])
#     save_path = 'record.csv'
#     df.to_csv(save_path, index=None)  # 'record.csv'
#
#     act_frame += 24
#     capture.set(cv2.CAP_PROP_POS_FRAMES, act_frame)
#     ret, frame = capture.read()
#
# capture.release()
# cv2.destroyAllWindows()
