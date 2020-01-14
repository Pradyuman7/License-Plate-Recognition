import cv2
from recognise_and_work import recognize, localization
from helpers import functions
import pandas as pd


def work_on_frame(image):
    # blur the image to make all colors of the frame uniform
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    # hue, saturation, value model, used to select various different colors needed for a particular picture
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # GaussianBlur :

    # find only yellow parts of the image, [10,65,40] upto [35,355,215]
    mask = cv2.inRange(hsv, (10, 65, 40), (35, 255, 215))
    bit_mast = cv2.bitwise_and(image, image, mask=mask)

    # bitwise_and : calculates the per-element bit-wise logical conjunction
    # Two arrays when src1 and src2 have the same size:
    # dst[i] = scr1[i] ^ scr2[i] if mask[i] != 0

    # inRange : Checks if array elements lie between the elements of two other arrays.
    # params, src is first input array, lower-bound inclusive, upper bound inclusive
    # return dst where dst[i[ = low[i] <= src[i] <= up[i]

    # convert hsv to gray
    gray = cv2.cvtColor(bit_mast, cv2.COLOR_BGR2GRAY)

    # binarisation of the image
    (thresh, binary) = cv2.threshold(gray, 62, 255, cv2.THRESH_BINARY)

    # threshold : Applies a fixed-level threshold to each array element.
    # params, src, threshold value, max value
    # for thresh_binary :
    # dst(x,y) = max_val if src(x,y) >  thresh else 0

    # edge detection using canny algorithm
    canny_edges = cv2.Canny(binary, 50, 100)

    # Canny : Finds edges in an image using the [Canny86] algorithm.
    # params, image, thesh_1, thresh_2
    # The smallest value between threshold1 and threshold2 is used for edge linking.
    # The largest value is used to find initial segments of strong edges.
    # canny algorithm works as following :
    # Apply Gaussian filter to smooth the image in order to remove the noise
    # Find the intensity gradients of the image (An image gradient is a
    # directional change in the intensity or color in an image)
    # Apply non-maximum suppression to get rid of spurious response to edge detection
    # NMS : transform a smooth response map that triggers many imprecise object
    # window hypotheses in, ideally, a single bounding-box for each detected object.
    # Apply double threshold to determine potential edges
    # Track edge by hysteresis: Finalize the detection of edges by suppressing all the other edges
    # that are weak and not connected to strong edges.
    # hysterisis: hysteresis compares two images to build an intermediate image. The function
    # takes two binary images that have been thresholded at different levels. The higher threshold
    # has a smaller population of white pixels. The values in the higher threshold are more likely to be real edges.

    # find contours of the image
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # findContours : Finds contours in a binary image.
    # params, image, mode, method
    # retr_tree retrieves all of the contours and reconstructs a full hierarchy of nested contours
    # chain_approx_none stores absolutely all the contour points. That is, any 2 subsequent points
    # (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors,
    # that is, max(abs(x1-x2),abs(y2-y1))==1.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate = localization.localise_plates(gray, contours)

    if plate is None:
        return

    plate = cv2.resize(plate, (int(plate.shape[1] * (85 / plate.shape[0])), 85), interpolation=cv2.INTER_LINEAR)

    # resize : Resizes an image.
    # params, image, dst_size, interpolation
    # inter_linar is a bilinear interpolation (used by default)
    # The function resize resizes the image src down to or up to the specified size. Note that the
    # initial dst type or size are not taken into account. Instead, the size and type are derived
    # from the src, dsize, fx, and fy .

    return recognize.recognition(cv2.threshold(plate, functions.isodata_threshold(plate), 255, cv2.THRESH_BINARY_INV)[1])


def start_video(file_path, sample_frequency, output_path):
    cap = cv2.VideoCapture(file_path)
    fps = 12
    speed = 0
    plates_found = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, speed)

    while True:
        flag, frame = cap.read()

        if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        plates_found.append([work_on_frame(frame), speed, speed / fps])
        # plates_found.append(do_everything(frame))
        speed += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, speed)

        data_frame = pd.DataFrame(plates_found, columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])

        if output_path is None:
            data_frame.to_csv('record.csv', index=None)
        else:
            data_frame.to_csv(output_path + '/record.csv', index=None)
        # cv2.imshow("original_frame", frame)

    show_plates(plates_found)

    cap.release()
    cv2.destroyAllWindows()


def show_plates(plates):
    for plate in plates:
        print(plate)


# def do_everything(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
#
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     plate = ''
#
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 50:
#             [x, y, w, h] = cv2.boundingRect(cnt)
#
#             if h > 28:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
#                 roi = thresh[y:y + h, x:x + w]
#                 roismall = cv2.resize(roi, (10, 10))
#                 # cv2.imshow('normal', frame)
#                 sample = roismall.reshape((1, 100))
#
#                 cv2.imshow("image", roi)
#                 char = recognize.recognize_template_matching(sample)
#                 # print(char)
#
#                 plate += char
#
#     return plate
