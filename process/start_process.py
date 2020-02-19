import cv2
import pandas as pd

from recognise_and_work import localization
from helpers import functions
from recognise_and_work import recognize


def work_on_frame(frame):
    # blur the image to make all colors of the frame uniform
    # reference youtube video and wikihow and articles online

    blurred_frame = cv2.GaussianBlur(frame, (9, 9), 0)
    # hue, saturation, value model, used to select various different colors needed for a particular picture
    frame_hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # find only yellow parts of the image, [10,65,40] upto [35,255,215]
    from_low = (17, 57, 47)
    to_high = (45, 245, 235)
    mask = cv2.inRange(frame_hsv, from_low, to_high)
    m_frame = cv2.bitwise_and(frame, frame, mask=mask)
    # convert hsv to gray

    gray = cv2.cvtColor(m_frame, cv2.COLOR_BGR2GRAY)

    (thresh, binary) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    edges = cv2.Canny(binary, 50, 100)

    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_plates = localization.localise_plates(gray, contours)
    plate_number = []

    if all_plates is not None:
        for plate in all_plates:
            plate = cv2.resize(plate, (int(plate.shape[1] * (85 / plate.shape[0])), 85),
                               interpolation=cv2.INTER_LINEAR)
            plate = plate[10:plate.shape[0] - 10, 10:plate.shape[1] - 10]
            plate = cv2.GaussianBlur(plate, (5, 5), 0)

            # first_time = 0
            intermediate_plate_number = None

            while intermediate_plate_number is None:
                T = functions.isodata_threshold(plate)
                bin_plate = cv2.threshold(plate, T, 255, cv2.THRESH_BINARY_INV)[1]
                intermediate_plate_number = recognize.recognition_segment(bin_plate)
                T -= 5

            # while (first_time < 5) and (intermediate_plate_number is None):
            #     if first_time == 0:
            #         T = functions.isodata_threshold(plate)
            #         bin_plate = cv2.threshold(plate, T, 255, cv2.THRESH_BINARY_INV)[1]
            #         first_time += 1
            #     else:
            #         T = T - 10
            #         bin_plate = cv2.threshold(plate, T, 255, cv2.THRESH_BINARY_INV)[1]
            #         first_time += 1
            #     ("bin_plate", bin_plate)
            #     (10)
            #     intermediate_plate_number = recognize.segment_and_recognize(bin_plate)
            plate_number.append(intermediate_plate_number)

    return plate_number


def start_video(file_path, sample_frequency, output_path):
    cap = cv2.VideoCapture(file_path)
    fps = 12
    speed = 0
    recognized_plates = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, speed)

    while True:
        flag, frame = cap.read()

        if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        plates = work_on_frame(frame)
        if plates is not None:
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
                # ('normal', frame)
                sample = roismall.reshape((1, 100))

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
#     ('Frame', frame)
#     (10)  # gives enough time for image to be displayed
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
