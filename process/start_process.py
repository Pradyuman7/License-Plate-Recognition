import cv2
import pandas as pd

from helpers import  functions
from recognise_and_work import localization
from recognise_and_work import recognize


def help_recognize(frame):
    blur = cv2.GaussianBlur(frame, (9, 9), 0)

    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    light_orange = (15, 60, 50)
    dark_orange = (37, 255, 220)
    mask = cv2.inRange(hsv_img, light_orange, dark_orange)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("Masked", masked)
    cv2.waitKey(10)

    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    (thresh, binary) = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    edged = cv2.Canny(binary, 50, 100)

    contours = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = localization.plate_detection(gray_original, contours)

    if plates is not None:
        plate_number = []
        for plate_image in plates:
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
                    T = T - 20
                    bin_plate = cv2.threshold(plate_image, T, 255, cv2.THRESH_BINARY_INV)[1]
                    first_time += 1
                cv2.imshow("bin_plate", bin_plate)
                cv2.waitKey(10)
                intermediate_plate_number = recognize.segment_and_recognize(bin_plate)
            plate_number.append(intermediate_plate_number)

    else:
        plate_number = None

    return plate_number


file_path = "/Users/pd/Desktop/License-Plate-Recognition/trainingsvideo.avi"  # "TrainingSet/Categorie III/Video47_2.avi"  #
capture = cv2.VideoCapture(file_path)

# parameters
act_frame = 0
fps = 12
sample_frequency = 0.5  # frequency for choosing the frames to analyze

# initialization
ret, frame = capture.read()
recognized_plates = []

# display image to analyze (each 24 frames)
while ret:
    # Show actual frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(10)  # gives enough time for image to be displayed
    mode = 0
    plates = help_recognize(frame)
    if plates != None:
        for ind in range(len(plates)):
            recognized_plates.append([plates[ind], act_frame, act_frame / fps])

    # Write csv file (using pandas) to keep a record of plate number
    df = pd.DataFrame(recognized_plates, columns=['License plate', 'Frame no.', 'Timestamp(seconds)'])
    save_path = 'record.csv'
    df.to_csv(save_path, index=None)  # 'record.csv'

    act_frame += 24
    capture.set(cv2.CAP_PROP_POS_FRAMES, act_frame)
    ret, frame = capture.read()

capture.release()
cv2.destroyAllWindows()
