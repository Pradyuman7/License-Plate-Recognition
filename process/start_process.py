import cv2
from recognise_and_work import recognize, localization
from helpers import functions
import pandas as pd


def work_on_frame(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (10, 65, 40), (35, 255, 215))
    bit_mast = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(bit_mast, cv2.COLOR_BGR2GRAY)

    (thresh, binary) = cv2.threshold(gray, 62, 255, cv2.THRESH_BINARY)
    canny_edges = cv2.Canny(binary, 50, 100)

    # opencv 4 has two in return
    contours, _ = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate = localization.find_plate_in_frame(gray, contours)

    if plate is None:
        return

    plate = cv2.resize(plate, (int(plate.shape[1] * (85 / plate.shape[0])), 85), interpolation=cv2.INTER_LINEAR)

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
