import cv2
import localization
import recognize
import help_functions


def work_on_frame(image):
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    hsv_img = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_img, (10, 65, 40), (35, 255, 215))
    masked_bitwise = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked_bitwise, cv2.COLOR_BGR2GRAY)

    (thresh, binary) = cv2.threshold(gray, 62, 255, cv2.THRESH_BINARY)
    canny_edges = cv2.Canny(binary, 50, 100)

    # opencv 4
    contours, hierarchy = cv2.findContours(canny_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plate = localization.find_plate_in_frame(gray, contours)
    # cv2.imshow('Plate image', plate_image)

    plate = cv2.resize(plate, (int(plate.shape[1] * (85 / plate.shape[0])), 85), interpolation=cv2.INTER_LINEAR)
    # cv2.imshow('Resized plate', plate_image)

    thresh = help_functions.isodata_threshold(plate)

    return recognize.recognition(cv2.threshold(plate, thresh, 255, cv2.THRESH_BINARY_INV)[1])


def start_video(file_path, sample_frequency, output_path):
    cap = cv2.VideoCapture(file_path)
    fps = 12
    param = 0
    plates_found = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, param)

    while True:
        flag, frame = cap.read()

        if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        plates_found.append([work_on_frame(frame), param, param / fps])
        param += 24

        cv2.imshow("image", frame)

    cap.release()
    cv2.destroyAllWindows()


