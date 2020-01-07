import cv2


def runVideo(file_path, sample_frequency, output_path):
    cap = cv2.VideoCapture(file_path)
    number = 0
    count = 0

    while True:
        flag, frame = cap.read()
        plateNumber = ""

        if (not flag) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Noise removal with iterative bilateral filter(removes noise while preserving edges)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        # cv2.imshow("2 - Bilateral Filter", gray)

        # Find Edges of the grayscale image
        edged = cv2.Canny(gray, 170, 200)
        # cv2.imshow("4 - Canny Edges", edged)

        # Find contours based on Edges
        (cnts, thresh) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
        # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
        NumberPlateCnt = 0  # we currently have no Number plate contour

        # loop over our contours to find the best possible approximate contour of number plate
        count = 0
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:  # Select the contour with 4 corners
                NumberPlateCnt = approx  # This is our approx Number Plate Contour
                break

        # Drawing the selected contour on the original image
        cv2.drawContours(frame, cnts, NumberPlateCnt, (0, 255, 0), 3)
        cv2.imshow("Final Image", frame)

    cap.release()
    cv2.destroyAllWindows()