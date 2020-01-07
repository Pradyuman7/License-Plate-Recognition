import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "/usr/local/Cellar/tesseract/4.1.1/bin/tesseract"


# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)


# Load image, create mask, grayscale, Otsu's threshold
def find(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Filter for ROI using contour area and aspect ratio
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)

        if area > 2000 and aspect_ratio > .5:
            mask[y:y + h, x:x + w] = image[y:y + h, x:x + w]

    # Perfrom OCR with Pytesseract
    data = pytesseract.image_to_string(mask, lang='eng', config='--psm 6')
    data = (data.encode('ascii', 'ignore')).decode("utf-8")
    data = data.translate({ord(i):None for i in '!@#$'})
    print(data)
