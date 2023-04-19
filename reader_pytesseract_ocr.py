import cv2
import numpy as np
import pytesseract
from loguru import logger as logging


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image, lower=0, higher=255):
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    return cv2.threshold(image, lower, higher, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# closing - dilation followed by erosion
def closing(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


pytesseract.pytesseract.tesseract_cmd = r"E:\Ueiya\Tesseract-OCR\tesseract.exe"

# read full image
full_image = cv2.imread("image_0.jpg")

# crop the region where meter reading are present
crop_img = full_image[288 * 2 : 354 * 2, 388 * 2 : 685 * 2]

crop_h, crop_w = crop_img.shape[:2]
len_of_chars = 8

# divide the crop region into subregions of size of len of characters
each_char_w = int(crop_w / len_of_chars)

meter_prediction = ""

# loop through each character
for i in range(len_of_chars):
    each_char = crop_img[:, i * each_char_w : (i + 1) * each_char_w, :]

    gray = get_grayscale(each_char)

    # all pixels with value less than threshold are set to 0
    gray[gray < 5] = 0

    predicted_val = pytesseract.image_to_string(
        gray, config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
    )

    # remove trailing '\n'
    predicted_val = predicted_val.rstrip()

    if predicted_val == "":
        predicted_val = "X"

    meter_prediction += predicted_val

logging.info(f"Predicted meter reading is : {meter_prediction}")
