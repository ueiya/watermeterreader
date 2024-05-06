import cv2
import easyocr
import numpy as np
from loguru import logger as logging
import glob
import os
import detect_roi


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


easyocr_reader = easyocr.Reader(["en"])

images_path = "E:/Ueiya/Images/images_2023-08-01_to_2023-08-08/*.jpg"

for image_path in glob.glob(images_path):

    _, image_name = os.path.split(image_path)

    # read full image
    full_image = cv2.imread(image_path)

    bbox_roi = detect_roi.run(weights="best.pt", source=image_path, device='cpu')

    x1 = int(bbox_roi[0].item())
    y1 = int(bbox_roi[1].item())
    x2 = int(bbox_roi[2].item())
    y2 = int(bbox_roi[3].item())

    # crop the region where meter reading are present
    crop_img = full_image[y1:y2, x1:x2]

    gray = get_grayscale(crop_img)

    # all pixels with value less than threshold are set to 0
    gray[gray < 5] = 0

    crop_h, crop_w = gray.shape[:2]
    len_of_chars = 8

    # divide the crop region into subregions of size of len of characters
    each_char_w = int(crop_w / len_of_chars)

    characters = []

    # loop through each character
    for i in range(len_of_chars):
        each_char = gray[:, i * each_char_w : (i + 1) * each_char_w]

        invert = cv2.bitwise_not(each_char)

        detection = easyocr_reader.readtext(invert, allowlist="0123456789")
        if len(detection) == 0:
            character = 'X'
        else:
            character = detection[0][1]
            if len(character) > 1:
                character = character[1]
            elif len(character) == 1:
                character = character[0]

        characters.append(character)
    logging.debug(f"Image name : {image_name}, prediction : {''.join(characters)}")

    cv2.imshow("Current image", crop_img)
    cv2.waitKey(0)
