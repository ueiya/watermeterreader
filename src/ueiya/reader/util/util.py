import re
from typing import Any, List, Tuple

import cv2 as cv
import numpy as np
import pytesseract


def stack_images(scale: float, imgArray: Tuple[List[Any]]):
    """

    :param scale:
    :param imgArray:
    :return:
    """
    # TODO fix Any in type annotation for imgArray
    rows = len(imgArray)
    cols = len(imgArray[0])
    rows_available = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y], (0, 0), None, scale, scale
                    )
                else:
                    imgArray[x][y] = cv.resize(
                        imgArray[x][y],
                        (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv.cvtColor(imgArray[x][y], cv.COLOR_GRAY2BGR)

        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        hor_con = [image_blank] * rows

        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(
                    imgArray[x],
                    (imgArray[0].shape[1], imgArray[0].shape[0]),
                    None,
                    scale,
                    scale,
                )
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def process_section(iterations, crop_size, img_cropped, output: List):
    print(iterations)

    img_crop_section = img_cropped[
        crop_size["height1"] : crop_size["height2"],
        crop_size["width1"] : crop_size["width2"],
    ]
    # Gray and blur
    gray = cv.cvtColor(img_crop_section, cv.COLOR_BGR2GRAY)

    # v2

    # thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

    # Resize inputs
    scale_percent = 300  # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    resize = cv.resize(gray, dim, interpolation=cv.INTER_AREA)

    # Blur image
    blur = cv.GaussianBlur(resize, (5, 5), 0)

    # Equalize Histogram
    equalize = cv.equalizeHist(blur)

    # Adaptive Thresholding
    # thresh = cv.adaptiveThreshold(equalize,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    # edges = cv.Canny(thresh,100,200)

    # Otsu thresholding
    # blur = cv.GaussianBlur(equalize,(5,5),0)
    # ret,thresh = cv.threshold(equalize,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Morphological tranformations
    kernel = np.ones((5, 5), np.uint8)
    # erosion = cv.erode(equalize,kernel,iterations = 2)
    dilation = cv.dilate(equalize, kernel, iterations=1)
    # opening = cv.morphologyEx(equalize, cv.MORPH_OPEN, kernel)
    # closing = cv.morphologyEx(equalize, cv.MORPH_CLOSE, kernel)

    # Adaptive Thresholding v6.2.1.
    thresh = cv.adaptiveThreshold(
        dilation, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 41, 11
    )

    # Padding (make borders for image)
    # white = [255,255,255]
    # padding = cv.copyMakeBorder(thresh,20,20,20,20,cv.BORDER_CONSTANT,value=white)

    # Draw rectangle
    rect_height, rect_width = resize.shape  # gets the size of the resized image
    cv.rectangle(
        thresh, (0, 0), (rect_width, rect_height), (255, 255, 255), 15
    )  # 255,255,255 is white

    # Morphological tranformations v.6.2.1
    # closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    # erosion = cv.erode(closing,kernel,iterations = 1)

    # Morphological tranformations v.6.3.1
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    erosion = cv.erode(closing, kernel, iterations=2)

    # Blur to b/w image
    blur2 = cv.GaussianBlur(erosion, (5, 5), 10)
    blur3 = cv.GaussianBlur(blur2, (5, 5), 10)
    blur4 = cv.GaussianBlur(blur3, (5, 5), 10)
    blur5 = cv.GaussianBlur(blur4, (5, 5), 10)

    # Edge detection
    edges = cv.Canny(blur5, 200, 200)

    # Invert image
    # invert = cv.bitwise_not(edges)

    # Contours
    contours, hierarchy = cv.findContours(
        edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
    )

    # Contours v6.2.1. (detects 5/5)
    contours_dict = dict()
    for cont in contours:
        x, y, w, h = cv.boundingRect(cont)
        area = cv.contourArea(cont)
        if 10 < area and 10 < w and h > 5:
            contours_dict[(x, y, w, h)] = cont

    # Contours v.6.3.1 (detects 5/5, keep v6.2.1)
    # contours_dict = dict()
    # for cont in contours:
    #    x, y, w, h = cv.boundingRect(cont)
    #    area = cv.contourArea(cont)
    #    if 30 < area and 30 < w and h > 15:
    #        contours_dict[(x, y, w, h)] = cont

    contours_filtered = sorted(contours_dict.values(), key=cv.boundingRect)

    blank_background = np.zeros_like(edges)

    # img_contours = cv.drawContours(blur5, contours_filtered, -1, (255,255,255), thickness=2) #white
    img_contours = cv.drawContours(
        blur5, contours_filtered, -1, (0, 0, 0), thickness=2
    )  # black

    # Resize image 2
    # resize2 = cv.resize(img_contours, dim, interpolation = cv.INTER_AREA)

    # Invert image (the one with Contours)
    # invert_contours = cv.bitwise_not(img_contours)

    # Pytesseract text recognition
    digit = pytesseract.image_to_string(
        blur5, config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
    )
    # Strip Values
    stripValues = digit.strip()
    print("List Value is:", stripValues)
    # Find Numbers
    findNumber = re.findall("[0-9]+", stripValues)
    print(findNumber)
    # Get number from list
    # getNumber = findNumber[0]
    if len(findNumber) == 1:
        getNumber = findNumber[0]
    else:
        getNumber = 100  # I pass 100 and later in data cleaning I filter out this value, as digits can only be 0-9
    # Append to CSV
    output.append(int(getNumber))
    print("- Image Recognition. The water meter number is:", getNumber)
    # Show the images / Stacked
    imgBlank = np.zeros_like(resize)
    imgStack = stack_images(
        0.8,
        (
            [equalize],
            [dilation],
            [thresh],
            [closing],
            [erosion],
            [blur2],
            [blur3],
            [blur4],
            [blur5],
            [edges],
            [img_contours],
        ),
    )
    # imgStack2 = stackImages(0.8,([blur5],[edges],[img_contours]))
    cv.imshow("Stack", imgStack)
    cv.waitKey(0)
    cv.destroyAllWindows()
