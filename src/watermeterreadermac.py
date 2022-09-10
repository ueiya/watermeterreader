# ueiya_water_prototype_mac_v06.3.1.py

# date created: 2022-01-20
# date modified: 2022-02-08

### FOR USE IN MAC TO TEST, NOT RASPBERRY

# Detects 5 out of 5 digits in water meter (sometimes)

# Description

# Raspberry Picamera, takes a picture, capture to openCV object,
# Analyzes individual sections/digits,
# Uses for loop to iterate digits
# Adds output to data.csv
# Appends data to Google Sheets with gspread
# Pytesseract oem 10 identifies numbers best

    # Possible alternative/next steps, suggested by Richard Gould https://github.com/rgould

    # A HTTP API running on your laptop, listening for HTTP requests (ie. http://192.168.1.101/send_image)
    # after taking a picture, send image to above URL
    # ipconfig getifaddr en0
    # http://192.168.2.104:8080/send_image

### Crop Sections Dictionary

# Watermeter physical digits from left to Write


# 1 # To get camera digits
# >>>>> USE CODE withe SelectROI() method in OPENCV in your Mac o PC

# Camera digits with selectRoi 2022-02-01

# Digit 1 = n/a
# Digit 2 = n/a
# Digit 3 = Section 1, w = 45 to 70 (diff 25)
# Digit 4 = Section 2, w = 85 to 110 (diff 25)
# Digit 5 = Section 3, w = 125 to 150
# Digit 6 = Section 4, w = 170 to 195
# Digit 7 = Section 5, w = 210 to 235
# Digit 8 = n/a

## Camera Digits reference table
## The meter we use, uses 8 digits. We will remove digit 1 and 2 from the picture, because we´re not going to track them in this version of the prototype.

# Digit 1 = n/a
# Digit 2 = n/a
# Digit 3 = Section 1, w = 235 to 340
# Digit 4 = Section 2, w = 340 to 455
# Digit 5 = Section 3, w = 455 to 570
# Digit 6 = Section 4, w = 570 to 685
# Digit 7 = Section 5, w = 685 to 789
# Digit 8 = n/a

#sections = {'section1': {'height1': '0', 'height2': '150', 'width1': '230','width2': '305'},'section2': {'height1': '0', 'height2': '150', 'width1': '340','width2': '420'},'section3': {'height1': '0', 'height2': '150', 'width1': '445','width2': '530'},'section4': {'height1': '0', 'height2': '150', 'width1': '570','width2': '653'},'section5': {'height1': '0', 'height2': '150', 'width1': '680','width2': '760'}}


## Camera Resolution

# Max Resolution # Error: not enough resources

# Width: 4056
# Heigth: 3040

# camera.resolution = (2028, 1520) # Good alternative


# Import packages

import numpy as np
import cv2 as cv
import pytesseract
import csv
import datetime
#from picamera import PiCamera
from time import sleep
import re
#import gspread

##### # Define Camera
##### camera = PiCamera()
#####
##### # Start camera, define ISO and resolution
##### camera.start_preview()
##### camera.iso = 800
##### camera.resolution = (2028, 1520)
#####
##### # Camera warm-up time
##### sleep(2)
#####
##### # Take picture, save and resize
##### ## In your Raspberry Pi, add folder path to images
#####
##### # v1
##### #camera.capture('/home/pi/python/environments/ueiya_env/images/image_0.jpg', resize=(4056, 3040))
#####
##### # v2
##### camera.capture('/home/pi/python/environments/ueiya_env/images/image_0.jpg')
##### camera.stop_preview()

# Stack images Function
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv.cvtColor( imgArray[x][y], cv.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv.cvtColor(imgArray[x], cv.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

# Accesing image
path = '/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/v2_test_photos/2022-02-15_image_0.jpg'
img = cv.imread(path)

# Crop image v1
imgCropped = img[675:723,878:1114]
cv.imwrite('/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/images/last_image_taken.jpg', imgCropped)

# Define list for output to CSV (this will be used in the last step)
outputList = []

# Dictionary for Sections
sections = {'section1':{'height1': '0', 'height2': '49', 'width1': '45','width2': '70'},\
'section2': {'height1': '0', 'height2': '49', 'width1': '85','width2': '110'},\
'section3': {'height1': '0', 'height2': '49', 'width1': '125','width2': '150'},\
'section4': {'height1': '0', 'height2': '49', 'width1': '170','width2': '195'},\
'section5': {'height1': '0', 'height2': '49', 'width1': '210','width2': '235'}}

# Loop through dictionary for Sections

for iterations, cropSize in sections.items():
    print(iterations)
    h1 = cropSize['height1']
    h2 = cropSize['height2']
    w1 = cropSize['width1']
    w2 = cropSize['width2']
    # print(h1,h2,w1,w2)
    # Crop section using dictionary values
    imgCropSection = imgCropped[int(h1):int(h2),int(w1):int(w2)]
    # Gray and blur
    gray = cv.cvtColor(imgCropSection, cv.COLOR_BGR2GRAY)

    # v2

    #thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)

    # Resize inputs
    scale_percent = 300 # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)

    # Resize image
    resize = cv.resize(gray, dim, interpolation = cv.INTER_AREA)

    # Blur image
    blur = cv.GaussianBlur(resize,(5,5),0)

    # Equalize Histogram
    equalize = cv.equalizeHist(blur)

    # Adaptive Thresholding
    #thresh = cv.adaptiveThreshold(equalize,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #edges = cv.Canny(thresh,100,200)

    # Otsu thresholding
    #blur = cv.GaussianBlur(equalize,(5,5),0)
    #ret,thresh = cv.threshold(equalize,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    # Morphological tranformations
    kernel = np.ones((5,5),np.uint8)
    #erosion = cv.erode(equalize,kernel,iterations = 2)
    dilation = cv.dilate(equalize,kernel,iterations = 1)
    #opening = cv.morphologyEx(equalize, cv.MORPH_OPEN, kernel)
    #closing = cv.morphologyEx(equalize, cv.MORPH_CLOSE, kernel)

    # Adaptive Thresholding v6.2.1.
    thresh = cv.adaptiveThreshold(dilation,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,41,11)

    # Padding (make borders for image)
    #white = [255,255,255]
    #padding = cv.copyMakeBorder(thresh,20,20,20,20,cv.BORDER_CONSTANT,value=white)

    # Draw rectangle
    rect_height, rect_width = resize.shape # gets the size of the resized image
    cv.rectangle(thresh,(0,0),(rect_width, rect_height),(255,255,255),15) # 255,255,255 is white

    # Morphological tranformations v.6.2.1
    #closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    #erosion = cv.erode(closing,kernel,iterations = 1)

    # Morphological tranformations v.6.3.1
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    erosion = cv.erode(closing,kernel,iterations = 2)

    # Blur to b/w image
    blur2 = cv.GaussianBlur(erosion,(5,5),10)
    blur3 = cv.GaussianBlur(blur2,(5,5),10)
    blur4 = cv.GaussianBlur(blur3,(5,5),10)
    blur5 = cv.GaussianBlur(blur4,(5,5),10)

    # Edge detection
    edges = cv.Canny(blur5,200,200)

    # Invert image
    # invert = cv.bitwise_not(edges)

    # Contours
    contours, hierarchy = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #Contours v6.2.1. (detects 5/5)
    contours_dict = dict()
    for cont in contours:
        x, y, w, h = cv.boundingRect(cont)
        area = cv.contourArea(cont)
        if 10 < area and 10 < w and h > 5:
            contours_dict[(x, y, w, h)] = cont


    # Contours v.6.3.1 (detects 5/5, keep v6.2.1)
    #contours_dict = dict()
    #for cont in contours:
    #    x, y, w, h = cv.boundingRect(cont)
    #    area = cv.contourArea(cont)
    #    if 30 < area and 30 < w and h > 15:
    #        contours_dict[(x, y, w, h)] = cont

    contours_filtered = sorted(contours_dict.values(), key=cv.boundingRect)

    blank_background = np.zeros_like(edges)

    #img_contours = cv.drawContours(blur5, contours_filtered, -1, (255,255,255), thickness=2) #white
    img_contours = cv.drawContours(blur5, contours_filtered, -1, (0,0,0), thickness=2) #black

    # Resize image 2
    #resize2 = cv.resize(img_contours, dim, interpolation = cv.INTER_AREA)

    # Invert image (the one with Contours)
    #invert_contours = cv.bitwise_not(img_contours)

    # Pytesseract text recognition
    digit = pytesseract.image_to_string(blur5, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    # Strip Values
    stripValues = digit.strip()
    print("List Value is:", stripValues)
    # Find Numbers
    findNumber = re.findall('[0-9]+', stripValues)
    print(findNumber)
    # Get number from list
    #getNumber = findNumber[0]
    if len(findNumber) == 1:
        getNumber = findNumber[0]
    else:
        getNumber = 100 # I pass 100 and later in data cleaning I filter out this value, as digits can only be 0-9
    # Append to CSV
    outputList.append(int(getNumber))
    print("- Image Recognition. The water meter number is:", getNumber)
    # Show the images / Stacked
    imgBlank = np.zeros_like(resize)
    imgStack = stackImages(0.8,([equalize],[dilation],[thresh],[closing],[erosion],[blur2],[blur3],[blur4],[blur5],[edges],[img_contours]))
    #imgStack2 = stackImages(0.8,([blur5],[edges],[img_contours]))
    cv.imshow("Stack", imgStack)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Add last digit to output list. Digit is 0.

lastDigit = 0
outputList.append(lastDigit)

# Open CSV and append outputList in different columns (each value is separated by a comma / creo)

timestamp = datetime.datetime.now() # create a timestamp variable
outputList.append(str(timestamp))

with open('/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/data.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(outputList)

print(outputList)

# Access Google Sheets with gspread

gc = gspread.service_account(filename='/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/credentials.json')

# Open spreadsheet by key
sh = gc.open_by_key('1SpFfW5fRbZJ-Acs3yePcDEsvWOo88p22K5Y_NP_c26U')

# Open worksheet
wks = sh.worksheet("raw-data")

# Search for a table in the worksheet and append a row to it
wks.append_rows([outputList], value_input_option='USER-ENTERED', insert_data_option=None, table_range=None)
#wks.append_rows([outputList]) # Simple append, no extra options


## PyTesseract Documentation

#   Page segmentation modes:
#     0    Orientation and script detection (OSD) only.
#     1    Automatic page segmentation with OSD.
#     2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
#     3    Fully automatic page segmentation, but no OSD. (Default)
#     4    Assume a single column of text of variable sizes.
#     5    Assume a single uniform block of vertically aligned text.
#     6    Assume a single uniform block of text.
#     7    Treat the image as a single text line.
#     8    Treat the image as a single word.
#     9    Treat the image as a single word in a circle.
#    10    Treat the image as a single character.
#    11    Sparse text. Find as much text as possible in no particular order.
#    12    Sparse text with OSD.
#    13    Raw line. Treat the image as a single text line,
#          bypassing hacks that are Tesseract-specific.
#
#   OCR Engine modes:
#     0    Legacy engine only.
#     1    Neural nets LSTM engine only.
#     2    Legacy + LSTM engines.
#     3    Default, based on what is available.
