# ueiya_water_prototype_mac_v06.3.1.py

# date created: 2022-01-20
# date modified: 2022-03-16

### FOR USE IN RASPBERRY

# Detects 5 out of 5 digits in water meter (most of the times)

# Description

# Raspberry Picamera, takes a picture, capture to openCV object,
# Analyzes individual sections/digits,
# Uses for loop to iterate digits
# Adds output to data.csv
# Appends data to Google Sheets with gspread
# Pytesseract oem 10 identifies numbers best

### Step 1
## Add number sequence gotten from selectroi.py to 'Crop image' step in the code

### Step 2
## Crop digits manually for 'Dictionary for Sections'. Use image preview on Mac to open images/cropped_image.jpg and get x,y coordinates of digits
## x = h, y = w

### Reference for Dictionary for Sections

# Watermeter physical digits from left to Write

# Camera digits with selectRoi 2022-03-16

## The meter we use has 8 digits. We will remove digit 1 and 2 from the picture, and we're
## measuring the last digit (digit 8) as 0, because we´re not going to track them in this version of the prototype.

# Digit 1 = n/a
# Digit 2 = n/a
# Digit 3 = Section 1, w = 53 to 86 (diff = 33) + 10 pixels difference, add to every section
# Digit 4 = Section 2, w = 96 to 129 
# Digit 5 = Section 3, w = 139 to 172
# Digit 6 = Section 4, w = 182 to 215
# Digit 7 = Section 5, w = 225 to 258
# Digit 8 = n/a

## Camera Resolution

# camera.resolution = (2028, 1520) # Good alternative

# Import packages

import numpy as np
import cv2 as cv
import pytesseract
import csv
import datetime
from picamera import PiCamera
from time import sleep
import re
import gspread

###### Commented for watermeterreadermac.py
# Define Camera
camera = PiCamera()

# Start camera, define ISO and resolution
camera.start_preview()
camera.iso = 800
camera.resolution = (2028, 1520)

# Camera warm-up time
sleep(2)

# Take picture, save and resize
## In your Raspberry Pi, add folder path to images

camera.capture('/home/pi/python/environments/ueiya_env/images/image_0.jpg')
camera.stop_preview()

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
path = '/home/pi/python/environments/ueiya_env/images/image_0.jpg'
img = cv.imread(path)

# Crop image - Add number sequence gotten from selectroi.py to Crop image
imgCropped = img[787:837,839:1109]
cv.imwrite('/home/pi/python/environments/ueiya_env/images/last_image_taken.jpg', imgCropped)

# Define list for output to CSV (this will be used in the last step)
outputList = []

# Dictionary for Sections
sections = {'section1':{'height1': '0', 'height2': '50', 'width1': '53','width2': '83'},\
'section2': {'height1': '0', 'height2': '50', 'width1': '96','width2': '129'},\
'section3': {'height1': '0', 'height2': '50', 'width1': '142','width2': '172'},\
'section4': {'height1': '0', 'height2': '50', 'width1': '192','width2': '215'},\
'section5': {'height1': '0', 'height2': '50', 'width1': '235','width2': '263'}}

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


    # Contours v.6.3.1 (detects 5/5, but v6.2.1 is better)
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

with open('/home/pi/python/environments/ueiya_env/data.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(outputList)

print(outputList)

### Uncomment if to set up Google Sheets with gspread
## # Access Google Sheets with gspread
## 
## gc = gspread.service_account(filename='/home/pi/python/environments/ueiya_env/credentials.json')
## 
## # Open spreadsheet by key
## sh = gc.open_by_key('1SpFfW5fRbZJ-Acs3yePcDEsvWOo88p22K5Y_NP_c26U')
## 
## # Open worksheet
## wks = sh.worksheet("raw-data")
## 
## # Search for a table in the worksheet and append a row to it
## wks.append_rows([outputList], value_input_option='USER-ENTERED', insert_data_option=None, table_range=None)
## #wks.append_rows([outputList]) # Simple append, no extra options

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
