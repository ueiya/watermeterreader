# watermeterreadermac-tm.py

# date created: 2022-01-20
# date modified: 2022-03-23

### FOR USE IN MAC TO TEST, NOT RASPBERRY

# Detects 5 out of 5 digits in water meter (most of the times)

# Description

# Raspberry Picamera, takes a picture, capture to openCV object,
# Analyzes individual sections/digits,
# Uses for loop to iterate digits
# Adds output to data.csv
# Appends data to Google Sheets with gspread
# OpenCV template matching identifies numbers

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
#from picamera import PiCamera
from time import sleep
import re
#import gspread

######## TEMPLATE MATCHING TEST

import imutils
from imutils import contours
from matplotlib import pyplot as plt

######## TEMPLATE MATCHING TEST

###### Commented for watermeterreadermac.py
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
##### camera.capture('/home/pi/python/environments/ueiya_env/images/image_0.jpg')
##### camera.stop_preview()

######## TEMPLATE MATCHING TEST
 
# Custom display function
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)

n = 'test'

# Read in template map
path_template = '/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/ueiya_gimp/ocr-ueiya-chalkboard-white.jpg'
img_template = cv.imread(path_template)

# Template conversion to grayscale image
ref = cv.cvtColor(img_template, cv.COLOR_BGR2GRAY)
# cv_show(n, ref)

#Convert to binary graph, turn the digital part into white
#, function multiple return values are tuples, here take the second return value
ref = cv.threshold(ref, 10, 255, cv.THRESH_BINARY_INV)[1]
#cv_show(n, ref)

## Detecting Contours

## In this step, we find the contours present in the pre-processed image and then store the returned contour information. Next, we sort the contours from left-to-right as well as initialize a dictionary, digits, which map the digit name to the region of interest.
    
refCnts = cv.findContours(ref.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# A single contour is extracted into a dictionary
    # Copy the outline in the template
    # changed to contour of the same size
    # at this time, the outline corresponding to the dictionary key is the corresponding number. For example, key '1' corresponds to contour '1'

for (i, c) in enumerate(refCnts):
    (x, y, w, h) = cv.boundingRect(c)
    roi = ref[y:y + h, x:x + w] 
    roi = cv.resize(roi,(57,88)) 
    digits[i] = roi 

groupOutput = []

######## TEMPLATE MATCHING TEST

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
path = '/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/v2_test_photos/2022-04-04_1551_calibrate.jpg'
img = cv.imread(path)

# Crop image - Add number sequence gotten from selectroi.py to Crop image
imgCropped = img[782:830,847:1112]
cv.imwrite('/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/images/last_image_taken.jpg', imgCropped)

# Define list for output to CSV (this will be used in the last step)
outputList = []
outputListScores = []

# Dictionary for Sections
sections = {'section1':{'height1': '0', 'height2': '50', 'width1': '49','width2': '76'},\
'section2': {'height1': '0', 'height2': '50', 'width1': '95','width2': '125'},\
'section3': {'height1': '0', 'height2': '50', 'width1': '140','width2': '168'},\
'section4': {'height1': '0', 'height2': '50', 'width1': '184','width2': '213'},\
'section5': {'height1': '0', 'height2': '50', 'width1': '230','width2': '260'}}

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

    ######## TEMPLATE MATCHING TEST

    # Path to image template
    invert_image = np.invert(closing)
    #invert_image = np.invert(blur5)

    template = invert_image
    cv_show(n, invert_image)
    
    # Calculate match score:  What's the score of 0 , What's the score of 1 ...

    scores = [] # in a single cycle, scores stores the maximum score of one value matching 10 template values

    # Calculate each score in the template

    # The digit of digits is exactly the value 0,1,..., 9; digitroi is the characteristic representation of each value for (digit, digitROI) in digits.items():

    for (digit, digitROI) in digits.items():

        # For template matching, res is the result matrix
        # In this case, ROI is x, digitroi is 0 and then 1 , two ..  Match 10 times , What's the highest score of the template
        # res, Return 4 , Take the second maximum maxscore
        # max score, 10 maximum values
        res = cv.matchTemplate(template, digitROI, cv.TM_CCOEFF)
        max_score = cv.minMaxLoc(res)[1]
        scores.append(max_score)
        #print("scores：",scores)


    # Get the most appropriate number
    digit_score = float(np.max(scores))
    print("max score：",str(np.max(scores)))

    # Returns the position of the maximum value in the input list
    getNumber = str(np.argmax(scores))
    groupOutput.append(str(np.argmax(scores)))
    
    print("template matching output：",groupOutput)

    ######## TEMPLATE MATCHING TEST

    # # Pytesseract text recognition
    # digit = pytesseract.image_to_string(blur5, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    # 
    # # Strip Values
    # stripValues = digit.strip()
    # print("List Value is:", stripValues)
    # 
    # # Find Numbers
    # findNumber = re.findall('[0-9]+', stripValues)
    # print(findNumber)
    # 
    # # Get number from list
    # #getNumber = findNumber[0]
    # if len(findNumber) == 1:
    #     getNumber = findNumber[0]
    # else:
    #     getNumber = 100 # I pass 100 and later in data cleaning I filter out this value, as digits can only be 0-9
    
    # Append to OutputList
    outputList.append(int(getNumber))
    outputListScores.append(float(digit_score))

    # print("- Image Recognition. The water meter number is:", getNumber)
    # 
    # # Show the images / Stacked
    # imgBlank = np.zeros_like(resize)
    # imgStack = stackImages(0.8,([equalize],[dilation],[thresh],[closing],[erosion],[blur2],[blur3],[blur4],[blur5],[edges],[img_contours]))
    # 
    # #imgStack2 = stackImages(0.8,([blur5],[edges],[img_contours]))
    # cv.imshow("Stack", imgStack)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

# Add last digit to output list. Digit is 0.

lastDigit = 0
outputList.append(lastDigit)

# Open CSV and append outputList in different columns

timestamp = datetime.datetime.now() # create a timestamp variable
outputList.append(str(timestamp))

finalList = outputList + outputListScores

with open('/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/data-v2.10.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(finalList)

print(outputList, outputListScores)

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
