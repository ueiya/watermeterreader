# watermeterreader-tm.py

# Description

# Raspberry Picamera, takes a picture, capture to openCV object,
# Analyzes individual sections/digits,
# Uses for loop to iterate digits
# Adds output to data.csv
# OpenCV template matching identifies numbers
# Appends data to Google Sheets with gspread

### Step 1
## Add number sequence gotten from selectroi.py to 'Crop image' step in the code

### Step 2
## Use selectroi.py get x,y coordinates of digits and add manually to 'Dictionary for Sections'. 
## x = h, y = w

### Reference for Dictionary for Sections

# Watermeter physical digits from left to Write

# Camera digits with selectRoi 2022-03-16

## The meter we use has 8 digits. We will ignore digit 1 and 2 from the picture, and we're
## measuring the last digit (digit 8) as 0, because we´re not going to track them in this version of the prototype.

# Digit 1 = n/a
# Digit 2 = n/a
# Digit 3 = Section 1, w = 53 to 86 (diff = 33) + 10 pixels difference, add to every section
# Digit 4 = Section 2, w = 96 to 129 
# Digit 5 = Section 3, w = 139 to 172
# Digit 6 = Section 4, w = 182 to 215
# Digit 7 = Section 5, w = 225 to 258
# Digit 8 = n/a

# Import packages

import numpy as np
import cv2 as cv
import csv
import datetime
#from picamera import PiCamera
from time import sleep
import re
#import gspread
import random
import os.path

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
##### camera.capture('images/image_0.jpg')
##### camera.stop_preview()

######## TEMPLATE MATCHING TEST
 
# Custom display function
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)

n = 'image preview'

# Read in template map
path_template = 'images/ocr-ueiya-manual-white-02.jpg'
img_template = cv.imread(path_template)

# Template conversion to grayscale image
ref = cv.cvtColor(img_template, cv.COLOR_BGR2GRAY)
#cv_show(n, ref)

# Convert to binary graph, turn the digital part into white
# function multiple return values are tuples, here take the second return value
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
path = 'images/image_0.jpg'
img = cv.imread(path)

# Rotate image - add '0' in rotate_angle if no rotation is necessary
(h, w) = img.shape[:2]
rotate_center = (w / 2, h / 2)
rotate_angle = -2
rotate_scale = 1

rotate_matrix = cv.getRotationMatrix2D(rotate_center, rotate_angle, rotate_scale)
imgRotated = cv.warpAffine(img, rotate_matrix, (w, h))
#cv_show(n, imgRotated)

# Crop image - Add number sequence gotten from selectroi.py to Crop image
#imgCropped = imgRotated[763:813,845:1112] # 6 digits/sections
imgCropped = imgRotated[760:814,895:1114] # 5 digits/sections

cv.imwrite('images/last_image_taken.jpg', imgCropped)
#cv_show(n, imgCropped)

# Define list for output to CSV (this will be used in the last step)
outputList = []
outputListScores = []

# Dictionary for Sections
sections = {'section1':{'height1': '0', 'height2': '54', 'width1': '0','width2': '25'},\
'section2': {'height1': '0', 'height2': '54', 'width1': '45','width2': '70'},\
'section3': {'height1': '0', 'height2': '54', 'width1': '93','width2': '118'},\
'section4': {'height1': '0', 'height2': '54', 'width1': '140','width2': '167'},\
'section5': {'height1': '0', 'height2': '54', 'width1': '185','width2': '215'}}

# Loop through dictionary for Sections

for iterations, cropSize in sections.items():
    #print(iterations)
    h1 = cropSize['height1']
    h2 = cropSize['height2']
    w1 = cropSize['width1']
    w2 = cropSize['width2']
    # print(h1,h2,w1,w2)
    # Crop section using dictionary values
    imgCropSection = imgCropped[int(h1):int(h2),int(w1):int(w2)]
    # Gray and blur
    gray = cv.cvtColor(imgCropSection, cv.COLOR_BGR2GRAY)

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

    # Morphological tranformations
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.dilate(equalize,kernel,iterations = 1)

    # Adaptive Thresholding v6.2.1.
    thresh = cv.adaptiveThreshold(dilation,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,41,11)

    # Draw rectangle
    rect_height, rect_width = resize.shape # gets the size of the resized image
    cv.rectangle(thresh,(0,0),(rect_width, rect_height),(255,255,255),15) # 255,255,255 is white

    # Morphological tranformations
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    ######## TEMPLATE MATCHING TEST

    # Path to image template
    invert_image = np.invert(closing)

    template = invert_image
    #cv_show(n, invert_image)
    
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
    #print("max score：",str(np.max(scores)))

    # Returns the position of the maximum value in the input list
    getNumber = str(np.argmax(scores))
    groupOutput.append(str(np.argmax(scores)))
    
    #print("template matching output：",groupOutput)

    ######## TEMPLATE MATCHING TEST
    
    # Append to OutputList
    outputList.append(int(getNumber))
    outputListScores.append(float(digit_score))

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

## Write data to CSV
#with open('data-v2.10.csv', 'a') as csvfile:
#    writer = csv.writer(csvfile)
#    writer.writerow(finalList)

## Write data to CSV, add headers if none exist

filename = 'data/data.csv'
file_exists = os.path.isfile(filename)
v = random.randint(0, 100)

with open(filename, "a") as csvfile:
    headers = ['d1','d2','d3','d4','d5','d6','timestamp','d1_score','d2_score','d3_score','d4_score','d5_score']
    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=headers)
    if not file_exists:
        writer.writeheader()  # file doesn't exist yet, write a header

    writer = csv.writer(csvfile)
    writer.writerow(finalList)

print(outputList, outputListScores)

### Uncomment to set up Google Sheets with gspread
## # Access Google Sheets with gspread
## 
## gc = gspread.service_account(filename='credentials.json')
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
