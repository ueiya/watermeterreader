"""ueiya_water_prototype_mac_v06.3.1.py

Detects 5 out of 5 digits in water meter (sometimes)

Raspberry Picamera, takes a picture, capture to openCV object,
Analyzes individual sections/digits,
Uses for loop to iterate digits
Adds output to data.csv
Appends data to Google Sheets with gspread
Pytesseract oem 10 identifies numbers best

    Possible alternative/next steps, suggested by Richard Gould https://github.com/rgould

    A HTTP API running on your laptop, listening for HTTP requests (ie. http://192.168.1.101/send_image)
    after taking a picture, send image to above URL
    ipconfig getifaddr en0
    http://192.168.2.104:8080/send_image

## Crop Sections Dictionary

Watermeter physical digits from left to Write


1 # To get camera digits
USE CODE withe SelectROI() method in OPENCV in your Mac o PC

Camera digits with selectRoi 2022-02-01

Digit 1 = n/a
Digit 2 = n/a
Digit 3 = Section 1, w = 45 to 70 (diff 25)
Digit 4 = Section 2, w = 85 to 110 (diff 25)
Digit 5 = Section 3, w = 125 to 150
Digit 6 = Section 4, w = 170 to 195
Digit 7 = Section 5, w = 210 to 235
Digit 8 = n/a

# Camera Digits reference table
# The meter we use, uses 8 digits. We will remove digit 1 and 2 from the picture, because we´re not going to track them in this version of the prototype.

Digit 1 = n/a
Digit 2 = n/a
Digit 3 = Section 1, w = 235 to 340
Digit 4 = Section 2, w = 340 to 455
Digit 5 = Section 3, w = 455 to 570
Digit 6 = Section 4, w = 570 to 685
Digit 7 = Section 5, w = 685 to 789
Digit 8 = n/a

sections = {'section1': {'height1': '0', 'height2': '150', 'width1': '230','width2': '305'},'section2': {'height1': '0', 'height2': '150', 'width1': '340','width2': '420'},'section3': {'height1': '0', 'height2': '150', 'width1': '445','width2': '530'},'section4': {'height1': '0', 'height2': '150', 'width1': '570','width2': '653'},'section5': {'height1': '0', 'height2': '150', 'width1': '680','width2': '760'}}
"""


# Import packages

import csv
import datetime
import re

import cv2 as cv
import numpy as np
import pytesseract

from ..config import (
    credentials_file,
    csv_data_path,
    google_key,
    image_path_in,
    image_path_out,
    sections,
)
from .util.util import process_section

# from picamera import PiCamera
# from time import sleep

# import gspread

## Camera Resolution


# Width: 4056
# Heigth: 3040

# camera.resolution = (2028, 1520) # Good alternative


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


# Accesing image
img = cv.imread(image_path_in)

# Crop image v1
img_cropped = img[675:723, 878:1114]
cv.imwrite(
    image_path_out,
    img_cropped,
)

# Define list for output to CSV (this will be used in the last step)
output = []

for iterations, crop_size in sections.items():
    process_section(iterations, crop_size, img_cropped=img_cropped, output=output)

# Add last digit to output list. Digit is 0.
last_digit: int = 0
output.append(last_digit)

# Open CSV and append outputList in different columns (each value is separated by a comma / creo)
timestamp = datetime.datetime.now()  # create a timestamp variable
output.append(str(timestamp))

with open(csv_data_path, "a") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(output)

print(output)

# Access Google Sheets with gspread

gc = gspread.service_account(filename=credentials_file)

# Open spreadsheet by key
sh = gc.open_by_key(google_key)

# Open worksheet
wks = sh.worksheet("raw-data")

# Search for a table in the worksheet and append a row to it
wks.append_rows(
    [output],
    value_input_option="USER-ENTERED",
    insert_data_option=None,
    table_range=None,
)
# wks.append_rows([outputList]) # Simple append, no extra options
