#2022-03-23 digitprocessingfortemplate.py

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

######## TEMPLATE MATCHING TEST
 
# Custom display function
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)

# Open image
path = '/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/v2_test_photos/d9.jpg'
img = cv.imread(path)

# Gray and blur
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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

# Morphological tranformations v.6.3.1
closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

# erosion = cv.erode(closing,kernel,iterations = 2)
# 
# # Blur to b/w image
# blur2 = cv.GaussianBlur(erosion,(5,5),10)
# blur3 = cv.GaussianBlur(blur2,(5,5),10)
# blur4 = cv.GaussianBlur(blur3,(5,5),10)
# blur5 = cv.GaussianBlur(blur4,(5,5),10)
# 
# # Edge detection
# edges = cv.Canny(blur5,200,200)
# 
# # Contours
# contours, hierarchy = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# 
# #Contours v6.2.1. (detects 5/5)
# contours_dict = dict()
# for cont in contours:
#     x, y, w, h = cv.boundingRect(cont)
#     area = cv.contourArea(cont)
#     if 10 < area and 10 < w and h > 5:
#         contours_dict[(x, y, w, h)] = cont
# 
# contours_filtered = sorted(contours_dict.values(), key=cv.boundingRect)
# 
# blank_background = np.zeros_like(edges)
# 
# img_contours = cv.drawContours(blur5, contours_filtered, -1, (0,0,0), thickness=2) #black

# Invert image
invert_image = np.invert(closing)

# Save image
cv.imwrite('/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/v2_test_photos/digit_processed_9.jpg',invert_image)
