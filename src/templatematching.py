# ocr_template_match.py

# import the necessary packages
from imutils import contours
import numpy as np
# import argparse
import imutils # add how to install in tutorial
import cv2

## construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#help="/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/images/last_image_taken.jpg")
#ap.add_argument("-r", "--reference", required=True,
#help="/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/ueiya_gimp/ocr-b.jpg")
#args = vars(ap.parse_args())

# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
path = "/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/ueiya_gimp/ocr-ueiya.jpg"
ref = cv2.imread(path)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

cv2.imshow("ref", ref)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # find contours in the OCR-A image (i.e,. the outlines of the digits)
# # sort them from left to right, and initialize a dictionary to map
# # digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {
}
 
# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))
	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi

print(digits)

## >>> NEXT STEP: https://pythonmana.com/2021/11/20211126170518135E.html
