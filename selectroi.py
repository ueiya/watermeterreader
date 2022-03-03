# select_roi.py

# Run this code in Mac or PC to define the image regions of interest (ROI) of the test image taken in raspberry pi

import cv2

im = cv2.imread("/Users/daviderubio/Desktop/Ueiya Prototype V2/prototype_v2_photos/v2-2028x1520-image_0.jpg")

roi = cv2.selectROI(im)

print(roi)

im_cropped = im[int(roi[1]):int(roi[1]+roi[3]),
			int(roi[0]):int(roi[0]+roi[2])]

cv2.imshow("Cropped Image", im_cropped)
cv2.waitKey(0)

# Example of printed Result (878, 673, 236, 50)

# Reference table of Roi

# roi 0 = 878
# roi 1 = 673
# roi 2 = 236
# roi 3 = 50

# Example text of what numbers to add to Main Code to crop image 

# [675:723, 878:1114]

# The above text is calculated by [roi 1:roi 1 + roi 3, roi 0:roi 0 + roi 2]