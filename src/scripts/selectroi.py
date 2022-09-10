# selectroi.py

# Run this code in Mac or PC to define the image regions of interest (ROI) of the test image taken in raspberry pi
# Then use the output text for watermeterreader.py

import cv2 as cv
from matplotlib import pyplot as plt

# Custom display function
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)

n = 'image preview'

# Open the image
path = "images/calibrate.jpg"
img = cv.imread(path)

# Rotate image - add '0' in rotate_angle if no rotation is necessary
(h, w) = img.shape[:2]
rotate_center = (w / 2, h / 2)
rotate_angle = -2
rotate_scale = 1

rotate_matrix = cv.getRotationMatrix2D(rotate_center, rotate_angle, rotate_scale)
imgRotated = cv.warpAffine(img, rotate_matrix, (w, h))
#cv_show(n, imgRotated)

# Select image roi - User input required in GUI
roi = cv.selectROI(imgRotated)

print('ROI is: ' + str(roi))

img_cropped = imgRotated[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]

cv.imshow("Cropped Image", img_cropped)

cv.imwrite('images/cropped_image.jpg',img_cropped)

# Print output ordered for watermeterreader.py

print('Copy the following number sequence to watermeterreader.py: ' + str(roi[1])+ ':' + str(roi[1]+roi[3])+ ',' +str(roi[0])+ ':' +str(roi[0]+roi[2]))

# Show image cropped to find widths for 'Dictionary for Sections'
plt.imshow(cv.cvtColor(img_cropped, cv.COLOR_BGR2RGB))
plt.title('Image Cropped'); plt.show()

# Reference table of Roi

# roi 0 = 878
# roi 1 = 673
# roi 2 = 236
# roi 3 = 50

# Example number sequence

# [675:723, 878:1114]

# The above text is calculated by [roi 1:roi 1 + roi 3, roi 0:roi 0 + roi 2]