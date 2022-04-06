# select_roi.py

# Run this code in Mac or PC to define the image regions of interest (ROI) of the test image taken in raspberry pi

import cv2
from matplotlib import pyplot as plt

# Open the image
im = cv2.imread("/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/v2_test_photos/calibrate.jpg")

# Select image roi - User input required in GUI
roi = cv2.selectROI(im)

print('ROI is: ' + str(roi))

im_cropped = im[int(roi[1]):int(roi[1]+roi[3]),int(roi[0]):int(roi[0]+roi[2])]

cv2.imshow("Cropped Image", im_cropped)
cv2.waitKey(0)

cv2.imwrite('/Users/daviderubio/Desktop/Python_stuff/environments/ueiya_env/images/cropped_image.jpg',im_cropped)

# Print output ordered for watermeterreader.py

print('Copy the following number sequence to watermeterreader.py: ' + str(roi[1])+ ':' + str(roi[1]+roi[3])+ ',' +str(roi[0])+ ':' +str(roi[0]+roi[2]))

# Show image cropped to find widths for 'Dictionary for Sections'
plt.imshow(cv2.cvtColor(im_cropped, cv2.COLOR_BGR2RGB))
plt.title('Image Cropped'); plt.show()


# Reference table of Roi

# roi 0 = 878
# roi 1 = 673
# roi 2 = 236
# roi 3 = 50

# Example text of what numbers to add to Main Code to crop image 

# [675:723, 878:1114]

# The above text is calculated by [roi 1:roi 1 + roi 3, roi 0:roi 0 + roi 2]