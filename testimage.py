# Import packages

from picamera import PiCamera
from time import sleep

# Define Camera
camera = PiCamera()

# Start camera, define ISO and resolution
camera.start_preview()
camera.iso = 800
camera.resolution = (2028, 1520)

# Camera warm-up time
sleep(2)

# Save image
camera.capture('/home/pi/python/environments/ueiya_env/images/calibrate.jpg')
camera.stop_preview()