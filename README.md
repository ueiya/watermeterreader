# About Ueiya
Measure and optimise your water usage, decrease your utility costs, and help finance water projects around the world.
www.ueiya.com

# About this project
A DIY project that allows you to track the water usage of your home (or facility) in real-time. The tools needed are raspberry pi, a camera, python, OpenCV (among other libraries)

# Overview
- A picture of an analogue water meter is taken every X seconds by raspberry pi + camera.
- The picture is analised with a python script that uses OpenCV and digits are extracted.
- The digits are saved in CSV file.
- A data munging & visualisation script cleans the data and plots graphs of water liters used per hour/day.

# Documentation can be found here:

[Initial Setup: Raspberry Pi Zero W + 📷 Camera Module V2](https://www.notion.so/Initial-Setup-Raspberry-Pi-Zero-W-Camera-Module-V2-804b0005f1f042e094ff3412941c86d1)

[Raspberry Pi: Install virtual env for Python](https://www.notion.so/Raspberry-Pi-Install-virtual-env-for-Python-ef04a7a80c4d4a08a23c2ed14822aa38)

[Raspberry Pi: Install VNC (Virtual Network Computing) to access from Mac](https://www.notion.so/Raspberry-Pi-Install-VNC-Virtual-Network-Computing-to-access-from-Mac-b58ce36f229243d0834ebfec9fd9194c)

[Prototype v2: Raspberry Pi Zero W + 📷 Camera Module V2](https://www.notion.so/Prototype-v2-Raspberry-Pi-Zero-W-Camera-Module-V2-bf881f63faac436d9354521077ed2290)

[Data Munging & Visualisation (TO BE DONE…)](https://www.notion.so/Data-Munging-Visualisation-TO-BE-DONE-aefbbb97911d456bab190dc51b41b910)

# Challenges (Where I need your support)
- User Interface, data munging & visualisation: there is currently no UI. I use Jupter Notebook to analyse and visualise the data. The data.csv file that is generated in the raspberry pi is copied manually from to my Mac. There's big room for improvement here.
- Scalability: This is a prototype v2 and is not yet at a stage than can scale. There are many types of analogue water meters with individual differences. Manual setup in the physical world is required and custom adjustments to the scripts are necessary. 
- Accuracy of reading: A higher quality camera would allow beter image definition and more accurate readings.
