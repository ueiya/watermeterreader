# watermeterreader
Prototype raspberry pi project to read analogue water meters in real-time

# About this project
A DIY project that allows you to track water usage of your home (or facility) in real-time.

# Overview
- A picture is taken every X seconds by raspberry pi
- The picture is analised with OpenCV and digits are extracted
- The digits are saved in CSV file
- A data munging & visualisation script cleans the data and plots graphs of liters used per hour.

# Documentation can be found here:

[Initial Setup: Raspberry Pi Zero W + 📷 Camera Module V2](https://www.notion.so/Initial-Setup-Raspberry-Pi-Zero-W-Camera-Module-V2-804b0005f1f042e094ff3412941c86d1)

[Raspberry Pi: Install virtual env for Python](https://www.notion.so/Raspberry-Pi-Install-virtual-env-for-Python-ef04a7a80c4d4a08a23c2ed14822aa38)

[Raspberry Pi: Install VNC (Virtual Network Computing) to access from Mac](https://www.notion.so/Raspberry-Pi-Install-VNC-Virtual-Network-Computing-to-access-from-Mac-b58ce36f229243d0834ebfec9fd9194c)

[Prototype v2: Raspberry Pi Zero W + 📷 Camera Module V2](https://www.notion.so/Prototype-v2-Raspberry-Pi-Zero-W-Camera-Module-V2-bf881f63faac436d9354521077ed2290)

[Data Munging & Visualisation (TO BE DONE…)](https://www.notion.so/Data-Munging-Visualisation-TO-BE-DONE-aefbbb97911d456bab190dc51b41b910)

# Current challenges
- User Interface, data munging & visualisation: there is currently no UI. I use Jupter Notebook to analyse the data. The data.csv file is copied manually from Raspberry PI to Mac. This could be done in a web server.
- Scalability: This is a prototype and not yet at a stage to scale. There are many types of analogue water meters. Manual setup in the physical world is required and adjustments to he scripts are currently necessary. 
- Accuracy of reading: A higher quality camera would allow beter image definition and more accurate readings.
