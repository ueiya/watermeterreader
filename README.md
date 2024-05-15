# About Ueiya

Helping people and businesses to measure and optimise your water usage, decrease utility costs, detect leakage and help finance water projects around the world. 

www.ueiya.com

# About this project

The vision of this project is to build a DIY project that allows anyone to track the water usage of an apartment, home or facility in real-time. 

# Overview

    Detect the ROI a picture or pictures of a water meter, read the ROI and detect the digits.
    The Python Script uses a trained Yolov5 Ultralytics model, OpenCV and EasyOCR.
    The ROI is detected with YoloV5 Ultralytics. You need to separately download our trained model ¨best.pt¨ for an accurate detection.
    Digits are extracted with EasyOCR.
    Ideal for water meters with 8 digits.

# Instructions

1. Clone Repository
2. Install Requirements
3. Download best.pt
4. Modify script to point to image folder and best.pt location.

# Documentation

    To be added soon.

# Challenges
    
    Identifying the digits is not accurate enough. We'll improve the script with another method.
    User Interface, data munging & visualisation: there is currently no UI. We recommend using Jupter Notebook to analyse and visualise the data.
