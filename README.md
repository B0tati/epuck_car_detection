# Car Detection using YOLO and Webots Simulation

This repository contains code and resources for detecting cars using YOLOv8 and simulating the robot's behavior in Webots. The system uses a pre-trained YOLO model to detect cars from images captured by the robot's camera while navigating in a simulated environment. It also includes files necessary for training the YOLO model and setting up the simulation environment.

## Files in this Repository

1. **colab_file.txt**: 
   - This file contains the code for training the YOLO model using a dataset stored on Google Drive. The code is designed to work in a Google Colab environment.
   - You can upload this file to your Google Colab environment and run the code to train the YOLO model.
2. **webots.py**:
   - A Python script for controlling a Webots robot using its sensors (proximity sensors) and camera. The script integrates with the YOLOv8 model to detect cars in the robot's environment and avoid obstacles.
3. **car_dataset.yaml**:
   - This YAML file defines the dataset paths and configuration needed to train the YOLO model. It specifies the training, validation, and test image paths, the number of classes (1 class for 'car'), and the names of the classes.

## Libraries Used

1. **Ultralytics YOLO**: 
   - `from ultralytics import YOLO`
   - For object detection using the YOLOv8 model.
   - Install with: `pip install ultralytics`
2. **OpenCV**:
   - `import cv2`
   - For image processing and displaying images with bounding boxes.
   - Install with: `pip install opencv-python`
3. **NumPy**:
   - `import numpy as np`
   - For numerical operations and image manipulation.
   - Install with: `pip install numpy`
4. **Webots API**:
   - `from controller import Robot`
   - For interacting with Webots robot and controlling its sensors and motors.
5. **Google Colab Libraries**:
   - `from google.colab import drive, files`
   - For mounting Google Drive and uploading files in Google Colab.
6. **Shutil**:
   - `import shutil`
   - For file operations such as copying files.
7. **IPython**:
   - `from IPython.display import Image`
   - For displaying images in Colab.

## How to Use

### 1. Training the YOLO Model
   - Upload the `colab_file.txt` to [Google Colab](https://colab.research.google.com/).
   - Make sure to update the file paths for your dataset [Kaggle](https://kaggle.com/), as they are currently set to specific Google Drive locations.
   - Run the code to start training the model for detecting cars. The model will be saved after training is complete.
### 2. Using the Webots Robot for Object Detection
   - Upload `webots.py` to your local environment and ensure you have the Webots simulation running.
   - Make sure the Webots robot is equipped with proximity sensors and a camera.
   - The script captures images from the robot's camera, runs object detection using the trained YOLO model, and performs obstacle avoidance based on sensor readings.
   - You will need to change the path to your trained model file (`best.pt`) in the code.
### 3. Model Configuration (car_dataset.yaml)
   - The `car_dataset.yaml` file provides paths for training, validation, and testing datasets. Ensure the paths are correctly set to the folders containing your images.
   - This configuration file is used during the training of the YOLO model in the Colab environment.

## Pictures:
![annotated_image_1739784123](https://github.com/user-attachments/assets/16c50564-ddb7-480b-8359-33c2a6057e9a)

## Project by - Varshini
