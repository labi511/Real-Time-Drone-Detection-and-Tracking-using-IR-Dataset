# Real-Time-Drone-Detection-and-Tracking-using-IR-Dataset
# IRDroneTracker

A real-time system for detecting and tracking drones and birds in distorted infrared (IR) imagery. Built using YOLOv8n, YOLOv11n for object detection, ByteTrack, DeepSORT and the Supervision library suitable for multi-object tracking. The project is based on thermal image and video datasets to monitor how different models and algorithms work.


###  Required Libraries

All necessary libraries are imported within each individual Python script

## Repository Contents
All files are placed in the root directory for simplicity. Here's what each script does:

Detection.py:	Handles detection using YOLO models
Tracking_ByteTrack.py:	Performs tracking with ByteTrack
Tracking_DeepSORT.py:	Performs tracking with DeepSORT (optional alternative)
ModelTrain.py:	Model training setup and configuration for YOLO models
Extract_videoframes.py:	Extracts individual frames from IR videos
Val&Test.py:	Validation and testing logic
ZenodoLabels.py:	Converts Zenodo dataset annotations
data_yaml.txt:Dataset YAML configuration for YOLO training

## Methodology

Input
Infrared images and videos from public datasets

Preprocessing
Frame extraction, resizing, and conversion to YOLO format

Detection
YOLOv8n and YOLO11n models trained on IR drone/bird datasets

Tracking
ByteTrack or DeepSORT used for frame-to-frame tracking

Evaluation
Performance tested on unseen validation data using visual and quantitative metrics


## Datasets
We used two high-quality public datasets:

📌 1. ICIP 2023 Thermal Drone-Bird Dataset (via Roboflow)

📌 2. Thermal Video Dataset (via Zenodo)
DOI: 10.5281/zenodo.5500575


