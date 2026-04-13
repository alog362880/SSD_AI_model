# SSD_AI_model
this is a AI model for object detection
SSD-MobileNet Object Detection on COCO
This project implements an object detection pipeline using SSDLite320 with a MobileNetV3-Large backbone. The model is trained on a subset of the COCO 2017 dataset to detect specific traffic-related categories. It includes full workflows for data preparation, model training, and inference on both static images and video files.

Table of Contents
Features

Target Categories

Project Structure

Getting Started

Training Details

Inference

Evaluation

Features
Automated Data Management: Downloads and extracts the COCO 2017 dataset automatically.

Custom Dataset Class: A specialized COCOClassificationDataset class that filters for specific categories and handles bounding box normalization.

SSDLite Architecture: Utilizes ssdlite320_mobilenet_v3_large for efficient detection suitable for mobile or edge devices.

Video Processing: Supports frame-by-frame inference on uploaded video files, generating an output video with overlaid bounding boxes and labels.

Target Categories
The model is configured to detect the following 8 classes from the COCO dataset:

Person

Car

Truck

Motorcycle

Stop Sign

Traffic Light

Bus

Bicycle

Project Structure
Data Setup: Mounts Google Drive and defines paths for COCO training, validation, and annotations.

Preprocessing: Implements horizontal flipping for training augmentation and tensor conversion for validation.

Model Building: Initializes the SSDLite model with pre-trained MobileNetV3 backbone weights.

Training Loop: Implements training with SGD optimizer, StepLR scheduler, and automatic checkpoint saving for the best validation loss.

Inference: Provides tools to upload images or videos and visualize detection results with confidence thresholds.

Getting Started
Prerequisites
Python 3.x

PyTorch / Torchvision

OpenCV (cv2)

PIL (Pillow)

Matplotlib

Tqdm

Setup
Open the notebook in Google Colab.

Ensure your Colab instance has access to Google Drive for permanent dataset storage.

Set the runtime to GPU for significantly faster training and inference.

Training Details
Seed: Fixed at 42 for reproducibility.

Dataset Size: Sampled to 150 images per class for training and 30 per class for validation.

Optimizer: SGD with a learning rate of 0.0001, momentum of 0.9, and weight decay of 0.0005.

Epochs: Default is set to 10, with a scheduler reducing the learning rate every 5 epochs.

Inference
Static Images
The notebook allows you to upload a local image. The model then performs detection and displays the image with red bounding boxes and blue labels for detected objects.

Video Processing
Upload any .mp4 or compatible video. The script processes every 10th frame (adjustable) and reconstructs a video with detection overlays, which is then automatically downloaded to your local machine.

Evaluation
The project includes an Intersection over Union (IoU) calculation function to evaluate the spatial accuracy of the predicted bounding boxes against ground truth.
