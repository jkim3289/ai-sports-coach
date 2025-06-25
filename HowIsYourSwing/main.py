from ultralytics import YOLO
import cv2
import os

# Load the VOLOv8 model
model = YOLO("yolov8n.pt")  # Try yolov8s.pt for better accuracy

# Select class IDs for 'person', 'sports ball', and 'tennis racket'
target_classes = [k for k, v in model.names.items() if v in ["person", "sports ball", "tennis racket"]]

# Run inference on the input video, filtering only target classes and save the output
results = model("/Users/jehunkim/Desktop/Summer 2025/ai-sports-coach/videos/test1.MOV", classes=target_classes, save=True)

# Get the directory where the results were saved
save_dir = results[0].save_dir
print(f"Output saved at: {save_dir}")