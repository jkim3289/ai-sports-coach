from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("yolov8n.pt")  # Try yolov8s.pt for better accuracy

# Run inference and save results
results = model("/Users/jehunkim/Desktop/Summer 2025/ai-sports-coach/videos/test2.MOV", save=True)

# Find where results are saved
save_dir = results[0].save_dir
print(f"Output saved at: {save_dir}")