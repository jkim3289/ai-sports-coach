import cv2
import numpy as np
from ultralytics import YOLO

# Load the VOLOv8 model
model = YOLO("yolo11n-pose.pt")
# model = YOLO("yolov8n.pt")

# Testing out yolov8n pose
# model = YOLO("yolov8n.pt")

# Select class IDs for 'person', 'sports ball', and 'tennis racket'
# target_classes = [k for k, v in model.names.items() if v in ["person", "sports ball", "tennis racket"]]

# Run inference on the input video, filtering only target classes and save the output
# results = model("/Users/jehunkim/Desktop/Summer 2025/ai-sports-coach/videos/test1.MOV", classes=target_classes, save=True)
results = model("/Users/jehunkim/Desktop/Summer 2025/ai-sports-coach/videos/Serve1.MOV", save=True)

# Get the directory where the results were saved
save_dir = results[0].save_dir
print(f"Output saved at: {save_dir}")





# # Load the yolo11n-pose and yolov8n model
# pose_model = YOLO("yolo11n-pose.pt")
# detect_model = YOLO("yolov8n.pt")


# # === Input video ===
# # Load video and process frame by frame
# video_path = "/Users/jehunkim/Desktop/Summer 2025/ai-sports-coach/videos/Serve1.MOV"
# cap = cv2.VideoCapture(video_path)
# fps = cap.get(cv2.CAP_PROP_FPS)
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # === Output video writer ===
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("annotated_output.mp4", fourcc, fps, (w, h))

# # === Define COCO skeleton connections ===
# SKELETON = [
#     (5, 7), (7, 9),      # Left arm
#     (6, 8), (8, 10),     # Right arm
#     (5, 6),              # Shoulders
#     (11, 13), (13, 15),  # Left leg
#     (12, 14), (14, 16),  # Right leg
#     (11, 12),            # Hips
#     (5, 11), (6, 12)     # Torso sides
# ]

# # === Ball tracking memory ===
# ball_trail = []

# # === Process video frame by frame ===
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # === Run pose estimation ===
#     pose_result = pose_model(frame)[0]
#     keypoints = pose_result.keypoints.xy  # (num_people, 17, 2)

#     # === Run detection for sports ball (class 32) ===
#     detect_result = detect_model(frame, classes=[32])[0]
#     boxes = detect_result.boxes

#     # === Draw pose stickman ===
#     if keypoints is not None and len(keypoints) > 0:
#         for person in keypoints:
#             person = np.array(person)
#             if person.shape[0] < 17:
#                 continue  # Skip if incomplete detection

#             # Draw skeleton lines
#             for pt1, pt2 in SKELETON:
#                 if pt1 >= person.shape[0] or pt2 >= person.shape[0]:
#                     continue
#                 x1, y1 = person[pt1]
#                 x2, y2 = person[pt2]
#                 cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#             # Draw joints
#             for x, y in person:
#                 cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
#     else:
#         print("No pose detected in this frame.")

#     # === Draw sports ball detections and update trail ===
#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         label = f"Sports Ball {conf:.2f}"

#         # Center of ball
#         cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
#         ball_trail.append((cx, cy))

#         # Draw bounding box and label
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
#         cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

#     # === Draw trajectory of the ball (last 50 points) ===
#     for i in range(1, min(len(ball_trail), 50)):
#         cv2.line(frame, ball_trail[i - 1], ball_trail[i], (0, 255, 255), 2)

#     # === Show and save output frame ===
#     cv2.imshow("Tennis Pose + Ball", frame)
#     out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # === Release everything ===
# cap.release()
# out.release()
# cv2.destroyAllWindows()