import torch
import cv2
import warnings
import os
from ultralytics import YOLO
import pygame

# Suppress the wayland warning
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Set to "wayland" if needed

# Suppress Qt font warnings
warnings.filterwarnings("ignore", message="QFont::fromString")
# Suppress PyTorch FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable CPU
device = 'cpu'
model = YOLO("yolo11s.pt").to(device)

# Open video file or webcam
cap = cv2.VideoCapture(0)  # Replace 0 with file path for a video

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarm.mp3')

alerted = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to smaller size for CPU
    #frame = cv2.resize(frame, (1024, 1024))
    
    # Convert image to RGB and to appropriate device and precision
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    img_rgb = img_rgb.float()  # Convert to floating point
    img_rgb = img_rgb / 255.0  # Normalize tensor to [0.0, 1.0]

    # Run YOLOv9 inference
    with torch.no_grad():
        results = model(img_rgb)

    # Check for human detection and play sound once
    if 0 in results[0].boxes.cls:
        if not alerted:
            alarm_sound.play()
            alerted = True
    else:
        alerted = False

    # Convert the results back to BGR format for OpenCV
    result_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)
    
    # Scale the result image for display
    result_img = cv2.resize(result_img, (1280, 860))  # Adjust dimensions as needed

    # Display the output frame
    cv2.imshow('Uykum geldi', result_img)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
