from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv8 model (pre-trained)
model = YOLO("yolov8n.pt")  # YOLOv8n (nano) for speed, use 'yolov8m.pt' for better accuracy

# Path to vehicle image
image_path = r"C:\Users\siddh\OneDrive\Desktop\Vehicle Recognition Project\Car for Recognition\car3.jpg"

# Read image
image = cv2.imread(image_path)

# Run YOLO on the image
results = model(image)

# Draw bounding boxes on detected vehicles
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        conf = box.conf[0].item()  # Confidence score
        class_id = int(box.cls[0].item())  # Class ID

        # Get class name (e.g., "car", "truck", "bus")
        class_name = model.names[class_id]

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(image, f"{class_name} ({conf:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

# Display processed image using Matplotlib (avoids OpenCV imshow() errors)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Vehicles with YOLO")
plt.axis("off")
plt.show()
