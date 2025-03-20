from ultralytics import YOLO
import cv2
import pandas as pd
import datetime
import easyocr
import numpy as np
import os
import re

# Initialize YOLO model for vehicle detection
model = YOLO("yolov8n.pt")

# Initialize EasyOCR for license plate recognition
reader = easyocr.Reader(['en'])

# Open webcam
cap = cv2.VideoCapture(0)

# CSV File Setup
log_file = "vehicle_log.csv"
columns = ["Timestamp", "License Plate", "Vehicle Type", "Confidence"]

# Check and fix CSV file issues
def fix_csv():
    try:
        df = pd.read_csv(log_file, usecols=[0, 1, 2, 3], on_bad_lines='skip')  # Keep first 4 columns
        df.to_csv(log_file, index=False)
        print("✅ CSV file fixed!")
    except Exception as e:
        print("⚠ CSV file issue:", e)
        df = pd.DataFrame(columns=columns)
        df.to_csv(log_file, index=False)

# Run CSV check
if not os.path.exists(log_file):
    fix_csv()

# Preprocessing Function
def preprocess_plate(plate_img):
    """ Preprocess license plate image for better OCR accuracy. """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edged = cv2.Canny(blur, 100, 200)  # Edge detection
    return edged

# Regular expression for Indian license plates (e.g., MH01AX8888)
plate_pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$")

def extract_license_plate_text(processed_plate):
    """ Extract license plate text using EasyOCR and filter using regex. """
    results = reader.readtext(processed_plate)
    filtered_plate = [text for _, text, _ in results if plate_pattern.match(text)]
    return filtered_plate[0] if filtered_plate else "Unknown"

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0].item())  # Vehicle class ID
            class_name = model.names[class_id]  # Vehicle type

            # Extract License Plate ROI (Lower part of vehicle)
            plate_roi = frame[y1:y2, x1:x2]
            plate_text = "Unknown"

            if plate_roi.shape[0] > 0 and plate_roi.shape[1] > 0:
                processed_plate = preprocess_plate(plate_roi)
                plate_text = extract_license_plate_text(processed_plate)

            # Get timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log the detection
            new_entry = pd.DataFrame([[timestamp, plate_text, class_name, conf]], columns=columns)
            new_entry.to_csv(log_file, mode='a', header=False, index=False)

            # Draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Plate: {plate_text}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display frame (Handle OpenCV GUI errors)
    try:
        cv2.imshow("Vehicle Detection & License Plate Recognition", frame)
    except cv2.error:
        cv2.imwrite("output_frame.jpg", frame)
        print("⚠ OpenCV GUI Error: Saved frame as 'output_frame.jpg'")

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()