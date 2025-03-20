from ultralytics import YOLO
import cv2
import pandas as pd
import datetime

# Load YOLO model
model = YOLO("yolov8n.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

# CSV File Setup
log_file = "vehicle_log.csv"
columns = ["Timestamp", "Vehicle Type", "Confidence"]
df = pd.DataFrame(columns=columns)

# If log file doesn't exist, create one with headers
try:
    existing_data = pd.read_csv(log_file)
except FileNotFoundError:
    df.to_csv(log_file, index=False)

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
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]

            # Get timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Log the detection in CSV
            new_entry = pd.DataFrame([[timestamp, class_name, conf]], columns=columns)
            new_entry.to_csv(log_file, mode='a', header=False, index=False)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Real-Time Vehicle Detection", frame)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()