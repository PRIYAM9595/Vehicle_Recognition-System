from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import datetime
import numpy as np
import os
import logging
import pandas as pd
from flask_mail import Mail
from models import db, Vehicle, BlacklistedVehicle, Alert
import easyocr
from services.notifications import NotificationService
from config import Config

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
mail = Mail(app)

# Initialize EasyOCR
logger.info("Initializing EasyOCR...")
try:
    reader = easyocr.Reader(['en'])
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR: {e}")
    reader = None

# Excel File Setup
EXCEL_FILE = "vehicle_detections.xlsx"
COLUMNS = ["Timestamp", "License Plate", "Vehicle Type", "Confidence", "Location"]

# Create Excel file if it doesn't exist
if not os.path.exists(EXCEL_FILE):
    logger.info(f"Creating new Excel file: {EXCEL_FILE}")
    df = pd.DataFrame(columns=COLUMNS)
    df.to_excel(EXCEL_FILE, index=False)

def log_vehicle_detection(timestamp, license_plate="Unknown", vehicle_type="Unknown", confidence=0.0, location=""):
    """Log vehicle detection to Excel file."""
    try:
        # Read existing data
        if os.path.exists(EXCEL_FILE):
            df = pd.read_excel(EXCEL_FILE)
        else:
            df = pd.DataFrame(columns=COLUMNS)
        
        # Add new detection
        new_row = {
            "Timestamp": timestamp,
            "License Plate": license_plate,
            "Vehicle Type": vehicle_type,
            "Confidence": confidence,
            "Location": location
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to Excel
        df.to_excel(EXCEL_FILE, index=False)
        logger.info(f"Logged vehicle detection: {license_plate}")
        return True
    except Exception as e:
        logger.error(f"Error logging vehicle detection: {e}")
        return False

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class Camera:
    def __init__(self):
        logger.info("Initializing camera...")
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Could not open camera.")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Reduced width
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced height
        self.camera.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Enable auto exposure
        
        logger.info("Camera initialized successfully")
        self.frame_count = 0
        self.last_detection_time = datetime.datetime.now()
        self.detection_interval = 1  # Reduced to 1 second for faster detection
        self.skip_frames = 1  # Process every frame for smoother video
        self.last_processed_frame = None
        self.last_detected_plate = None
        self.detection_count = 0

    def detect_license_plate(self, frame):
        """Detect license plate in the frame using EasyOCR."""
        if reader is None:
            logger.error("EasyOCR not initialized")
            return None, 0.0

        try:
            # Convert frame to grayscale for better text detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to enhance text
            # 1. Denoise (reduced strength for speed)
            gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
            
            # 2. Increase contrast (reduced parameters for speed)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
            gray = clahe.apply(gray)
            
            # 3. Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Detect text with optimized parameters
            results = reader.readtext(binary, 
                                   paragraph=False,
                                   batch_size=1,
                                   min_size=8,  # Reduced minimum size
                                   width_ths=0.7,
                                   height_ths=0.7,
                                   ycenter_ths=0.5,
                                   x_ths=1.0,
                                   y_ths=0.5,
                                   slope_ths=0.1)
            
            # Filter and process results
            for (bbox, text, prob) in results:
                # Clean the detected text
                text = ''.join(c for c in text if c.isalnum())
                
                if len(text) >= 5:  # Minimum length for a license plate
                    # Draw rectangle around the detected text
                    (top_left, top_right, bottom_right, bottom_left) = bbox
                    cv2.rectangle(frame, 
                                (int(top_left[0]), int(top_left[1])),
                                (int(bottom_right[0]), int(bottom_right[1])),
                                (0, 255, 0), 2)
                    
                    # Add text above the rectangle
                    cv2.putText(frame, text, 
                              (int(top_left[0]), int(top_left[1] - 10)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add confidence score
                    cv2.putText(frame, f"Conf: {prob:.2f}", 
                              (int(top_left[0]), int(bottom_right[1] + 20)),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    logger.info(f"Detected license plate: {text} with confidence: {prob}")
                    return text, prob
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error in license plate detection: {e}")
            return None, 0.0

    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        try:
            self.frame_count += 1
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Add a semi-transparent overlay for better text visibility
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 150), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Check if enough time has passed since last detection
            current_time = datetime.datetime.now()
            if (current_time - self.last_detection_time).total_seconds() >= self.detection_interval:
                # Detect license plate
                license_plate, confidence = self.detect_license_plate(frame)
                
                if license_plate:
                    # Only log if it's a new plate or enough time has passed
                    if license_plate != self.last_detected_plate or self.detection_count >= 3:
                        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                        log_vehicle_detection(
                            timestamp=timestamp,
                            license_plate=license_plate,
                            vehicle_type="Car",
                            confidence=confidence,
                            location="Camera 1"
                        )
                        self.last_detected_plate = license_plate
                        self.detection_count = 0
                        self.last_detection_time = current_time
                    else:
                        self.detection_count += 1
                    
                    # Draw detection box with vehicle details
                    height, width = frame.shape[:2]
                    box_height = 100
                    box_y = height - box_height - 10
                    
                    # Draw semi-transparent background for vehicle details
                    cv2.rectangle(frame, (10, box_y), (width-10, height-10), (0, 0, 0), -1)
                    cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
                    
                    # Draw vehicle details
                    cv2.putText(frame, f"Vehicle Detected!", 
                              (20, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"License Plate: {license_plate}", 
                              (20, box_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                              (20, box_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw timestamp and detection status on frame
            height, width = frame.shape[:2]
            
            # Draw timestamp
            cv2.putText(frame, f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add detection status with color coding
            status = "Detecting..." if (current_time - self.last_detection_time).total_seconds() < self.detection_interval else "Ready"
            color = (0, 255, 0) if status == "Ready" else (0, 255, 255)
            cv2.putText(frame, f"Status: {status}", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {self.frame_count}", 
                      (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add instructions
            cv2.putText(frame, "Hold license plate steady for detection", 
                      (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert frame to JPEG with reduced quality for better performance
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Store the processed frame
            self.last_processed_frame = buffer.tobytes()
            return self.last_processed_frame

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None

    def __del__(self):
        logger.info("Releasing camera...")
        self.camera.release()

camera = None

def gen_frames():
    global camera
    if camera is None:
        try:
            camera = Camera()
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return

    while True:
        try:
            frame = camera.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error in gen_frames: {e}")
            break

@app.route('/')
def index():
    logger.info("Serving index page")
    return render_template('test_index.html')

@app.route('/video_feed')
def video_feed():
    logger.info("Starting video feed")
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download_excel')
def download_excel():
    """Download the Excel file with vehicle detections."""
    try:
        return send_file(
            EXCEL_FILE,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='vehicle_detections.xlsx'
        )
    except Exception as e:
        logger.error(f"Error downloading Excel file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles')
def get_vehicles():
    """Get list of detected vehicles with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    vehicles = Vehicle.query.order_by(Vehicle.timestamp.desc()).paginate(page=page, per_page=per_page)
    return jsonify({
        'vehicles': [v.to_dict() for v in vehicles.items],
        'total': vehicles.total,
        'pages': vehicles.pages,
        'current_page': vehicles.page
    })

@app.route('/api/alerts')
def get_alerts():
    """Get list of alerts with pagination."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    alerts = Alert.query.order_by(Alert.timestamp.desc()).paginate(page=page, per_page=per_page)
    return jsonify({
        'alerts': [a.to_dict() for a in alerts.items],
        'total': alerts.total,
        'pages': alerts.pages,
        'current_page': alerts.page
    })

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    logger.info("Starting server...")
    logger.info("Open http://127.0.0.1:8080 in your browser")
    try:
        # Use port 8080 and bind to all interfaces
        app.run(
            host='127.0.0.1',  # Use localhost explicitly
            port=8080,
            debug=True,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error("Please check if the port is already in use or if you have sufficient permissions.")
        raise