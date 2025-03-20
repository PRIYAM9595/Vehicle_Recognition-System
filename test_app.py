from flask import Flask, render_template, Response
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class Camera:
    def __init__(self):
        logger.info("Initializing camera...")
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Could not open camera.")
        logger.info("Camera initialized successfully")

    def get_frame(self):
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None

        try:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
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

if __name__ == '__main__':
    logger.info("Starting server...")
    logger.info("Open http://127.0.0.1:5000 in your browser")
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise 