# Vehicle Recognition System

A real-time vehicle license plate detection system using Python, Flask, OpenCV, and EasyOCR.

## Features

- Real-time license plate detection using webcam
- Automatic logging of detected vehicles to Excel
- User-friendly web interface
- Downloadable vehicle detection logs
- High-performance camera feed
- Easy-to-use interface

## Requirements

- Python 3.8+
- OpenCV
- Flask
- EasyOCR
- Pandas
- OpenPyXL

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vehicle-recognition.git
cd vehicle-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://127.0.0.1:8080
```

## Usage

1. Start the application
2. Hold a license plate steady in front of the camera
3. The system will automatically detect and log the plate number
4. Download the Excel file to view all detections

## Project Structure

```
vehicle-recognition/
├── app.py              # Main application file
├── config.py           # Configuration settings
├── models.py           # Database models
├── requirements.txt    # Project dependencies
├── static/            # Static files
│   └── uploads/       # Upload directory
├── templates/         # HTML templates
│   └── test_index.html
└── vehicle_detections.xlsx  # Detection logs
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for image processing
- EasyOCR for text recognition
- Flask for web framework 