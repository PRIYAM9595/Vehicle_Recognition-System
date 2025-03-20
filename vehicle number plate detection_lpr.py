import easyocr
import cv2
import re
import matplotlib.pyplot as plt

# Path to the sample image
image_path = r"C:\Users\siddh\OneDrive\Desktop\Vehicle Recognition Project\Car for Recognition\car2.jpg"

# Load the image
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding for better contrast
processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Perform OCR on the processed image
results = reader.readtext(processed)

# Print all detected text to debug
print("Raw OCR Output:")
for bbox, text, prob in results:
    print(f"Detected: {text} (Confidence: {prob:.2f})")

# Regular expression for Indian license plates (e.g., MH01AX8888)
plate_pattern = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$")

# Extract only valid license plate text
filtered_plate = [text for _, text, _ in results if plate_pattern.match(text)]

# Display results
if filtered_plate:
    print("Final License Plate:", filtered_plate[0])
else:
    print("No valid license plate detected.")

# **Fix OpenCV imshow() issue using Matplotlib**
plt.figure(figsize=(8, 4))
plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
plt.title("Processed Image for OCR")
plt.axis("off")  # Hide axes
plt.show()
