"""Wrapper class for our required camera functionality"""

import cv2
import numpy as np

class Camera:
    def __init__(self):
        pass

    """Take the image as a jpeg and return it"""
    def capture_image(self):
        # Open the first webcam (index 0)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        # Optionally set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Capture a single frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture image from webcam")

        # Encode the frame as JPEG in memory
        ret, jpeg = cv2.imencode('.jpg', frame)

        if not ret:
            raise RuntimeError("Failed to encode image to JPEG")
        
        # release the camera
        cap.release()

        # Return JPEG image as byte array
        return jpeg.tobytes()
    
    """Draw bounding boxes around faces and crop to largest face"""
    def process_image(self, image_bytes):
        # Convert byte array back to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Load pre-trained face detection model (Haar Cascade)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            raise RuntimeError("No faces detected in the image")

        # Find the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = largest_face

        # Draw bounding box around the largest face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop to the largest face
        cropped_face = img[y:y+h, x:x+w]

        # Encode the processed image back to JPEG
        ret, processed_jpeg = cv2.imencode('.jpg', cropped_face)

        if not ret:
            raise RuntimeError("Failed to encode processed image to JPEG")

        return processed_jpeg.tobytes()
    
