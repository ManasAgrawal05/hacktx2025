"""Wrapper class for our required camera functionality"""

import cv2
import numpy as np
import os

#TODO: make bounding box square and resize to 250 x 250

class Camera:
    def __init__(self):
        # Load the DNN face detector
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        proto_path = os.path.join(model_dir, 'deploy.prototxt')
        model_path = os.path.join(model_dir, 'res10_300x300_ssd_iter_140000.caffemodel')

        if not os.path.exists(proto_path) or not os.path.exists(model_path):
            raise FileNotFoundError("DNN model files not found in 'models/' directory.")

        self.face_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    def capture_image(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Failed to capture image from webcam")

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            raise RuntimeError("Failed to encode image to JPEG")

        return jpeg.tobytes()

    def process_image(self, image_bytes):
        # Decode JPEG bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode JPEG bytes")

        # Detect faces using DNN
        faces = self.detect_faces_dnn(img)

        if not faces:
            raise RuntimeError("No faces detected in the image")

        # Get largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        (x, y, w, h) = largest_face

        # Draw bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop to largest face
        cropped_face = img[y:y + h, x:x + w]

        # Encode cropped face back to JPEG
        ret, processed_jpeg = cv2.imencode('.jpg', cropped_face)
        if not ret:
            raise RuntimeError("Failed to encode processed image to JPEG")

        return processed_jpeg.tobytes()


    def detect_faces_dnn(self, image):
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                x1, y1 = max(0, x1), max(0, y1)
                w_box, h_box = x2 - x1, y2 - y1
                faces.append((x1, y1, w_box, h_box))

        return faces



# Instantiate the camera
camera = Camera()

# Step 1: Capture image as JPEG bytes
jpeg_bytes = camera.capture_image()

# Step 2: Decode and display the original image
original_np = np.frombuffer(jpeg_bytes, np.uint8)
original_image = cv2.imdecode(original_np, cv2.IMREAD_COLOR)
cv2.imshow("Original Image", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 3: Process the image (detect + crop largest face)
processed_jpeg = camera.process_image(jpeg_bytes)

# Step 4: Decode and display the processed image
processed_np = np.frombuffer(processed_jpeg, np.uint8)
processed_image = cv2.imdecode(processed_np, cv2.IMREAD_COLOR)
cv2.imshow("Cropped Face", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()