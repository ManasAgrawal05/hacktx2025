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
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop to largest face
        # cropped_face = img[y:y + h, x:x + w]
        cropped_face = crop_face_to_square_and_resize(img, largest_face, output_size=250, margin=0.0, pad_color=(0,0,0))
    


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

def crop_face_to_square_and_resize(img, box, output_size=250, margin=0.0, pad_color=(0,0,0)):
    """
    img: BGR image (numpy array)
    box: (x, y, w, h) bounding box of detected face (integers)
    output_size: final size (int) for both width and height
    margin: fraction of the max(w,h) to add as padding around the face (e.g. 0.2 = 20%)
    pad_color: BGR tuple used to pad when the square extends beyond image boundary
    """
    x, y, w, h = map(int, box)

    # Scale the box by 2.2x while keeping center the same
    scale = 2.2
    cx = x + w // 2
    cy = y + h // 2
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    x = cx - new_w // 2
    y = cy - new_h // 2
    w = new_w
    h = new_h

    h_img, w_img = img.shape[:2]

    # center and desired square side length
    cx = x + w // 2
    cy = y + h // 2
    side = max(w, h)
    if margin:
        side = int(round(side * (1.0 + margin)))

    # make sure side at least 1
    side = max(1, side)

    # compute square coordinates (may go out of image bounds)
    half = side // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + side
    y2 = y1 + side

    # Calculate intersection with image
    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w_img, x2)
    crop_y2 = min(h_img, y2)

    # Crop what we can from the image
    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    # Compute padding required on each side to make it exactly `side x side`
    pad_left = crop_x1 - x1 if x1 < 0 else 0
    pad_top  = crop_y1 - y1 if y1 < 0 else 0
    pad_right = x2 - crop_x2 if x2 > w_img else 0
    pad_bottom = y2 - crop_y2 if y2 > h_img else 0

    # OpenCV wants padding amounts as ints
    pad_left = int(pad_left)
    pad_top = int(pad_top)
    pad_right = int(pad_right)
    pad_bottom = int(pad_bottom)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        cropped = cv2.copyMakeBorder(
            cropped,
            top=pad_top,
            bottom=pad_bottom,
            left=pad_left,
            right=pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color
        )

    # At this point cropped.shape should be (side, side, channels) or very close.
    # If due to rounding it's off by 1 pixel, fix it by cropping or padding minimally.
    cur_h, cur_w = cropped.shape[:2]
    if (cur_h != side) or (cur_w != side):
        # If one dimension is too small, pad; if too large, crop.
        diff_w = side - cur_w
        diff_h = side - cur_h

        left = right = top = bottom = 0
        if diff_w > 0:
            # pad extra pixels on right
            right = diff_w
        elif diff_w < 0:
            # crop extra from right
            cropped = cropped[:, :side]

        if diff_h > 0:
            bottom = diff_h
        elif diff_h < 0:
            cropped = cropped[:side, :]

        if diff_w > 0 or diff_h > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_color
            )

    # Final resize to output_size x output_size
    final = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    return final


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