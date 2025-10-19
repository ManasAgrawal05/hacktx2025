"""Wrapper class for our required camera functionality (headless version)"""

import cv2
import numpy as np
import os


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

        # Crop to largest face and resize
        cropped_face = crop_face_to_square_and_resize(
            img, largest_face, output_size=250, margin=0.0, pad_color=(0, 0, 0)
        )

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


def crop_face_to_square_and_resize(img, box, output_size=250, margin=0.0, pad_color=(0, 0, 0)):
    x, y, w, h = map(int, box)
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
    cx = x + w // 2
    cy = y + h // 2
    side = max(w, h)
    if margin:
        side = int(round(side * (1.0 + margin)))
    side = max(1, side)

    half = side // 2
    x1 = cx - half
    y1 = cy - half
    x2 = x1 + side
    y2 = y1 + side

    crop_x1 = max(0, x1)
    crop_y1 = max(0, y1)
    crop_x2 = min(w_img, x2)
    crop_y2 = min(h_img, y2)

    cropped = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    pad_left = crop_x1 - x1 if x1 < 0 else 0
    pad_top = crop_y1 - y1 if y1 < 0 else 0
    pad_right = x2 - crop_x2 if x2 > w_img else 0
    pad_bottom = y2 - crop_y2 if y2 > h_img else 0

    pad_left, pad_top, pad_right, pad_bottom = map(int, [pad_left, pad_top, pad_right, pad_bottom])

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

    cur_h, cur_w = cropped.shape[:2]
    if (cur_h != side) or (cur_w != side):
        diff_w = side - cur_w
        diff_h = side - cur_h

        if diff_w > 0 or diff_h > 0:
            cropped = cv2.copyMakeBorder(
                cropped,
                top=0 if diff_h <= 0 else diff_h,
                bottom=0,
                left=0,
                right=0 if diff_w <= 0 else diff_w,
                borderType=cv2.BORDER_CONSTANT,
                value=pad_color
            )
        else:
            cropped = cropped[:side, :side]

    final = cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return final


if __name__ == "__main__":
    camera = Camera()

    jpeg_bytes = camera.capture_image()
    processed_jpeg = camera.process_image(jpeg_bytes)

    # Optionally save both images
    with open("original.jpg", "wb") as f:
        f.write(jpeg_bytes)

    with open("processed_face.jpg", "wb") as f:
        f.write(processed_jpeg)

    print("Images saved: original.jpg and processed_face.jpg")
