import socketio
import os
import time
import io
from base64 import b64encode
from PIL import Image
import cv2
from masker import Masker
from camera_stuff import Camera

# ======= Configuration =======
TUNNEL_URL = "https://theta-hour-hygiene-happening.trycloudflare.com"
IMAGE_SEND_DELAY = 2  # seconds between captures
# =============================

sio = socketio.Client()


@sio.event
def connect():
    print("✅ Connected to server")
    sio.emit('chat', {'msg': 'Pi image sender connected'})


@sio.on('server_message')
def on_server_message(data):
    print("📨 Server:", data.get('msg'))


@sio.event
def disconnect():
    print("❌ Disconnected from server")


# === Camera + Masker ===
def capture_and_process_image():
    camera = Camera()
    jpeg_bytes = camera.capture_image()
    processed_jpeg = camera.process_image(jpeg_bytes)
    processed_image = Image.open(io.BytesIO(processed_jpeg))
    return processed_image


def save_clean_image(clean_path, jpeg):
    jpeg.save(clean_path, format='JPEG')
    print(f"[info] Saved clean image to {clean_path}")


def save_dirty_image(dirty_path, jpeg):
    masker_instance = Masker()
    dirty_img = masker_instance.generate_overlay(jpeg, masker_instance.load_noise_tensor())
    dirty_img.save(dirty_path)
    print(f"[info] Saved dirty image to {dirty_path}")


def send_image(filepath):
    try:
        with open(filepath, "rb") as f:
            encoded = b64encode(f.read()).decode("utf-8")
        filename = os.path.basename(filepath)
        sio.emit("send_image", {"filename": filename, "content": encoded})
        print(f"📤 Sent image: {filename}")
        os.remove(filepath)
        print(f"🗑️ Deleted file: {filename}")
    except Exception as e:
        print(f"⚠️ Failed to send {filepath}: {e}")


def take_and_send_images():
    # 1️⃣ Anchor image
    anchor = capture_and_process_image()
    anchor_path = "anchor.jpg"
    save_clean_image(anchor_path, anchor)
    send_image(anchor_path)
    time.sleep(IMAGE_SEND_DELAY)

    # 2️⃣ Negative (clean)
    image_b = capture_and_process_image()
    negative_path = "negative.jpg"
    save_clean_image(negative_path, image_b)
    send_image(negative_path)
    time.sleep(IMAGE_SEND_DELAY)

    # 3️⃣ Positive (dirty)
    positive_path = "positive.jpg"
    save_dirty_image(positive_path, image_b)
    send_image(positive_path)

    print("✅ All images captured and sent!")


if __name__ == "__main__":
    print("🌐 Connecting to server...")
    sio.connect(TUNNEL_URL, transports=["websocket"])
    take_and_send_images()
    sio.wait()
