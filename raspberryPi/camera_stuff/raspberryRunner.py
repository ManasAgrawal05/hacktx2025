import cv2
from masker import Masker
from camera_stuff import Camera
from PIL import Image
import time
import io
import keyboard
import torch
import numpy as np


def capture_image():
    """Capture a raw image as JPEG bytes."""
    camera = Camera()
    jpeg_bytes = camera.capture_image()
    return jpeg_bytes


def process_image(jpeg_bytes):
    """Process image and return PIL.Image and bounding box info."""
    camera = Camera()
    processed_jpeg, bbox = camera.process_image_with_bbox(jpeg_bytes)  # <-- modify camera_stuff
    processed_image = Image.open(io.BytesIO(processed_jpeg))
    return processed_image, bbox


def capture_and_process_image():
    return process_image(capture_image())


def save_clean_image(clean_path, jpeg):
    """Save clean image to disk."""
    jpeg.save(clean_path, format='JPEG')
    print(f"[info] Saved clean image to {clean_path}")


def save_dirty_image(dirty_path, jpeg):
    """Generate and save masked (dirty) version."""
    masker_instance = Masker()
    dirty_img = masker_instance.generate_overlay(jpeg, masker_instance.load_noise_tensor())
    dirty_img.save(dirty_path)
    print(f"[info] Saved dirty image to {dirty_path}")


def apply_mask_on_full_image(full_image, bbox, masker_instance):
    """
    Apply noise overlay only to the face region of the full image.
    bbox = (x, y, w, h)
    """
    noise = masker_instance.load_noise_tensor()
    overlay_region = masker_instance.generate_overlay(
        full_image.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])),
        noise
    )

    full_np = np.array(full_image.convert("RGB"))
    overlay_np = np.array(overlay_region.convert("RGB"))
    full_np[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = overlay_np

    return Image.fromarray(full_np)


def main():
    masker_instance = Masker()

    while True:
        if keyboard.is_pressed('q'):
            print("Exiting...")
            break

        elif keyboard.is_pressed(' '):
            print("Executing capture sequence...")

            # Capture the original image (unprocessed)
            anchor_raw = capture_image()
            anchor_raw_img = Image.open(io.BytesIO(anchor_raw))
            anchor_raw_img.save("../../wormhole/send_images/original_full.jpg")
            print("[info] Saved original full image")

            # Process for anchor
            anchor_proc, bbox = process_image(anchor_raw)
            save_clean_image("../../wormhole/send_images/anchor.jpg", anchor_proc)
            time.sleep(2)

            # Capture second image (B)
            image_b_raw = capture_image()
            image_b_raw_img = Image.open(io.BytesIO(image_b_raw))
            image_b_raw_img.save("../../wormhole/send_images/original_full_b.jpg")
            print("[info] Saved second original full image")

            image_b_proc, bbox_b = process_image(image_b_raw)
            save_clean_image("../../wormhole/send_images/negative.jpg", image_b_proc)
            time.sleep(2)
            save_dirty_image("../../wormhole/send_images/positive.jpg", image_b_proc)

            # Apply the noise mask onto the original full B image within its bounding box
            full_modified = apply_mask_on_full_image(image_b_raw_img, bbox_b, masker_instance)
            full_modified.save("../../wormhole/send_images/full_modified.jpg")
            print("[info] Saved full modified image with masked face region")

            time.sleep(1)


if __name__ == '__main__':
    main()
