"""Code to run on the pi, get the 3 images, and send them over"""

import cv2
from masker import Masker
from camera_stuff import Camera
from PIL import Image
import time
import io

# take an image and store as a jpeg
def capture_and_process_image():
    camera = Camera()
    jpeg_bytes = camera.capture_image()
    processed_jpeg = camera.process_image(jpeg_bytes)
    processed_image = Image.open(io.BytesIO(processed_jpeg))
    return processed_image

# save a clean version of the image
def save_clean_image(clean_path, jpeg):
    masker_instance = Masker()

    # Save clean image
    jpeg.save(clean_path, format='JPEG')
    print(f"[info] Saved clean image to {clean_path}")


# save a dirty version of the image
def save_dirty_image(dirty_path, jpeg):
    masker_instance = Masker()

    dirty_img = masker_instance.generate_overlay(jpeg, masker_instance.load_noise_tensor())

    # Save dirty image
    dirty_img.save(dirty_path)
    print(f"[info] Saved dirty image to {dirty_path}")

# get and save an anchor image + a clean and dirty image b
def main():
    # get and save the anchor image
    anchor = capture_and_process_image()
    save_clean_image("../../wormhole/send_images/anchor.jpg", anchor)
    time.sleep(2)

    # get image b
    image_b = capture_and_process_image()
    save_clean_image("../../wormhole/send_images/negative.jpg", image_b)
    time.sleep(2)
    save_dirty_image("../../wormhole/send_images/positive.jpg", image_b)

main()





