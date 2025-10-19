"""Code to run on the pi, get the 3 images, and send them over"""

import cv2
from masker import Masker
from camera_stuff import Camera
from PIL import Image
import time
import io
import keyboard

# take an image and store as a jpeg
def capture_image():
    camera = Camera()
    jpeg_bytes = camera.capture_image()
    return jpeg_bytes

# process the image
def process_image(jpeg_bytes):
    camera = Camera()
    processed_jpeg = camera.process_image(jpeg_bytes)
    processed_image = Image.open(io.BytesIO(processed_jpeg))
    return processed_image

# take an image and store as a jpeg
def capture_and_process_image():
    return process_image(capture_image())

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


# get and save an anchor image + a clean and dirty image b
def main():
    while(True):
        # q for quit
        if keyboard.is_pressed('q'):
            print("Exiting...")
            break

        # space runs the capture system
        elif keyboard.is_pressed(' '):
            print("Executing program...")
            
            # get and save the anchor image
            anchor_large = capture_image()
            # save the main anchor image
            save_clean_image("../../wormhole/send_images/anchor_un_processed.jpg", anchor)
            anchor = process_image(anchor_large)
            save_clean_image("../../wormhole/send_images/anchor.jpg", anchor)
            time.sleep(2)

            # get image b
            image_b_large = capture_image()
            # save the large image b
            save_clean_image("../../wormhole/send_images/test_un_processed.jpg", image_b_large)
            image_b = process_image(image_b_large)
            save_clean_image("../../wormhole/send_images/negative.jpg", image_b)
            time.sleep(2)
            save_dirty_image("../../wormhole/send_images/positive.jpg", image_b)

main()

if __name__ == '__main__':
    main()
