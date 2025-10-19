"""Very basic terminal version of the whole thing"""

import camera_stuff
from camera_stuff import Camera
import masker
from masker import Masker
import cv2

# take image and store to given filepath
def capture_and_process_image():
    camera = Camera()
    jpeg_bytes = camera.capture_image()
    processed_jpeg = camera.process_image(jpeg_bytes)
    return processed_jpeg

# make a clean and dirty version of the image and save to given filepaths
def save_clean_and_dirty(clean_path, dirty_path, jpeg):
    masker_instance = Masker()
    
    # Save clean image
    with open(clean_path, "wb") as f:
        f.write(jpeg)
    print(f"[info] Saved clean image to {clean_path}")
    
    # Load clean image and generate dirty image
    clean_img = masker_instance.get_clean_image(clean_path)
    dirty_img = masker_instance.get_dirty_image()
    
    # Save dirty image
    dirty_img.save(dirty_path)
    print(f"[info] Saved dirty image to {dirty_path}")

"""
validate that the mask works by calling the facial recognition model on a 
masked an unmasked image of the same person
"""
def validate_screwup(clean_image, dirty_image):
    pass


def main():
    # paths for test images
    clean_image_a_path = "clean_image_a.jpg"
    dirty_image_a_path = "dirty_image_a.jpg"
    clean_image_b_path = "clean_image_b.jpg"
    dirty_image_b_path = "dirty_image_b.jpg"

    # Capture and process images
    jpeg_a = save_clean_and_dirty(clean_image_a_path, dirty_image_a_path, capture_and_process_image())
    jpeg_b = save_clean_and_dirty(clean_image_b_path, dirty_image_b_path, capture_and_process_image())

    # show clean a as a reference point
    cv2.imshow("Image A", cv2.imread(clean_image_a_path))

    # show clean b and dirty b to compare the visual difference
    cv2.imshow("Image B Clean", cv2.imread(clean_image_b_path))
    cv2.imshow("Image B Dirty", cv2.imread(dirty_image_b_path))

    # Get recognition results between clean A and clean B, then clean A and dirty B
    print("Recognition between clean A and clean B: " + str(validate_screwup(clean_image_a_path, clean_image_b_path))) 
    print("Recognition between clean A and dirty B: " + str(validate_screwup(clean_image_a_path, dirty_image_b_path)))

    # wait to and then close all images
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    






