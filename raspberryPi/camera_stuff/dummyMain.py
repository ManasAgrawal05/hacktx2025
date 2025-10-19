"""Very basic terminal version of the whole thing"""

import camera_stuff
from camera_stuff import Camera
import masker
from masker import Masker
import cv2
import torch
import cli_wrapper as clw

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

    dirty_img = masker_instance.generate_overlay(jpeg, masker_instance.load_noise_tensor())

    # Save dirty image
    dirty_img.save(dirty_path)
    print(f"[info] Saved dirty image to {dirty_path}")


"""
validate that the mask works by calling the facial recognition model on a
masked an unmasked image of the same person
"""


def validate_screwup(clean_image_path, dirty_image_path, device=None,
                     use_training_preprocess=False, hf_repo_id=None, hf_filename=None):
    """
    Compute FaceNet Euclidean distance between two image file paths.

    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        device (str or torch.device): 'cpu', 'cuda', or None for auto.
        use_training_preprocess (bool): Whether to use 224→160 preprocessing.
        hf_repo_id (str): Optional HuggingFace repo ID for model weights.
        hf_filename (str): Optional filename in the HuggingFace repo.

    Returns:
        float: Euclidean distance between FaceNet embeddings.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    print(f"[info] Using device: {device}")
    model = clw.load_facenet_from_hf(device, repo_id=hf_repo_id, filename=hf_filename)

    img1 = clw.Image.open(clean_image_path).convert("RGB")
    img2 = clw.Image.open(dirty_image_path).convert("RGB")

    if use_training_preprocess:
        print("[info] Using TRAINING preprocessing: 224x224 (PIL) → 160x160 (torch)")
        x1 = clw.preprocess_for_facenet_TRAINING_STYLE(img1).to(device)
        x2 = clw.preprocess_for_facenet_TRAINING_STYLE(img2).to(device)
    else:
        print("[info] Using ORIGINAL preprocessing: 160x160 (PIL)")
        x1 = clw.preprocess_for_facenet_ORIGINAL(img1).to(device)
        x2 = clw.preprocess_for_facenet_ORIGINAL(img2).to(device)

    with torch.no_grad():
        e1 = model(x1)
        e2 = model(x2)
        dist = torch.norm(e1 - e2, p=2).item()

    return dist


def main():
    # paths for test images
    clean_image_a_path = "clean_image_a.jpg"
    dirty_image_a_path = "dirty_image_a.jpg"
    clean_image_b_path = "clean_image_b.jpg"
    dirty_image_b_path = "dirty_image_b.jpg"

    # Capture and process images
    save_clean_and_dirty(clean_image_a_path, dirty_image_a_path, capture_and_process_image())
    save_clean_and_dirty(clean_image_b_path, dirty_image_b_path, capture_and_process_image())

    # show clean a as a reference point
    cv2.imshow("Image A", cv2.imread(clean_image_a_path))

    # show clean b and dirty b to compare the visual difference
    cv2.imshow("Image B Clean", cv2.imread(clean_image_b_path))
    cv2.imshow("Image B Dirty", cv2.imread(dirty_image_b_path))

    # Get recognition results between clean A and clean B, then clean A and dirty B
    print("Recognition between clean A and clean B: " +
          str(validate_screwup(clean_image_a_path, clean_image_b_path)))
    print("Recognition between clean A and dirty B: " +
          str(validate_screwup(clean_image_a_path, dirty_image_b_path)))

    # wait to and then close all images
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
