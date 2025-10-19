import torch
from PIL import Image
from visualize_noise import make_overlay
from camera_stuff import Camera  # Make sure this points to your actual camera module
import numpy as np
import requests

class Masker:
    def __init__(self,
                 noise_path="models/noise.pt",
                 image_path="processed_face.jpg",
                 overlay_output_path="overlay_output.png",
                 difference_output_path="difference.png",
                 epsilon=0.03,
                 amplify_factor=10.0):
        self.noise_path = noise_path
        self.image_path = image_path
        self.overlay_output_path = overlay_output_path
        self.difference_output_path = difference_output_path
        self.epsilon = epsilon
        self.amplify_factor = amplify_factor
        self.camera = Camera()

    def capture_and_process_image(self):
        """Capture and process an image from the camera, then save it."""
        jpeg_bytes = self.camera.capture_image()
        processed_jpeg = self.camera.process_image(jpeg_bytes)

        with open(self.image_path, "wb") as f:
            f.write(processed_jpeg)

        print(f"[info] Saved processed face image to {self.image_path}")
        return Image.open(self.image_path)

    def load_noise_tensor(self):
        """Load and return the noise tensor from file."""
        obj = torch.load(self.noise_path, map_location="cpu")

        if isinstance(obj, dict):
            if "noise_bounded" in obj:
                print("[info] Auto-detected key 'noise_bounded'")
                return obj["noise_bounded"]
            else:
                raise KeyError("Expected key 'noise_bounded' not found in dict.")
        elif isinstance(obj, torch.Tensor):
            return obj
        else:
            raise ValueError("Unrecognized format in noise.pt")

    def generate_overlay(self, base_img, noise_tensor):
        """Generate the overlay image with noise applied."""
        return make_overlay(base_img, noise_tensor, epsilon=self.epsilon)

    def visualize_difference(self, original_img, noisy_img):
        """Create and return a grayscale image showing differences."""
        original = original_img.convert("RGB").resize(noisy_img.size)
        noisy = noisy_img.convert("RGB")

        orig_np = np.array(original).astype(np.float32)
        noisy_np = np.array(noisy).astype(np.float32)

        diff = np.abs(orig_np - noisy_np)
        diff_gray = diff.mean(axis=2)
        diff_enhanced = np.clip(diff_gray * self.amplify_factor, 0, 255).astype(np.uint8)

        return Image.fromarray(diff_enhanced, mode="L")
    
    # FOR EXTERNAL USAGE
    def get_clean_image(self):
        """
        Capture and return a clean (un-noised) image.
        
        Returns:
            PIL.Image: Clean processed image.
        """
        jpeg_bytes = self.camera.capture_image()
        processed_jpeg = self.camera.process_image(jpeg_bytes)

        with open(self.image_path, "wb") as f:
            f.write(processed_jpeg)

        print(f"[info] Captured clean image at {self.image_path}")
        return Image.open(self.image_path)

    # FOR EXTERNAL USAGE
    def get_dirty_image(self):
        """
        Capture an image and apply noise overlay.

        Returns:
            PIL.Image: Image with noise overlay applied.
        """
        # Capture and process
        base_img = self.get_clean_image()

        # Load noise
        noise_tensor = self.load_noise_tensor()

        # Apply overlay
        overlay_img = self.generate_overlay(base_img, noise_tensor)
        print("[info] Generated dirty (overlayed) image")

        return overlay_img
    
    def request_fooled_score(self, url="http://10.155.30.209:5000/inference"):
        print("IN REQUEST FOOLED SCORE")
        files = {
            "original": open(self.image_path, "rb"),
            "modified": open(self.overlay_output_path, "rb")
        }

        print("PRE RESPONSE")
        response = requests.post(url, files=files)
        print("POST RESPONSE")
        print(response.text)



    def run(self):
        """Run the full noise visualization pipeline."""
        # Step 1: Capture and save the face image
        base_img = self.capture_and_process_image()

        # Step 2: Load noise tensor
        noise_tensor = self.load_noise_tensor()

        # Step 3: Create overlay image
        overlay_img = self.generate_overlay(base_img, noise_tensor)
        overlay_img.save(self.overlay_output_path)
        print(f"[info] Saved overlay image to {self.overlay_output_path}")

        # Step 4: Visualize difference
        diff_img = self.visualize_difference(base_img, overlay_img)
        diff_img.save(self.difference_output_path)
        print(f"[info] Saved difference visualization to {self.difference_output_path}")

