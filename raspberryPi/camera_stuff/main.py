import torch
from PIL import Image
from visualize_noise import make_overlay
from camera_stuff import Camera  # Make sure this points to your actual camera module
import numpy as np


def visualize_difference(original_img, noisy_img, amplify_factor=10.0):
    """
    Compute and return a visualization of the pixel-wise absolute difference
    between the original and noisy images.

    Args:
        original_img (PIL.Image): Original image (before noise applied).
        noisy_img (PIL.Image): Image after noise applied.
        amplify_factor (float): Multiplier to enhance visibility of small differences.

    Returns:
        PIL.Image: Grayscale image showing pixel differences.
    """
    # Ensure both images are RGB and the same size
    original = original_img.convert("RGB").resize(noisy_img.size)
    noisy = noisy_img.convert("RGB")

    # Convert to numpy arrays
    orig_np = np.array(original).astype(np.float32)
    noisy_np = np.array(noisy).astype(np.float32)

    # Compute absolute difference
    diff = np.abs(orig_np - noisy_np)

    # Convert to grayscale using mean across RGB channels
    diff_gray = diff.mean(axis=2)

    # Amplify the difference for visibility
    diff_enhanced = np.clip(diff_gray * amplify_factor, 0, 255).astype(np.uint8)

    # Convert back to image
    return Image.fromarray(diff_enhanced, mode="L")

# Step 1: Capture and process the face image
camera = Camera()
jpeg_bytes = camera.capture_image()
processed_jpeg = camera.process_image(jpeg_bytes)

# Step 2: Save the processed face image temporarily
image_path = "processed_face.jpg"
with open(image_path, "wb") as f:
    f.write(processed_jpeg)

# Step 3: Load the noise tensor
noise_path = "models/noise.pt"
obj = torch.load(noise_path, map_location="cpu")

# Optional: use a key if your .pt file is a dict
if isinstance(obj, dict):
    if "noise_bounded" in obj:
        print("[info] Auto-detected key 'noise_bounded'")
        noise_tensor = obj["noise_bounded"]
    else:
        raise KeyError("Expected key 'noise_bounded' not found.")
elif isinstance(obj, torch.Tensor):
    noise_tensor = obj
else:
    raise ValueError("Unrecognized format in noise.pt")

# Step 4: Load the processed face image
base_img = Image.open(image_path)

# Step 5: Generate the overlay image
epsilon = 0.03  # Or whatever value you want
overlay_img = make_overlay(base_img, noise_tensor, epsilon=epsilon)

# Step 6: Save the overlay image
overlay_img.save("overlay_output.png")
print("[info] Saved overlay image to overlay_output.png")

diff_img = visualize_difference(base_img, overlay_img)
diff_img.save("difference.png")
print("[info] Saved difference visualization to difference.png")
