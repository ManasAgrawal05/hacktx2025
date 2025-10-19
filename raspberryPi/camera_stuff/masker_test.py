from PIL import Image
from masker import Masker as NoiseVisualizer  # adjust if your class is in a different file
import requests

def test_get_clean_image():
    visualizer = NoiseVisualizer()
    clean_img = visualizer.get_clean_image()

    assert isinstance(clean_img, Image.Image), "Clean image is not a PIL.Image"
    clean_img.save("test_clean.png")
    print("[PASS] Clean image captured and saved as 'test_clean.png'")


def test_get_dirty_image():
    visualizer = NoiseVisualizer()
    dirty_img = visualizer.get_dirty_image()

    assert isinstance(dirty_img, Image.Image), "Dirty image is not a PIL.Image"
    dirty_img.save("test_dirty.png")
    print("[PASS] Dirty image captured and saved as 'test_dirty.png'")

def request_fooled_score(url="http://192.168.1.42:5000/inference"):
    files = {
        "original": open("original.jpg", "rb"),
        "modified": open("modified.jpg", "rb")
    }

    response = requests.post(url, files=files)
    print(response.text)



    test_get_dirty_image()
    print("\nAll tests completed.")
