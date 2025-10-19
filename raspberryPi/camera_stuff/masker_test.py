from masker import Masker
import os

def test_get_clean_image():
    masker = Masker()
    clean_img = masker.get_clean_image()
    clean_img.show(title="Clean Image")  # This will open the image using default image viewer
    print("[test] get_clean_image passed")

def test_get_dirty_image():
    masker = Masker()
    dirty_img = masker.get_dirty_image()
    dirty_img.show(title="Dirty Image (with Noise Overlay)")
    print("[test] get_dirty_image passed")


if __name__ == "__main__":
    print("Starting tests...\n")
    test_get_clean_image()
    print("get dirty images")
    test_get_dirty_image()
    print("request fooled score")
