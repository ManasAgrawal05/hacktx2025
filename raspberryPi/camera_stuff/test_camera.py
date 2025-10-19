import cv2


def list_camera_fps_modes(camera_index=0):
    """
    Try to detect the supported FPS (frames per second) modes of the webcam.

    Args:
        camera_index (int): Index of the webcam device.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[error] Could not open webcam.")
        return

    # Try common resolutions and check FPS for each
    common_resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    ]

    print("[info] Checking possible FPS modes...\n")
    for width, height in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Try to read one frame to make sure the mode is valid
        ret, _ = cap.read()
        if not ret:
            print(f"❌ {width}x{height}: not supported.")
            continue

        # Some webcams don’t report FPS correctly; we can estimate
        if fps == 0 or fps is None:
            fps = "unknown"

        print(f"✅ {width}x{height}: reported FPS = {fps}")

    cap.release()


if __name__ == "__main__":
    list_camera_fps_modes()
