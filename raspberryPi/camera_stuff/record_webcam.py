import cv2
import os
import datetime


def record_video(output_dir="output_videos", camera_index=0):
    """
    Record video from the webcam until interrupted by Ctrl+C (or ESC key).
    Saves the recording as an MP4 file compatible with macOS QuickTime.

    Args:
        output_dir (str): Directory to save recorded videos.
        camera_index (int): Index of the webcam (default=0).
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"recording_{timestamp}.mp4")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[error] Could not open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # fallback if not reported
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use mp4v codec for macOS compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[info] Recording started. Press Ctrl+C or ESC to stop.")
    print(f"[info] Saving to: {output_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[warning] Frame capture failed, stopping.")
                break

            cv2.imshow("Recording (Press Ctrl+C or ESC to stop)", frame)
            out.write(frame)

            if cv2.waitKey(1) == 27:  # ESC key
                print("[info] ESC pressed. Stopping recording.")
                break

    except KeyboardInterrupt:
        print("\n[info] Keyboard interrupt received. Stopping recording.")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[info] Video saved to {output_path}")


if __name__ == "__main__":
    record_video()
