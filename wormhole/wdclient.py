import socketio
from base64 import b64encode
import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ======= Configuration =======
TUNNEL_URL = "https://orange-composer-cancer-peripheral.trycloudflare.com"
WATCH_FOLDER = "./send_images"  # folder to watch for new images
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
# =============================

# Create folder if not exists
os.makedirs(WATCH_FOLDER, exist_ok=True)

sio = socketio.Client()


@sio.event
def connect():
    print("✅ Connected to server")
    sio.emit('chat', {'msg': 'Image watcher started'})


@sio.on('server_message')
def on_server_message(data):
    print("📨 Server:", data.get('msg'))


@sio.event
def disconnect():
    print("❌ Disconnected from server")

# === File Watcher Setup ===


class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        filepath = event.src_path
        _, ext = os.path.splitext(filepath)
        if ext.lower() not in IMAGE_EXTENSIONS:
            print(f"⏭️ Skipping non-image file: {filepath}")
            return

        # small delay to ensure file is ready
        time.sleep(0.3)

        try:
            with open(filepath, 'rb') as f:
                encoded = b64encode(f.read()).decode('utf-8')
            filename = os.path.basename(filepath)

            sio.emit('send_image', {
                'filename': filename,
                'content': encoded
            })
            print(f"📤 Sent image: {filename}")

            # delete file after successful send
            os.remove(filepath)
            print(f"🗑️ Deleted file: {filename}")

        except Exception as e:
            print(f"⚠️ Failed to send/delete image {filepath}: {e}")


def start_watching():
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_FOLDER, recursive=False)
    observer.start()
    print(f"👀 Watching folder: {WATCH_FOLDER}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    print("🌐 Connecting to server...")
    sio.connect(TUNNEL_URL, transports=['websocket'])

    # Start watching in the main thread
    start_watching()

    # Keep client alive
    sio.wait()
