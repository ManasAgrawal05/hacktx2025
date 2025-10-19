from flask import Flask
from flask_socketio import SocketIO, emit
from base64 import b64decode
import os
import threading
from dummyMain import test_pre_processed  # import your function

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

REQUIRED_FILES = {
    "anchor": "anchor.jpg",
    "negative": "negative.jpg",
    "positive": "positive.jpg"
}


@app.route('/')
def index():
    return "WebSocket image server is running."


@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('server_message', {'msg': 'Welcome from server'})


@socketio.on('chat')
def handle_chat(data):
    print("Received chat:", data)
    emit('server_message', {'msg': f"Echo: {data['msg']}"})


@socketio.on('send_image')
def handle_image(data):
    filename = data.get('filename', 'uploaded.jpg')
    content_b64 = data.get('content')

    if not content_b64:
        emit('server_message', {'msg': 'No image content received.'})
        return

    try:
        image_bytes = b64decode(content_b64)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(save_path, 'wb') as f:
            f.write(image_bytes)

        print(f"✅ Saved image to {save_path}")
        emit('server_message', {'msg': f"Image '{filename}' received and saved."})

        # After saving, check for the three required files
        check_and_run_test()

    except Exception as e:
        print("❌ Error saving image:", e)
        emit('server_message', {'msg': f"Failed to save image: {str(e)}"})


def check_and_run_test():
    """Check if all required images exist and run the test when ready."""
    paths = {key: os.path.join(UPLOAD_FOLDER, fname)
             for key, fname in REQUIRED_FILES.items()}

    # If all three files exist, trigger the test in a background thread
    if all(os.path.exists(p) for p in paths.values()):
        print("🧠 All required images present. Running test_pre_processed...")

        def worker():
            try:
                result = test_pre_processed(
                    control_image_path=paths["anchor"],
                    clean_b_path=paths["negative"],
                    dirty_b_path=paths["positive"]
                )
                print("✅ test_pre_processed result:", result)
                socketio.emit('server_message', {'msg': f"Test completed: {result}"})
            except Exception as e:
                print("⚠️ Error running test_pre_processed:", e)
                socketio.emit('server_message', {'msg': f"Test failed: {str(e)}"})
                raise
#            finally:
#                # Clean up files so process can repeat
#                for p in paths.values():
#                    try:
#                        if os.path.exists(p):
#                            os.remove(p)
#                            print(f"🗑️ Deleted {os.path.basename(p)}")
#                    except Exception as ex:
#                        print(f"⚠️ Failed to delete {p}: {ex}")

        # Run in background so server stays responsive
        threading.Thread(target=worker, daemon=True).start()


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8888)
