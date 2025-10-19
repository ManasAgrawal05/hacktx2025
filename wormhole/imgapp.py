from flask import Flask
from flask_socketio import SocketIO, emit
from base64 import b64decode
import os

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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

        print(f"Saved image to {save_path}")
        emit('server_message', {'msg': f"Image '{filename}' received and saved."})

    except Exception as e:
        print("Error saving image:", e)
        emit('server_message', {'msg': f"Failed to save image: {str(e)}"})


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8888)
